"""The agent loop: model turn -> tool calls -> results -> repeat.

The Messages API is stateless, so this module owns the conversation history:
it appends each assistant response and each batch of tool results, then calls
the model again until the model answers with text instead of asking for tools.

Two rules that are easy to get wrong and are load-bearing here: append the
*whole* ``response.content`` (not just the text) so ``tool_use`` and thinking
blocks survive the round trip, and return every tool result in a single user
message so parallel tool calls keep working.

Everything the loop does is narrated to stdout -- turn number, tool name,
truncated input, one-line result preview -- so a run is readable as it happens.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import anthropic

from agent.llm import LLM
from agent.prompts import build_system_prompt, build_task_message
from agent.tools import TOOLS, dispatch, set_repo_root

# Safety valve: stop after this many model turns even if the agent is still
# asking for tools, so a confused run can't spin forever.
MAX_TURNS = 40

# One retry on a transient API failure, on top of the SDK's own retries.
RETRY_ATTEMPTS = 2
RETRY_BASE_DELAY = 2.0

MAX_INPUT_PREVIEW = 160
MAX_RESULT_PREVIEW = 110
MAX_TEXT_PREVIEW = 500


# --------------------------------------------------------------------------
# Console formatting
# --------------------------------------------------------------------------


def _unicode_ok() -> bool:
    """Return True if stdout can encode the box-drawing glyphs."""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "─·→✓✗".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


if _unicode_ok():
    _RULE, _CALL, _ARROW, _OK, _BAD = "─", "·", "→", "✓", "✗"
else:  # pragma: no cover - depends on the terminal
    _RULE, _CALL, _ARROW, _OK, _BAD = "-", "*", "->", "[ok]", "[!]"

_WIDTH = 72


def _rule() -> str:
    return _RULE * _WIDTH


def _say(text: str = "") -> None:
    """Print a line and flush, so output stays in step with the work."""
    print(text, flush=True)


def _flatten(text: str) -> str:
    """Collapse whitespace so a preview stays on one line."""
    return " ".join(text.split())


def _clip(text: str, limit: int) -> str:
    """Truncate to ``limit`` characters with an ellipsis marker."""
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _format_input(tool_input: Any) -> str:
    """Render a tool's input compactly, clipping long values.

    Long string arguments (a whole file passed to write_file, say) are cut
    down before rendering so one call can't flood the console.
    """
    if not isinstance(tool_input, dict):
        return _clip(_flatten(str(tool_input)), MAX_INPUT_PREVIEW)

    parts = []
    for key, value in tool_input.items():
        if isinstance(value, str):
            shown = _clip(_flatten(value), 60)
            parts.append(f"{key}={json.dumps(shown, ensure_ascii=False)}")
        else:
            parts.append(f"{key}={json.dumps(value, ensure_ascii=False, default=str)}")
    return _clip(", ".join(parts), MAX_INPUT_PREVIEW)


def _format_result(result: str) -> str:
    """One-line preview of a tool result, with a line count when it is long."""
    lines = result.splitlines() or [""]
    preview = _clip(_flatten(lines[0]), MAX_RESULT_PREVIEW)
    if len(lines) > 1:
        preview += f"  [{len(lines)} lines]"
    return preview


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def run(
    repo: Path,
    task: str,
    llm: LLM | None = None,
    max_turns: int = MAX_TURNS,
    extra_instructions: str | None = None,
) -> str:
    """Run the agent against a repository until the task is done.

    Args:
        repo: Repository root the agent is allowed to read and modify. Bound
            to `agent.tools` here, so every tool is confined to it.
        task: Natural-language description of what the agent should do.
        llm: Model wrapper to use. A default ``LLM()`` is constructed if
            omitted.
        max_turns: Hard cap on model turns before giving up.
        extra_instructions: Project-specific rules appended to the system
            prompt.

    Returns:
        The agent's final text response.

    Raises:
        RuntimeError: If ``max_turns`` is reached with the agent still
            requesting tools.
        anthropic.APIError: If a request fails and the retry also fails.
    """
    set_repo_root(repo)
    llm = llm or LLM()
    system = build_system_prompt(extra_instructions)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": build_task_message(repo, task)}
    ]

    _say(_rule())
    _say(f"  converseai  {_ARROW}  {llm.model}  (effort={llm.effort})")
    _say(f"  repo: {repo}")
    _say(f"  task: {_clip(_flatten(task), 60)}")
    _say(_rule())

    started = time.monotonic()
    tokens_in = tokens_out = 0
    tool_calls = 0

    for turn in range(1, max_turns + 1):
        # Header first, so a slow call or a retry notice lands under the turn
        # it belongs to rather than above it.
        _say(f"\n[turn {turn}/{max_turns}]")
        response = _complete_with_retry(llm, messages, system)

        tokens_in += response.usage.input_tokens
        tokens_out += response.usage.output_tokens

        _log_thinking(response)

        # Append the whole content: dropping tool_use or thinking blocks here
        # makes the next request invalid.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            _say(f"  {_BAD} the model declined this request")
            _log_footer(time.monotonic() - started, turn, tool_calls, tokens_in, tokens_out)
            return "The model declined to complete this task."

        if response.stop_reason != "tool_use":
            if response.stop_reason == "max_tokens":
                _say(f"  {_BAD} response hit max_tokens and may be cut short")
            final = _final_text(response)
            _log_summary(final)
            _log_footer(time.monotonic() - started, turn, tool_calls, tokens_in, tokens_out)
            return final

        _log_text(response)
        results = _handle_tool_use(response)
        tool_calls += len(results)
        messages.append({"role": "user", "content": results})

    _say("")
    _say(_rule())
    _say(f"  {_BAD} stopped: hit the {max_turns}-turn limit with the agent still")
    _say("     calling tools. The task is probably too large for one run, or")
    _say("     the agent is stuck retrying something that keeps failing.")
    _say("     Re-run with a narrower task, or raise --max-turns.")
    _say(_rule())
    raise RuntimeError(f"agent did not finish within {max_turns} turns")


def _complete_with_retry(
    llm: LLM, messages: list[dict[str, Any]], system: str
) -> anthropic.types.Message:
    """Call the model, retrying once on a transient failure.

    The SDK already retries 429s and 5xxs internally; this is the outer layer
    that survives a failure surviving *that*. Client errors (400, 401, 404)
    are not retried -- they will fail identically the second time.

    Args:
        llm: The model wrapper.
        messages: Conversation history so far.
        system: System prompt.

    Returns:
        The ``Message`` returned by the API.

    Raises:
        anthropic.APIError: If the final attempt fails.
    """
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return llm.complete(messages=messages, tools=TOOLS, system=system)
        except (
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        ) as exc:
            if attempt == RETRY_ATTEMPTS:
                _say(f"  {_BAD} API error, retries exhausted: {exc}")
                raise
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            _say(f"  {_BAD} API error ({type(exc).__name__}); retrying in {delay:g}s")
            time.sleep(delay)
        except anthropic.APIStatusError as exc:
            _say(f"  {_BAD} API error {exc.status_code}: {exc.message}")
            raise

    raise AssertionError("unreachable")  # pragma: no cover


def _handle_tool_use(response: Any) -> list[dict[str, Any]]:
    """Execute every ``tool_use`` block in a response, narrating as it goes.

    Tools are already bound to the repo root by `agent.tools.set_repo_root`,
    so nothing is passed through here.

    Args:
        response: The ``Message`` returned by `agent.llm.LLM.complete`.

    Returns:
        One ``tool_result`` block per ``tool_use`` block, in the same order,
        each carrying the matching ``tool_use_id``. The API rejects the next
        request if any ``tool_use`` id is left without a result, so failures
        are reported as results with ``is_error: true`` rather than skipped.
    """
    results: list[dict[str, Any]] = []

    for block in response.content:
        if block.type != "tool_use":
            continue

        _say(f"  {_CALL} {block.name}({_format_input(block.input)})")

        started = time.monotonic()
        result = dispatch(block.name, block.input)
        elapsed = time.monotonic() - started

        failed = result.startswith("Error:")
        marker = _BAD if failed else _OK
        timing = f"  ({elapsed:.1f}s)" if elapsed >= 0.5 else ""
        _say(f"    {marker} {_format_result(result)}{timing}")

        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
                "is_error": failed,
            }
        )

    return results


def _log_thinking(response: Any) -> None:
    """Print a one-line preview of the model's summarized thinking, if any."""
    for block in response.content:
        if block.type == "thinking" and getattr(block, "thinking", ""):
            _say(f"  {_clip(_flatten(block.thinking), 100)}")
            return


def _log_text(response: Any) -> None:
    """Print any assistant text that accompanied a batch of tool calls."""
    text = _final_text(response)
    if text:
        _say(f"  {_clip(_flatten(text), MAX_TEXT_PREVIEW)}")


def _log_summary(final: str) -> None:
    """Print the agent's closing summary in full."""
    _say("")
    _say(_rule())
    _say(f"  {_OK} done")
    _say(_rule())
    _say(final.strip() or "(the agent returned no summary)")


def _log_footer(
    elapsed: float, turns: int, tool_calls: int, tokens_in: int, tokens_out: int
) -> None:
    """Print the run's totals."""
    _say(_rule())
    _say(
        f"  {_plural(turns, 'turn')}  {_CALL}  {_plural(tool_calls, 'tool call')}"
        f"  {_CALL}  {elapsed:.0f}s  {_CALL}  "
        f"{tokens_in:,} in / {tokens_out:,} out tokens"
    )
    _say(_rule())


def _plural(count: int, noun: str) -> str:
    """Format ``count`` with ``noun``, pluralised with a trailing s."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _final_text(response: Any) -> str:
    """Join the text blocks of a response into a single string.

    Args:
        response: The ``Message`` returned by `agent.llm.LLM.complete`.

    Returns:
        The concatenated text, ignoring thinking and tool_use blocks.
    """
    return "\n".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
