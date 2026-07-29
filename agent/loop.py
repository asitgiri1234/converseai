"""The agent loop: model turn -> tool calls -> results -> repeat.

The chat completions API is stateless, so this module owns the conversation
history: it appends each assistant reply and each tool result, then calls the
model again until the model answers with text instead of asking for tools.

Two rules that are easy to get wrong and are load-bearing here: the assistant
message must be appended *with its ``tool_calls``* intact, and every tool call
needs exactly one ``role: "tool"`` reply carrying the matching
``tool_call_id`` -- Groq rejects the next request otherwise.

Everything the loop does is narrated to stdout -- turn number, tool name,
truncated input, one-line result preview -- so a run is readable as it happens.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import groq

from agent import memory as repo_memory
from agent.llm import LLM
from agent.prompts import SUMMARY_MARKER, build_system_prompt, build_task_message
from agent.tools import TOOLS, dispatch, set_repo_root

# Safety valve: stop after this many model turns even if the agent is still
# asking for tools, so a confused run can't spin forever.
MAX_TURNS = 40

# Retries on transient API failures, on top of the SDK's own. Groq's free
# tier is metered per minute and an agent replays its whole history every
# turn, so rate limits are routine rather than exceptional here.
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 2.0
MAX_RETRY_DELAY = 65.0

MAX_INPUT_PREVIEW = 160
MAX_RESULT_PREVIEW = 110
MAX_TEXT_PREVIEW = 100
MAX_TEXT_LINES = 16

# The agent works in phases (see agent.prompts) and its PLAN phase is text
# with no tool call, which would otherwise read as "finished". So a text-only
# turn ends the run only once it carries the summary marker; before that the
# agent gets nudged back to work, at most this many times so a model that
# never emits the marker still terminates.
# File contents dominate an agent's history: one 120-line file read twice is
# most of a small tier's per-minute budget. Older tool results are elided from
# the *request* (history itself is kept intact) so a long run stays under the
# limit. The elision is worded so the model knows it can re-read. Three keeps
# the latest read of the file being edited plus its immediate context; four
# was measured to let two full copies of one controller ride along and 413
# an 8k-TPM tier.
KEEP_TOOL_RESULTS = 3
ELIDE_OVER_CHARS = 400
ELIDED = "[earlier result elided to save context -- re-read the file if you still need it]"

MAX_NUDGES = 3
NUDGE = (
    "That was not a summary, so the task is not finished. Continue with the "
    "next phase now, using tools. When the work is genuinely complete and "
    f"verified, reply with your {SUMMARY_MARKER} section."
)


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
    use_intel: bool = True,
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
        use_intel: Pre-scan the repository with `agent.intel` and embed the
            summary in the first message. Costs zero API tokens to build;
            disable (--no-intel) to fall back to pure tool-driven
            exploration.

    Returns:
        The agent's final text response.

    Raises:
        RuntimeError: If ``max_turns`` is reached with the agent still
            requesting tools.
        groq.APIError: If a request fails and the retry also fails.
    """
    set_repo_root(repo)
    llm = llm or LLM()
    system = build_system_prompt(extra_instructions)

    mem = None
    intel_block = None
    if use_intel:
        # A scan failure must never kill the run; exploration still works
        # the old way without it.
        try:
            mem = repo_memory.build_memory(repo)
            intel_block = mem.intel.render()
            graph_block = mem.render_block()
            if graph_block:
                intel_block += "\n" + graph_block
        except Exception as exc:  # noqa: BLE001 - degrade, don't die
            mem = None
            _say(f"  {_BAD} repo intelligence scan failed ({exc}); continuing without it")

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": build_task_message(repo, task, intel_block)}
    ]

    _say(_rule())
    _say(f"  converseai  {_ARROW}  groq / {llm.model}  (temp={llm.temperature})")
    _say(f"  repo: {repo}")
    _say(f"  task: {_clip(_flatten(task), 60)}")
    if mem:
        graph_edges = len(mem.graph.edges)
        _say(
            f"  intel: {mem.intel.primary_language} / "
            f"{', '.join(mem.intel.frameworks) or 'no framework'} / "
            f"{mem.intel.architecture} -- "
            f"{mem.intel.files_scanned} files, "
            f"{len(mem.intel.symbols)} symbols, "
            f"{len(mem.intel.endpoints)} endpoints, "
            f"{graph_edges} graph edges"
        )
    _say(_rule())

    started = time.monotonic()
    tokens_in = tokens_out = 0
    tool_calls = 0
    nudges = 0

    for turn in range(1, max_turns + 1):
        # Header first, so a slow call or a retry notice lands under the turn
        # it belongs to rather than above it.
        _say(f"\n[turn {turn}/{max_turns}]")
        response = _complete_with_retry(llm, messages, system, mem)

        choice = response.choices[0]
        message = choice.message
        if response.usage:
            tokens_in += response.usage.prompt_tokens
            tokens_out += response.usage.completion_tokens

        _log_reasoning(message)

        # The assistant turn must go back with its tool_calls attached, or the
        # role="tool" replies below have nothing to answer.
        messages.append(_assistant_message(message))

        if choice.finish_reason == "length":
            _say(f"  {_BAD} response hit the token limit and may be cut short")

        if not message.tool_calls:
            final = (message.content or "").strip()

            # A phase like PLAN ends a turn with text and no tool call. That
            # is mid-run, not done, so only the summary marker stops the loop.
            if SUMMARY_MARKER not in final and nudges < MAX_NUDGES:
                nudges += 1
                _log_text(final)
                _say(f"  {_CALL} no {SUMMARY_MARKER} yet -- continuing")
                messages.append({"role": "user", "content": NUDGE})
                continue

            _log_summary(final)
            if mem:
                _say(f"  memory: {mem.stats_line()}")
            _log_footer(time.monotonic() - started, turn, tool_calls, tokens_in, tokens_out)
            return final

        _log_text((message.content or "").strip())
        results = _handle_tool_calls(message, mem)
        tool_calls += len(results)
        messages.extend(results)

    _say("")
    _say(_rule())
    _say(f"  {_BAD} stopped: hit the {max_turns}-turn limit with the agent still")
    _say("     calling tools. The task is probably too large for one run, or")
    _say("     the agent is stuck retrying something that keeps failing.")
    _say("     Re-run with a narrower task, or raise --max-turns.")
    _say(_rule())
    raise RuntimeError(f"agent did not finish within {max_turns} turns")


def _complete_with_retry(
    llm: LLM, messages: list[dict[str, Any]], system: str, mem: Any = None
) -> Any:
    """Call the model, retrying once on a transient failure.

    The SDK already retries 429s and 5xxs internally; this is the outer layer
    that survives a failure surviving *that*. Client errors (400, 401, 404)
    are not retried -- they will fail identically the second time.

    Args:
        llm: The model wrapper.
        messages: Conversation history so far.
        system: System prompt.

    Returns:
        The ``ChatCompletion`` returned by the API.

    Raises:
        groq.APIError: If the final attempt fails.
    """
    payload = _elide_old_results(messages, mem)

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return llm.complete(messages=payload, tools=TOOLS, system=system)
        except (
            groq.APIConnectionError,
            groq.RateLimitError,
            groq.InternalServerError,
        ) as exc:
            if attempt == RETRY_ATTEMPTS:
                _say(f"  {_BAD} API error, retries exhausted: {exc}")
                raise
            delay = _retry_delay(exc, attempt)
            _say(f"  {_BAD} API error ({type(exc).__name__}); retrying in {delay:g}s")
            time.sleep(delay)
        except groq.BadRequestError as exc:
            # `tool_use_failed` means the model's own tool-call JSON was
            # malformed or truncated -- a generation artifact, not a bad
            # request from us, so it is worth another sample.
            if "tool_use_failed" not in str(exc) or attempt == RETRY_ATTEMPTS:
                _say(f"  {_BAD} API error {exc.status_code}: {exc.message}")
                raise
            _say(f"  {_BAD} model emitted malformed tool-call JSON; resampling")
            time.sleep(RETRY_BASE_DELAY)
        except groq.APIStatusError as exc:
            _say(f"  {_BAD} API error {exc.status_code}: {exc.message}")
            raise

    raise AssertionError("unreachable")  # pragma: no cover


def _elide_old_results(
    messages: list[dict[str, Any]], mem: Any = None
) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with stale tool results compressed.

    Two tiers. When repository memory knows which file an old ``read_file``
    result showed, the full contents are replaced with that file's *current
    module summary* -- the model keeps a compressed understanding (exports,
    routes, imports, DB calls) instead of a blank. Anything memory cannot
    attribute falls back to the generic elision notice.

    Only the *content* is replaced, never the message itself: every
    ``role: "tool"`` entry has to keep its ``tool_call_id`` so it still
    answers the assistant call that requested it, or the request is rejected.

    Args:
        messages: The live history. Not mutated.
        mem: `agent.memory.RepositoryMemory`, or None for generic elision.

    Returns:
        A shallow copy safe to send as the request payload.
    """
    tool_positions = [
        i for i, entry in enumerate(messages) if entry.get("role") == "tool"
    ]
    recent = set(tool_positions[-KEEP_TOOL_RESULTS:])

    trimmed: list[dict[str, Any]] = []
    for i, entry in enumerate(messages):
        stale = (
            entry.get("role") == "tool"
            and i not in recent
            and len(entry.get("content") or "") > ELIDE_OVER_CHARS
        )
        if not stale:
            trimmed.append(entry)
            continue

        replacement = ELIDED
        if mem:
            call_id = entry.get("tool_call_id", "")
            path = mem.tool_call_file.get(call_id)
            summary = mem.summary_for(path) if path else None
            if summary:
                replacement = (
                    f"[compressed -- current summary of this file: {summary}. "
                    "Re-read the file only if you need its exact contents.]"
                )
                mem.compressed_ids.add(call_id)
        trimmed.append({**entry, "content": replacement})
    return trimmed


def _retry_delay(exc: Exception, attempt: int) -> float:
    """How long to wait before retrying ``exc``.

    A rate-limited response says exactly when the budget frees up, and that
    beats guessing: exponential backoff from 2s would retry long before a
    per-minute window has rolled over. Falls back to exponential backoff when
    the header is absent or unparseable.

    Args:
        exc: The exception that triggered the retry.
        attempt: 1-based attempt number that just failed.

    Returns:
        Seconds to sleep, capped at ``MAX_RETRY_DELAY``.
    """
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}).get("retry-after") if response else None
    if header:
        try:
            # A small margin, since the window rolls over slightly after the
            # server's own estimate.
            return min(float(header) + 1.0, MAX_RETRY_DELAY)
        except ValueError:
            pass
    return min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), MAX_RETRY_DELAY)


def _assistant_message(message: Any) -> dict[str, Any]:
    """Convert an assistant reply back into a plain history entry.

    Rebuilt field by field rather than dumping the SDK model, so nothing
    provider-specific (``reasoning``, ``refusal``, null ``function_call``)
    leaks back into the next request.

    Args:
        message: ``response.choices[0].message``.

    Returns:
        A ``role: "assistant"`` message, carrying ``tool_calls`` when present.
    """
    entry: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        entry["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ]
    return entry


def _handle_tool_calls(message: Any, mem: Any = None) -> list[dict[str, Any]]:
    """Execute every tool call on an assistant message, narrating as it goes.

    Tools are already bound to the repo root by `agent.tools.set_repo_root`,
    so nothing is passed through here.

    Args:
        message: ``response.choices[0].message``, with ``tool_calls`` set.
        mem: `agent.memory.RepositoryMemory` to notify of reads and writes,
            or None when running without the pre-scan.

    Returns:
        One ``role: "tool"`` message per call, in order, each carrying the
        matching ``tool_call_id``. Groq rejects the next request if any call
        is left unanswered, so failures -- including arguments that are not
        valid JSON -- come back as results rather than being skipped.
    """
    results: list[dict[str, Any]] = []

    for call in message.tool_calls:
        name = call.function.name
        raw_args = call.function.arguments or "{}"

        # Arguments arrive as a JSON string the model generated, so they can
        # be malformed. Report that back instead of crashing the run.
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            _say(f"  {_CALL} {name}({_clip(_flatten(raw_args), 80)})")
            _say(f"    {_BAD} arguments were not valid JSON: {exc}")
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": (
                        f"Error: arguments for {name} were not valid JSON "
                        f"({exc}). Re-issue the call with a valid JSON object."
                    ),
                }
            )
            continue

        _say(f"  {_CALL} {name}({_format_input(arguments)})")

        started = time.monotonic()
        result = dispatch(name, arguments)
        elapsed = time.monotonic() - started

        failed = result.startswith("Error:")
        marker = _BAD if failed else _OK
        timing = f"  ({elapsed:.1f}s)" if elapsed >= 0.5 else ""
        _say(f"    {marker} {_format_result(result)}{timing}")

        # Feed repository memory, so summaries stay current and old results
        # can later be compressed to summaries instead of blanks.
        if mem and not failed and isinstance(arguments, dict):
            path = _normalise_path(mem, arguments.get("path", ""))
            try:
                if name == "read_file" and path:
                    mem.record_read(path, call.id)
                elif name == "write_file" and path:
                    mem.record_write(path, call.id)
            except Exception as exc:  # noqa: BLE001 - memory must not kill a run
                _say(f"    {_BAD} memory update failed ({exc}); continuing")

        results.append(
            {"role": "tool", "tool_call_id": call.id, "content": result}
        )

    return results


def _normalise_path(mem: Any, path: str) -> str:
    """Repo-relative forward-slash form of a tool-call path."""
    if not path:
        return ""
    try:
        p = Path(path)
        if p.is_absolute():
            p = p.relative_to(mem.root)
        return p.as_posix()
    except (ValueError, OSError):
        return path.replace("\\", "/")


def _log_reasoning(message: Any) -> None:
    """Print a one-line preview of the model's reasoning, when it exposes it."""
    reasoning = getattr(message, "reasoning", None)
    if reasoning:
        _say(f"  {_clip(_flatten(reasoning), 100)}")


def _log_text(text: str) -> None:
    """Print assistant text, keeping its line structure.

    The PLAN phase is a numbered list, so flattening it to one line would make
    the most informative moment of a run the least readable one.
    """
    if not text:
        return

    lines = text.splitlines()
    for line in lines[:MAX_TEXT_LINES]:
        _say(f"  {_clip(line, MAX_TEXT_PREVIEW)}" if line.strip() else "")
    if len(lines) > MAX_TEXT_LINES:
        _say(f"  … {len(lines) - MAX_TEXT_LINES} more lines")


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
