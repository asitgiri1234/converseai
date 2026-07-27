"""The agent loop: model turn -> tool calls -> results -> repeat.

The Messages API is stateless, so this module owns the conversation history:
it appends each assistant response and each batch of tool results, then calls
the model again until the model stops asking for tools.

Shape of one iteration:

    response = llm.complete(messages, tools=TOOL_SCHEMAS, system=...)
    messages.append({"role": "assistant", "content": response.content})
    if response.stop_reason != "tool_use":
        break
    results = [run each tool_use block through tools.dispatch]
    messages.append({"role": "user", "content": results})

Two rules that are easy to get wrong: append the *whole* ``response.content``
(not just the text) so ``tool_use`` blocks survive the round trip, and return
every tool result in a single user message so parallel tool calls keep working.

Not implemented yet -- these are the stubs `main.py` calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.llm import LLM

# Safety valve: stop after this many model turns even if the agent is still
# asking for tools, so a confused run can't spin forever.
MAX_TURNS = 50


def run(repo: Path, task: str, llm: LLM | None = None, max_turns: int = MAX_TURNS) -> str:
    """Run the agent against a repository until the task is done.

    Args:
        repo: Repository root the agent is allowed to read and modify.
        task: Natural-language description of what the agent should do.
        llm: Model wrapper to use. A default ``LLM()`` is constructed if
            omitted.
        max_turns: Hard cap on model turns before giving up.

    Returns:
        The agent's final text response.

    Raises:
        RuntimeError: If ``max_turns`` is reached with the agent still
            requesting tools.
    """
    raise NotImplementedError


def _handle_tool_use(repo: Path, response: Any) -> list[dict[str, Any]]:
    """Execute every ``tool_use`` block in a response.

    Args:
        repo: Repository root, forwarded to `agent.tools.dispatch`.
        response: The ``Message`` returned by `agent.llm.LLM.complete`.

    Returns:
        One ``tool_result`` block per ``tool_use`` block, in the same order,
        each carrying the matching ``tool_use_id``. The API rejects the next
        request if any ``tool_use`` id is left without a result.
    """
    raise NotImplementedError


def _final_text(response: Any) -> str:
    """Join the text blocks of a response into a single string.

    Args:
        response: The ``Message`` returned by `agent.llm.LLM.complete`.

    Returns:
        The concatenated text, ignoring thinking and tool_use blocks.
    """
    raise NotImplementedError
