"""System prompts and prompt-assembly helpers.

Keep the stable parts of the prompt here and the volatile parts (the task, the
repo listing, timestamps) out of it. The system prompt is the front of the
cached prefix, so anything that changes per-run belongs in the first user
message instead -- see the prompt-caching notes in the README.

Not implemented yet -- `SYSTEM_PROMPT` is a placeholder and the builders are
stubs.
"""

from __future__ import annotations

from pathlib import Path

# The agent's persona and operating rules. Stays byte-identical across runs so
# it can be cached; anything run-specific goes in the user turn.
SYSTEM_PROMPT = """You are a coding agent working inside a single repository.
"""


def build_system_prompt(extra_instructions: str | None = None) -> str:
    """Assemble the system prompt.

    Args:
        extra_instructions: Project-specific rules to append (e.g. the
            contents of a CLAUDE.md found in the repo).

    Returns:
        The full system prompt string.
    """
    raise NotImplementedError


def build_task_message(repo: Path, task: str) -> str:
    """Build the opening user message for a run.

    Args:
        repo: Repository root, used to describe the working environment
            (path, and possibly a top-level file listing).
        task: The user's task description, verbatim.

    Returns:
        The text of the first user message.
    """
    raise NotImplementedError
