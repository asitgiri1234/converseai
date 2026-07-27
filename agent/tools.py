"""Tool definitions and dispatch for the coding agent.

Two things live here:

1. ``TOOL_SCHEMAS`` -- the JSON Schema definitions handed to the model in the
   ``tools`` parameter of a Messages API request.
2. ``dispatch`` -- the client-side executor that runs a tool the model asked
   for and returns a string result.

Every tool is scoped to the repository root passed in on the CLI. Paths that
resolve outside that root must be rejected: the ``path`` field of a tool call
is model output, not trusted input.

Nothing here is implemented yet -- these are the stubs the loop will call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Tool definitions sent to the model. Each entry needs `name`, `description`,
# and `input_schema`; the description is what the model reads to decide when
# to call the tool, so it should say *when* to use it, not just what it does.
TOOL_SCHEMAS: list[dict[str, Any]] = []


def read_file(repo: Path, path: str) -> str:
    """Return the contents of a file inside the repo.

    Args:
        repo: Repository root. All access is confined to this directory.
        path: Repo-relative path to read.

    Returns:
        File contents, or an error message if the path is missing, is a
        directory, or escapes ``repo``.
    """
    raise NotImplementedError


def write_file(repo: Path, path: str, content: str) -> str:
    """Create or overwrite a file inside the repo.

    Args:
        repo: Repository root. All access is confined to this directory.
        path: Repo-relative path to write. Parent directories are created.
        content: Full new contents of the file.

    Returns:
        A short confirmation, or an error message on failure.
    """
    raise NotImplementedError


def list_files(repo: Path, pattern: str = "**/*") -> str:
    """List files in the repo matching a glob pattern.

    Args:
        repo: Repository root.
        pattern: Glob relative to ``repo``. Defaults to everything, recursive.

    Returns:
        Newline-separated repo-relative paths.
    """
    raise NotImplementedError


def run_command(repo: Path, command: str) -> str:
    """Run a shell command with the repo as the working directory.

    Commands come from the model and are untrusted. Before this is
    implemented, decide on the containment strategy: an allowlist of
    executables, a timeout, and refusal of shell metacharacters at minimum.

    Args:
        repo: Repository root, used as the working directory.
        command: The command line to run.

    Returns:
        Combined stdout and stderr, plus the exit code.
    """
    raise NotImplementedError


def dispatch(repo: Path, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    """Execute one tool call from the model.

    Args:
        repo: Repository root, forwarded to the individual tool functions.
        name: The ``name`` field of the model's ``tool_use`` block.
        tool_input: The ``input`` field of that block, already parsed to a dict.

    Returns:
        A ``(result, is_error)`` pair. ``result`` becomes the ``content`` of
        the ``tool_result`` block; ``is_error`` becomes its ``is_error`` flag.
        Failures are reported this way rather than raised, so the model can
        see what went wrong and adjust.
    """
    raise NotImplementedError


def _resolve(repo: Path, path: str) -> Path:
    """Resolve a model-supplied path against the repo root, or raise.

    Args:
        repo: Repository root.
        path: Untrusted repo-relative path from a tool call.

    Returns:
        The absolute, symlink-resolved path.

    Raises:
        ValueError: If the resolved path falls outside ``repo``.
    """
    raise NotImplementedError
