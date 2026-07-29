"""Tool definitions and dispatch for the coding agent.

Two things live here:

1. ``TOOLS`` -- the tool definitions handed to the model, in the OpenAI
   function-calling format Groq speaks: a list of
   ``{"type": "function", "function": {"name", "description", "parameters"}}``.
2. ``dispatch`` -- the client-side executor that runs a tool the model asked
   for and returns a string result.

Every tool is scoped to a repository root set once per run with
``set_repo_root``. Paths that resolve outside that root are rejected: the
``path`` field of a tool call is model output, not trusted input.

``dispatch`` never raises. Failures come back as strings starting with
``Error:`` so the loop can hand them to the model as a ``tool_result`` with
``is_error: true`` and let it correct itself.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable

# Directories never walked by list_files or search_code.
SKIP_DIRS = frozenset({"node_modules", ".git"})

MAX_READ_LINES = 500
MAX_SEARCH_RESULTS = 50
MAX_LISTED_ENTRIES = 1_000
MAX_MATCH_CHARS = 200
MAX_SEARCH_FILE_BYTES = 2 * 1024 * 1024
SHELL_TIMEOUT_SECONDS = 30
MAX_SHELL_OUTPUT_CHARS = 20_000

_repo_root: Path | None = None


class PathEscapeError(ValueError):
    """Raised when a model-supplied path resolves outside the repo root."""


class UnsafeCommandError(ValueError):
    """Raised when a shell command is refused by the safety check."""


def set_repo_root(path: str | Path) -> Path:
    """Bind every tool in this module to a repository root.

    Call once at startup, before any tool runs.

    Args:
        path: Directory the agent is allowed to read and modify.

    Returns:
        The resolved absolute root.

    Raises:
        NotADirectoryError: If ``path`` is not an existing directory.
    """
    global _repo_root
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"not a directory: {resolved}")
    _repo_root = resolved
    return resolved


def get_repo_root() -> Path:
    """Return the bound repository root.

    Raises:
        RuntimeError: If ``set_repo_root`` has not been called.
    """
    if _repo_root is None:
        raise RuntimeError("repo root is not set; call set_repo_root() first")
    return _repo_root


def _resolve(path: str) -> Path:
    """Resolve a model-supplied path against the repo root, or raise.

    Symlinks are followed before the containment check, so a link pointing
    outside the repo is rejected rather than traversed.

    Args:
        path: Untrusted path from a tool call. Relative paths are taken
            relative to the repo root; absolute paths are allowed only if they
            already fall inside it.

    Returns:
        The absolute, symlink-resolved path.

    Raises:
        PathEscapeError: If the resolved path falls outside the repo root.
    """
    root = get_repo_root()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate

    resolved = candidate.resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise PathEscapeError(
            f"path escapes the repository root: {path!r} -> {resolved}"
        )
    return resolved


def _rel(path: Path) -> str:
    """Format an absolute path as repo-relative with forward slashes."""
    return path.relative_to(get_repo_root()).as_posix() or "."


def _walk(start: Path):
    """Walk ``start`` depth-first, pruning ``SKIP_DIRS``.

    Yields ``(dirpath, dirnames, filenames)`` like ``os.walk``, with
    ``dirnames`` sorted and filtered in place.
    """
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        filenames.sort()
        yield dirpath, dirnames, filenames


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------


def list_files(path: str = ".") -> str:
    """Recursively list files under ``path`` as an indented tree.

    Args:
        path: Repo-relative directory to list. Defaults to the repo root.

    Returns:
        An indented tree of directories and files, or an error message.
    """
    try:
        start = _resolve(path)
    except PathEscapeError as exc:
        return f"Error: {exc}"

    if not start.exists():
        return f"Error: no such directory: {path}"
    if not start.is_dir():
        return f"Error: not a directory: {path} (use read_file for files)"

    lines: list[str] = [f"{_rel(start)}/"]
    complete = _append_tree(start, 1, lines)

    if not complete:
        lines.append(
            f"... truncated at {MAX_LISTED_ENTRIES} entries; "
            "list a subdirectory to see more"
        )
    if len(lines) == 1:
        lines.append("  (empty)")
    return "\n".join(lines)


def _append_tree(directory: Path, depth: int, lines: list[str]) -> bool:
    """Append ``directory``'s contents to ``lines`` as an indented tree.

    Children follow their parent immediately, directories before files.
    Symlinked directories are listed but not descended into, so a link cycle
    cannot hang the walk.

    Args:
        directory: Directory to expand.
        depth: Indentation level for this directory's children.
        lines: Accumulator, mutated in place.

    Returns:
        ``False`` if the ``MAX_LISTED_ENTRIES`` budget ran out, else ``True``.
    """
    try:
        entries = sorted(
            directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
        )
    except OSError:
        lines.append(f"{'  ' * depth}(unreadable)")
        return True

    for entry in entries:
        if len(lines) >= MAX_LISTED_ENTRIES:
            return False
        if entry.is_dir():
            if entry.name in SKIP_DIRS:
                continue
            if entry.is_symlink():
                lines.append(f"{'  ' * depth}{entry.name}/ -> (symlink, not followed)")
                continue
            lines.append(f"{'  ' * depth}{entry.name}/")
            if not _append_tree(entry, depth + 1, lines):
                return False
        else:
            lines.append(f"{'  ' * depth}{entry.name}")
    return True


def read_file(path: str) -> str:
    """Read a file and return its contents with line numbers.

    Args:
        path: Repo-relative path to read.

    Returns:
        The file contents, one ``<lineno>\\t<text>`` per line, capped at
        ``MAX_READ_LINES`` lines; or an error message.
    """
    try:
        target = _resolve(path)
    except PathEscapeError as exc:
        return f"Error: {exc}"

    if not target.exists():
        return f"Error: no such file: {path}"
    if target.is_dir():
        return f"Error: {path} is a directory (use list_files)"

    try:
        raw = target.read_bytes()
    except OSError as exc:
        return f"Error: could not read {path}: {exc}"

    # A NUL byte is the giveaway for binary content: bytes like \x00\x01 decode
    # as valid UTF-8, so catching UnicodeDecodeError alone is not enough.
    if b"\0" in raw[:8192]:
        return f"Error: {path} looks like a binary file; not reading it as text."

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"Error: {path} is not valid UTF-8 text (binary file?)"

    if not text:
        return f"{_rel(target)} is empty."

    lines = text.splitlines()
    shown = lines[:MAX_READ_LINES]
    body = "\n".join(f"{i:6d}\t{line}" for i, line in enumerate(shown, start=1))

    if len(lines) > MAX_READ_LINES:
        body += (
            f"\n... truncated: showing {MAX_READ_LINES} of {len(lines)} lines. "
            "Use search_code or run_shell to inspect the rest."
        )
    return body


def search_code(pattern: str) -> str:
    """Regex-search every text file in the repo.

    Args:
        pattern: Python regular expression to match against each line.

    Returns:
        Up to ``MAX_SEARCH_RESULTS`` ``file:line:match`` rows, or an error
        message if the pattern is invalid.
    """
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"Error: invalid regular expression {pattern!r}: {exc}"

    root = get_repo_root()
    results: list[str] = []
    capped = False

    for dirpath, _dirnames, filenames in _walk(root):
        for name in filenames:
            target = Path(dirpath) / name
            try:
                if target.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                raw = target.read_bytes()
            except OSError:
                continue
            if b"\0" in raw[:8192]:  # binary
                continue

            text = raw.decode("utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if not regex.search(line):
                    continue
                snippet = line.strip()
                if len(snippet) > MAX_MATCH_CHARS:
                    snippet = snippet[:MAX_MATCH_CHARS] + "..."
                results.append(f"{_rel(target)}:{lineno}:{snippet}")
                if len(results) >= MAX_SEARCH_RESULTS:
                    capped = True
                    break
            if capped:
                break
        if capped:
            break

    if not results:
        return f"No matches for {pattern!r}."
    out = "\n".join(results)
    if capped:
        out += (
            f"\n... capped at {MAX_SEARCH_RESULTS} results; "
            "narrow the pattern to see the rest."
        )
    return out


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file, creating parent directories as needed.

    Args:
        path: Repo-relative path to write.
        content: Full new contents of the file.

    Returns:
        A short confirmation, or an error message.
    """
    try:
        target = _resolve(path)
    except PathEscapeError as exc:
        return f"Error: {exc}"

    if target.is_dir():
        return f"Error: {path} is a directory"

    existed = target.exists()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"Error: could not write {path}: {exc}"

    verb = "Overwrote" if existed else "Created"
    return f"{verb} {_rel(target)} ({len(content.splitlines())} lines, {len(content)} bytes)."


_search_memory = None  # RepositoryMemory, injected by the loop or built lazily


def set_search_index(memory) -> None:
    """Give the search tool a prebuilt repository memory.

    `agent.loop` calls this so ``search_repo`` reuses the scan the run already
    paid for, and sees files the agent has written this run.

    Args:
        memory: An `agent.memory.RepositoryMemory`, or None to clear.
    """
    global _search_memory
    _search_memory = memory


def search_repo(question: str) -> str:
    """Answer a question about the repository semantically.

    Args:
        question: A natural-language question, e.g. "which endpoint deletes a
            note?".

    Returns:
        Ranked results with locations and the evidence behind each match, or
        a clear statement that nothing matched.
    """
    # Imported here to keep module import cheap and avoid a cycle at load.
    from agent import memory as repo_memory
    from agent import search as repo_search

    global _search_memory
    if _search_memory is None:
        _search_memory = repo_memory.build_memory(get_repo_root())

    hits = repo_search.search(question, _search_memory)
    return repo_search.render(question, hits)


def _check_command(command: str) -> None:
    """Refuse obviously destructive shell commands.

    This is a guardrail, not a sandbox. It blocks privilege escalation, forced
    recursive deletes aimed outside the repo, and a handful of
    system-destroying commands. Anything that survives this check still runs
    with the caller's full privileges -- run the agent in a container or a
    throwaway checkout if that matters.

    Args:
        command: The command line the model asked to run.

    Raises:
        UnsafeCommandError: If the command matches a blocked pattern.
    """
    lowered = command.lower()

    for phrase, reason in (
        ("mkfs", "it formats a filesystem"),
        ("shutdown", "it shuts down the machine"),
        ("reboot", "it reboots the machine"),
        (":(){", "it is a fork bomb"),
    ):
        if phrase in lowered:
            raise UnsafeCommandError(f"refusing command: {reason}")

    if re.search(r"\bdd\b[^|;&]*\bof=/dev/", lowered):
        raise UnsafeCommandError("refusing to write directly to a device")

    try:
        tokens = shlex.split(command, posix=os.name != "nt")
    except ValueError:
        # Unbalanced quotes: fall back to a crude split so the checks below
        # still see the words.
        tokens = command.split()
    tokens = [t.strip("'\"") for t in tokens]

    for i, token in enumerate(tokens):
        word = token.lower()

        if word in {"sudo", "doas", "su", "runas"}:
            raise UnsafeCommandError(f"refusing privilege escalation via {word!r}")

        if word in {"rm", "rmdir", "del", "rd"}:
            _check_delete(tokens[i + 1 :])


def _check_delete(args: list[str]) -> None:
    """Reject a delete whose targets are not clearly inside the repo.

    Args:
        args: Tokens following the delete command, up to the next shell
            operator.

    Raises:
        UnsafeCommandError: If a target resolves outside the repo root, or if
            a recursive force delete has no explicit target.
    """
    targets: list[str] = []
    recursive_force = False
    flags = ""

    for arg in args:
        if arg in {"&&", "||", ";", "|"}:
            break
        if arg.startswith("-"):
            flags += arg.lower()
            continue
        targets.append(arg)

    if ("r" in flags and "f" in flags) or "/s" in flags or "recursive" in flags:
        recursive_force = True

    for target in targets:
        if any(ch in target for ch in "*?["):
            # Globs are resolved by the shell; check the directory they sit in.
            target = str(Path(target).parent)
        try:
            _resolve(target)
        except PathEscapeError:
            raise UnsafeCommandError(
                f"refusing to delete {target!r}: outside the repository root"
            ) from None
        except RuntimeError:
            raise UnsafeCommandError(
                "refusing to delete: repository root is not set"
            ) from None

    if recursive_force and not targets:
        raise UnsafeCommandError("refusing recursive force delete with no target")


def run_shell(command: str) -> str:
    """Run a shell command with the repo as the working directory.

    Args:
        command: The command line to run. Executed through the platform shell
            (``/bin/sh`` on POSIX, ``cmd.exe`` on Windows), so operators and
            pipes work.

    Returns:
        The exit code followed by stdout and stderr, or an error message if
        the command was refused or timed out.
    """
    try:
        root = get_repo_root()
        _check_command(command)
    except (UnsafeCommandError, RuntimeError) as exc:
        return f"Error: {exc}"

    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return (
            f"Error: command timed out after {SHELL_TIMEOUT_SECONDS}s and was killed: "
            f"{command}"
        )
    except OSError as exc:
        return f"Error: could not run command: {exc}"

    parts = [f"exit code: {completed.returncode}"]
    if completed.stdout:
        parts.append(f"stdout:\n{completed.stdout.rstrip()}")
    if completed.stderr:
        parts.append(f"stderr:\n{completed.stderr.rstrip()}")
    if not completed.stdout and not completed.stderr:
        parts.append("(no output)")

    out = "\n".join(parts)
    if len(out) > MAX_SHELL_OUTPUT_CHARS:
        out = out[:MAX_SHELL_OUTPUT_CHARS] + "\n... output truncated"
    return out


# --------------------------------------------------------------------------
# Schemas and dispatch
# --------------------------------------------------------------------------

def _function(name: str, description: str, properties: dict, required: list[str]):
    """Wrap a tool in the OpenAI function-calling envelope Groq expects."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOLS: list[dict[str, Any]] = [
    _function(
        "list_files",
        "List files and directories recursively as an indented tree. Call "
        "this to orient yourself in an unfamiliar repository or to confirm a "
        "path exists before reading or writing it. node_modules and .git are "
        "never listed.",
        {
            "path": {
                "type": "string",
                "description": (
                    "Directory to list, relative to the repository root. "
                    "Defaults to the root itself."
                ),
            }
        },
        [],
    ),
    _function(
        "read_file",
        "Read a text file and return its contents with line numbers. Call "
        "this before editing any file so your changes are based on what is "
        f"actually there. Output is capped at {MAX_READ_LINES} lines.",
        {
            "path": {
                "type": "string",
                "description": "File to read, relative to the repository root.",
            }
        },
        ["path"],
    ),
    _function(
        "search_code",
        "Search every text file in the repository with a regular expression "
        "and return matching 'file:line:text' rows. Call this to find where a "
        "symbol is defined or used, instead of reading files one by one. "
        f"Capped at {MAX_SEARCH_RESULTS} results.",
        {
            "pattern": {
                "type": "string",
                "description": (
                    "Python regular expression, matched against each line. "
                    r"Example: 'exports\.\w+'"
                ),
            }
        },
        ["pattern"],
    ),
    _function(
        "search_repo",
        "Ask a question about this repository in plain English and get "
        "ranked, located answers -- for example 'where are notes created?', "
        "'which endpoint deletes a note?', 'which controller uses this "
        "model?', or 'which functions write to the database?'. Prefer this "
        "over search_code when you want to find WHERE something happens; use "
        "search_code when you need a literal string or a precise regex. If "
        "it reports no matches, the repository genuinely does not appear to "
        "contain that -- do not assume otherwise.",
        {
            "question": {
                "type": "string",
                "description": (
                    "A natural-language question about the codebase, e.g. "
                    "'which controller handles note updates?'"
                ),
            }
        },
        ["question"],
    ),
    _function(
        "write_file",
        "Create a file or overwrite it completely, creating parent "
        "directories as needed. The content you supply replaces the whole "
        "file, so read it first unless you are creating it.",
        {
            "path": {
                "type": "string",
                "description": "File to write, relative to the repository root.",
            },
            "content": {
                "type": "string",
                "description": "Full new contents of the file.",
            },
        },
        ["path", "content"],
    ),
    _function(
        "run_shell",
        "Run a shell command with the repository as the working directory. "
        "Call this to run tests, linters, build steps, or git. Commands are "
        f"killed after {SHELL_TIMEOUT_SECONDS} seconds, and privilege "
        "escalation and destructive deletes are refused.",
        {
            "command": {
                "type": "string",
                "description": "The command line to run, e.g. 'npm test'.",
            }
        },
        ["command"],
    ),
]

_IMPLEMENTATIONS: dict[str, Callable[..., str]] = {
    "list_files": list_files,
    "read_file": read_file,
    "search_code": search_code,
    "search_repo": search_repo,
    "write_file": write_file,
    "run_shell": run_shell,
}


def dispatch(name: str, tool_input: dict[str, Any]) -> str:
    """Execute one tool call from the model.

    Never raises: unknown tools, bad arguments, and unexpected exceptions all
    come back as an ``Error:`` string so the model can see what went wrong and
    try something else.

    Args:
        name: The ``name`` field of the model's ``tool_use`` block.
        tool_input: The ``input`` field of that block.

    Returns:
        The tool's output, or a message starting with ``Error:``.
    """
    impl = _IMPLEMENTATIONS.get(name)
    if impl is None:
        known = ", ".join(sorted(_IMPLEMENTATIONS))
        return f"Error: unknown tool {name!r}. Available tools: {known}."

    if not isinstance(tool_input, dict):
        return f"Error: {name} expects an object of arguments, got {type(tool_input).__name__}."

    try:
        return impl(**tool_input)
    except TypeError as exc:
        return f"Error: bad arguments for {name}: {exc}"
    except Exception as exc:  # noqa: BLE001 - the model handles the failure
        return f"Error: {name} failed: {type(exc).__name__}: {exc}"
