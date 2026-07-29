"""Patch-Based Editing Engine: change the smallest region that must change.

Whole-file rewrites were the single largest source of diff noise in this
project. A model asked to add one endpoint would retype the file and "tidy"
everything it retyped -- one measured run produced 84 insertions and 63
deletions for a change that needed about 12 lines. Worse, a retyped file can
silently drop a function or convert a module's export shape.

This engine replaces that with anchored replacement. The caller supplies a
snippet of the *existing* text and its replacement; everything outside the
matched span is preserved byte for byte, so untouched code cannot be
reformatted, reordered, or lost.

Matching is deliberately strict:

    exact          the anchor appears verbatim, exactly once      -> apply
    whitespace     it appears once ignoring trailing whitespace   -> apply
    ambiguous      it appears more than once                      -> refuse
    missing        it does not appear                             -> refuse,
                   with the closest line in the file as a hint

Refusals are errors the model can act on, never silent guesses: applying an
edit to the wrong one of three identical blocks is worse than not editing.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass


class PatchError(ValueError):
    """Raised when a patch cannot be applied safely."""


@dataclass
class PatchResult:
    """Outcome of applying one patch."""

    text: str              # the new file contents
    start_line: int        # 1-based line where the change begins
    added: int             # lines added
    removed: int           # lines removed
    mode: str              # exact | whitespace
    diff: str              # unified diff of the change

    @property
    def churn(self) -> int:
        """Total lines touched -- the number this engine exists to minimise."""
        return self.added + self.removed


def _normalise(text: str) -> str:
    """Strip trailing whitespace from every line, for tolerant matching."""
    return "\n".join(line.rstrip() for line in text.splitlines())


def locate(haystack: str, needle: str) -> tuple[int, int, str]:
    """Find the unique span of ``needle`` inside ``haystack``.

    Args:
        haystack: The current file contents.
        needle: The snippet the caller expects to find.

    Returns:
        ``(start, end, mode)`` character offsets and how it matched.

    Raises:
        PatchError: If the snippet is empty, missing, or appears more than
            once. The message names the problem and what to do about it.
    """
    if not needle:
        raise PatchError("the text to replace is empty")

    count = haystack.count(needle)
    if count == 1:
        start = haystack.index(needle)
        return start, start + len(needle), "exact"
    if count > 1:
        raise PatchError(
            f"the text to replace appears {count} times, so the edit is "
            "ambiguous. Include more surrounding context to make it unique."
        )

    # Retry ignoring trailing whitespace, which models routinely get wrong.
    flat_hay = _normalise(haystack)
    flat_needle = _normalise(needle)
    if flat_needle and flat_hay.count(flat_needle) == 1:
        offset = flat_hay.index(flat_needle)
        # Map the normalised offset back by counting lines, since stripping
        # changes character positions but never line boundaries.
        line_no = flat_hay.count("\n", 0, offset)
        lines = haystack.splitlines(keepends=True)
        start = sum(len(x) for x in lines[:line_no])
        span = len(flat_needle.splitlines())
        end = start + sum(len(x) for x in lines[line_no:line_no + span])
        return start, end, "whitespace"

    raise PatchError(
        "the text to replace was not found in the file. "
        + _closest_hint(haystack, needle)
    )


def _closest_hint(haystack: str, needle: str) -> str:
    """Point at the nearest line, so a failed match is actionable."""
    first = next((ln for ln in needle.splitlines() if ln.strip()), "")
    if not first:
        return "Re-read the file and copy the exact text."

    lines = haystack.splitlines()
    best_ratio, best_line = 0.0, 0
    for i, line in enumerate(lines, 1):
        ratio = difflib.SequenceMatcher(None, first.strip(), line.strip()).ratio()
        if ratio > best_ratio:
            best_ratio, best_line = ratio, i

    if best_ratio > 0.6:
        return (
            f"The closest line is {best_line}: {lines[best_line - 1].strip()!r}. "
            "Re-read the file and copy the exact text, including indentation."
        )
    return "Re-read the file and copy the exact text, including indentation."


def make_diff(before: str, after: str, path: str) -> str:
    """Unified diff between two versions of a file."""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=2,
        )
    )


def diff_stats(before: str, after: str) -> tuple[int, int]:
    """Return ``(added, removed)`` line counts between two versions."""
    added = removed = 0
    for line in difflib.unified_diff(
        before.splitlines(), after.splitlines(), lineterm="", n=0
    ):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def apply_patch(before: str, old_text: str, new_text: str, path: str = "file") -> PatchResult:
    """Replace one located span, leaving everything else byte-identical.

    Args:
        before: Current file contents.
        old_text: Snippet to replace; must appear exactly once.
        new_text: Its replacement. Empty deletes the span.
        path: Repo-relative path, used in the diff header.

    Returns:
        A `PatchResult` carrying the new text, the diff, and churn counts.

    Raises:
        PatchError: If the span cannot be located uniquely, or if the edit
            would change nothing.
    """
    start, end, mode = locate(before, old_text)
    after = before[:start] + new_text + before[end:]

    if after == before:
        raise PatchError(
            "the replacement is identical to the existing text; nothing to do"
        )

    verify_untouched(before, after, start, end, len(new_text))
    added, removed = diff_stats(before, after)
    return PatchResult(
        text=after,
        start_line=before.count("\n", 0, start) + 1,
        added=added,
        removed=removed,
        mode=mode,
        diff=make_diff(before, after, path),
    )


def verify_untouched(
    before: str, after: str, start: int, end: int, new_length: int
) -> None:
    """Assert that only the intended span differs.

    This holds by construction, so it is a guard against a future refactor
    quietly breaking the guarantee the whole engine rests on -- not a check
    on the model.

    Raises:
        PatchError: If any byte outside the replaced span changed.
    """
    if before[:start] != after[:start]:
        raise PatchError("internal: text before the edit region changed")
    if before[end:] != after[start + new_length:]:
        raise PatchError("internal: text after the edit region changed")


def insert_after(before: str, anchor: str, addition: str, path: str = "file") -> PatchResult:
    """Insert ``addition`` immediately after the located ``anchor``.

    A convenience over `apply_patch` for the commonest agent edit -- adding a
    route, a handler, or a field -- where retyping the anchor as part of the
    replacement is pure overhead and an opportunity for typos.

    Args:
        before: Current file contents.
        anchor: Existing text to insert after; must appear exactly once.
        addition: Text to insert. A leading newline is added if absent.
        path: Repo-relative path, used in the diff header.

    Returns:
        A `PatchResult`.

    Raises:
        PatchError: If the anchor cannot be located uniquely.
    """
    start, end, mode = locate(before, anchor)

    # Always insert at a line boundary. An anchor that ends mid-line (a bare
    # identifier, say) would otherwise splice the new text into the middle of
    # a statement and corrupt it.
    newline = before.find("\n", end)
    end = len(before) if newline == -1 else newline

    block = addition if addition.startswith("\n") else "\n" + addition
    block = block.rstrip("\n")

    after = before[:end] + block + before[end:]
    verify_untouched(before, after, end, end, len(block))
    added, removed = diff_stats(before, after)
    return PatchResult(
        text=after,
        start_line=before.count("\n", 0, end) + 1,
        added=added,
        removed=removed,
        mode=mode,
        diff=make_diff(before, after, path),
    )


def summarise(result: PatchResult, path: str, max_diff_lines: int = 24) -> str:
    """Human- and model-readable report of what a patch changed.

    Returning the diff is deliberate: the model sees precisely what it did,
    which catches a wrong-anchor edit on the spot instead of at verification.
    """
    header = (
        f"Patched {path} at line {result.start_line}: "
        f"+{result.added}/-{result.removed} lines"
        + (" (matched ignoring trailing whitespace)" if result.mode == "whitespace" else "")
    )
    lines = result.diff.splitlines()
    if len(lines) > max_diff_lines:
        lines = lines[:max_diff_lines] + [f"... diff truncated ({len(lines)} lines)"]
    return header + "\n" + "\n".join(lines)
