"""Multi-step validation pipeline and confidence scoring.

`node --check` proves a file parses. It does not prove the application works,
and in this project it twice reported success on a broken change:

* a routes module was converted from ``module.exports = (app) => {...}`` to an
  ``express.Router``; valid syntax, dead at startup, because its caller
  invokes it as a function;
* a new literal route was registered *after* ``/notes/:noteId``, so the
  parameterised route swallowed it and the endpoint returned 404.

Both are structural, both are detectable without running anything, and both
now have a dedicated check. The pipeline runs five stages:

    1. syntax        node --check / py_compile on every changed file
    2. contracts     export shape still matches how importers call it
    3. routes        handlers exist; literal routes are not shadowed
    4. tests         npm test / pytest, with placeholder scripts recognised
    5. endpoints     live HTTP probe when a server is already reachable

Then it scores confidence from four observable signals -- repository
understanding, implementation quality, verification results, tests -- and the
loop returns to planning when the score falls below the threshold.

The score is computed by the harness from what actually happened, not
self-reported by the model. A model that has just written a broken change is
the least reliable narrator of whether the change is broken.
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

CONFIDENCE_THRESHOLD = 70.0
SHELL_TIMEOUT = 30

# Weights must total 100.
W_UNDERSTANDING = 25.0
W_IMPLEMENTATION = 25.0
W_VERIFICATION = 35.0
W_TESTS = 15.0

# npm's default placeholder, which always fails and means nothing.
PLACEHOLDER_TEST = re.compile(r"no test specified|Error: no test", re.I)

SYNTAX_CHECKERS = {
    ".js": ["node", "--check"],
    ".cjs": ["node", "--check"],
    ".mjs": ["node", "--check"],
    ".py": ["python", "-m", "py_compile"],
}


@dataclass
class Check:
    """One validation outcome."""

    stage: str
    name: str
    status: str          # pass | fail | skip
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def render(self) -> str:
        mark = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[self.status]
        return f"[{mark}] {self.stage}: {self.name}" + (f" -- {self.detail}" if self.detail else "")


@dataclass
class Report:
    """The full pipeline result plus its confidence breakdown."""

    checks: list[Check] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    added: int = 0
    removed: int = 0
    scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    def render(self) -> str:
        lines = [c.render() for c in self.checks]
        lines.append(
            "confidence "
            + f"{self.confidence:.0f}/100  ("
            + ", ".join(f"{k} {v:.0f}" for k, v in self.scores.items())
            + ")"
        )
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------


def changed_files(root: Path) -> tuple[list[str], int, int]:
    """Files changed versus git HEAD, with added/removed line counts.

    Git is the authority here rather than the agent's own bookkeeping: it
    reports what the working tree actually contains.

    Returns:
        ``(paths, added, removed)``. Empty and zeroed when the directory is
        not a git repository.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "diff", "--numstat", "HEAD"],
            capture_output=True, text=True, timeout=SHELL_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return [], 0, 0
    if out.returncode != 0:
        return [], 0, 0

    paths: list[str] = []
    added = removed = 0
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        a, r, path = parts
        paths.append(path)
        added += int(a) if a.isdigit() else 0
        removed += int(r) if r.isdigit() else 0
    return paths, added, removed


def check_syntax(root: Path, files: list[str]) -> list[Check]:
    """Stage 1 -- parse every changed source file."""
    checks: list[Check] = []
    for rel in files:
        checker = SYNTAX_CHECKERS.get(Path(rel).suffix.lower())
        if not checker or not (root / rel).is_file():
            continue
        try:
            out = subprocess.run(
                checker + [rel], cwd=root, capture_output=True,
                text=True, timeout=SHELL_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            checks.append(Check("syntax", rel, "skip", str(exc)[:80]))
            continue
        if out.returncode == 0:
            checks.append(Check("syntax", rel, "pass"))
        else:
            detail = (out.stderr or out.stdout).strip().splitlines()
            checks.append(Check("syntax", rel, "fail", detail[0][:160] if detail else "parse error"))
    return checks


def check_contracts(root: Path, memory, files: list[str]) -> list[Check]:
    """Stage 2 -- a module's export shape must still fit its callers.

    Catches the failure `node --check` cannot: converting
    ``module.exports = (app) => {...}`` into an object or a Router while a
    caller still does ``require('./x')(app)``.
    """
    checks: list[Check] = []
    for rel in files:
        if Path(rel).suffix.lower() not in (".js", ".cjs", ".mjs"):
            continue
        path = root / rel
        if not path.is_file():
            continue
        importers = memory.graph.importers_of(rel)
        if not importers:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        exports_function = bool(
            re.search(r"module\.exports\s*=\s*(async\s*)?(\(|function\b)", text)
        )

        for importer in importers:
            ipath = root / importer
            if not ipath.is_file():
                continue
            try:
                itext = ipath.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            # require('...')(  -- the caller invokes the module immediately.
            called = re.search(
                r"require\(\s*['\"][^'\"]*"
                + re.escape(Path(rel).stem)
                + r"[^'\"]*['\"]\s*\)\s*\(",
                itext,
            )
            if called and not exports_function:
                checks.append(Check(
                    "contracts", rel, "fail",
                    f"{importer} calls it as a function, but it no longer "
                    "exports one -- the app will crash at startup",
                ))
            elif called:
                checks.append(Check("contracts", rel, "pass",
                                    f"still a function for {importer}"))
    return checks


def check_routes(memory) -> list[Check]:
    """Stage 3 -- handlers must exist, and literal routes must not be shadowed."""
    checks: list[Check] = []
    intel, graph = memory.intel, memory.graph
    known = {f"{s.file}::{s.name}" for s in intel.symbols}

    for edge in graph.edges:
        if edge.kind != "registers":
            continue
        if edge.dst not in known:
            checks.append(Check(
                "routes", edge.src, "fail",
                f"handler {edge.dst} is not defined or not exported",
            ))
        else:
            checks.append(Check("routes", edge.src, "pass", f"-> {edge.dst}"))

    # Registration order: within a file, a parameterised route registered
    # earlier swallows a later literal route on the same prefix.
    by_file: dict[str, list] = {}
    for ep in intel.endpoints:
        if ep.kind == "route":
            by_file.setdefault(ep.file, []).append(ep)

    for file, routes in by_file.items():
        routes.sort(key=lambda e: e.line)
        for i, later in enumerate(routes):
            lm, lp = later.detail.split(" ", 1)
            lseg = lp.strip("/").split("/")
            for earlier in routes[:i]:
                em, ep_ = earlier.detail.split(" ", 1)
                eseg = ep_.strip("/").split("/")
                if em != lm or len(eseg) != len(lseg):
                    continue
                shadowed = all(
                    e.startswith(":") or e == l for e, l in zip(eseg, lseg)
                ) and any(e.startswith(":") and not l.startswith(":")
                          for e, l in zip(eseg, lseg))
                if shadowed:
                    checks.append(Check(
                        "routes", later.detail, "fail",
                        f"unreachable: {earlier.detail} is registered first "
                        f"({file}:{earlier.line}) and matches the same paths",
                    ))
    return checks


def check_tests(root: Path, memory) -> list[Check]:
    """Stage 4 -- run the project's own tests, if it really has any."""
    manifest = root / "package.json"
    if manifest.is_file():
        try:
            scripts = json.loads(manifest.read_text(encoding="utf-8")).get("scripts", {})
        except (OSError, json.JSONDecodeError):
            scripts = {}
        script = scripts.get("test", "")
        if not script:
            return [Check("tests", "npm test", "skip", "no test script defined")]
        if PLACEHOLDER_TEST.search(script):
            return [Check("tests", "npm test", "skip",
                          "placeholder script, not a real suite")]
        return [_run_tests(root, ["npm", "test", "--silent"], "npm test")]

    if (root / "pytest.ini").is_file() or (root / "tests").is_dir():
        return [_run_tests(root, ["python", "-m", "pytest", "-q"], "pytest")]
    return [Check("tests", "suite", "skip", "no test suite found")]


def _run_tests(root: Path, command: list[str], name: str) -> Check:
    try:
        out = subprocess.run(command, cwd=root, capture_output=True,
                             text=True, timeout=SHELL_TIMEOUT, shell=False)
    except subprocess.TimeoutExpired:
        return Check("tests", name, "fail", f"timed out after {SHELL_TIMEOUT}s")
    except (OSError, subprocess.SubprocessError) as exc:
        return Check("tests", name, "skip", f"could not run: {exc}"[:100])
    if out.returncode == 0:
        return Check("tests", name, "pass")
    tail = (out.stdout + out.stderr).strip().splitlines()
    return Check("tests", name, "fail", tail[-1][:160] if tail else "non-zero exit")


def check_endpoints(memory, base_url: str = "http://localhost:3000") -> list[Check]:
    """Stage 5 -- probe live endpoints, when something is already serving.

    Opportunistic by design: starting the application would need a database,
    a port, and longer than the shell timeout allows. When nothing is
    listening this reports ``skip``, never a false pass.
    """
    routes = [e for e in memory.intel.endpoints
              if e.kind == "route" and e.detail.startswith("GET ")]
    if not routes:
        return [Check("endpoints", "probe", "skip", "no GET routes to probe")]

    try:
        urllib.request.urlopen(base_url, timeout=2).read()
    except urllib.error.HTTPError:
        pass  # responding, just not with 2xx at "/" -- good enough
    except OSError:
        return [Check("endpoints", "probe", "skip",
                      f"nothing serving at {base_url}")]

    checks: list[Check] = []
    for route in routes:
        path = route.detail.split(" ", 1)[1]
        if ":" in path:
            continue  # needs a real id; skip rather than guess
        url = base_url.rstrip("/") + path
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                body = response.read(2000).decode("utf-8", errors="replace")
                ctype = response.headers.get("Content-Type", "")
            if response.status >= 500:
                checks.append(Check("endpoints", path, "fail", f"HTTP {response.status}"))
            elif "json" in ctype.lower():
                try:
                    json.loads(body)
                    checks.append(Check("endpoints", path, "pass",
                                        f"HTTP {response.status}, valid JSON"))
                except json.JSONDecodeError:
                    checks.append(Check("endpoints", path, "fail",
                                        "Content-Type says JSON but body does not parse"))
            else:
                checks.append(Check("endpoints", path, "pass", f"HTTP {response.status}"))
        except urllib.error.HTTPError as exc:
            status = "fail" if exc.code >= 500 or exc.code == 404 else "pass"
            checks.append(Check("endpoints", path, status, f"HTTP {exc.code}"))
        except OSError as exc:
            checks.append(Check("endpoints", path, "skip", str(exc)[:80]))
    return checks


# ----------------------------------------------------------------------
# Pipeline and confidence
# ----------------------------------------------------------------------


def validate(root, memory, probe_url: str | None = "http://localhost:3000") -> Report:
    """Run every stage and score the result.

    Args:
        root: Repository root.
        memory: `agent.memory.RepositoryMemory`, refreshed with any edits.
        probe_url: Base URL for the live endpoint probe, or None to skip it.

    Returns:
        A populated `Report`.
    """
    root = Path(root)
    report = Report()
    report.changed_files, report.added, report.removed = changed_files(root)

    if not report.changed_files:
        report.checks.append(Check("changes", "git diff", "skip",
                                   "no tracked changes detected"))
    report.checks += check_syntax(root, report.changed_files)
    report.checks += check_contracts(root, memory, report.changed_files)
    report.checks += check_routes(memory)
    report.checks += check_tests(root, memory)
    if probe_url:
        report.checks += check_endpoints(memory, probe_url)

    score_confidence(report, memory)
    return report


def score_confidence(report: Report, memory) -> None:
    """Fill in ``report.scores`` and ``report.confidence``.

    Four observable signals, none of them the model's own opinion:

    understanding  did it read the files it changed, and did it have the scan
    implementation was the change small and made with patches
    verification   did the structural checks pass
    tests          did a real suite pass
    """
    scores: dict[str, float] = {}

    # -- repository understanding -------------------------------------
    changed = [f for f in report.changed_files]
    if changed:
        seen = sum(1 for f in changed if f in memory.explored_files)
        ratio = seen / len(changed)
    else:
        ratio = 1.0
    understanding = W_UNDERSTANDING * (0.35 + 0.65 * ratio)
    scores["understanding"] = understanding

    # -- implementation quality ---------------------------------------
    # Rewriting far more than was added suggests a whole-file rewrite;
    # a clean additive patch removes little or nothing.
    churn = report.added + report.removed
    if churn == 0:
        quality = 0.4
    else:
        noise = report.removed / max(report.added, 1)
        quality = 1.0 if noise <= 0.35 else max(0.25, 1.0 - (noise - 0.35))
        if churn > 200:
            quality *= 0.7
    scores["implementation"] = W_IMPLEMENTATION * quality

    # -- verification --------------------------------------------------
    graded = [c for c in report.checks if c.status in ("pass", "fail")
              and c.stage in ("syntax", "contracts", "routes", "endpoints")]
    if graded:
        passed = sum(1 for c in graded if c.ok)
        verification = W_VERIFICATION * (passed / len(graded))
        # A contract or route failure is disqualifying, not merely averaged:
        # the application is broken.
        if any(c.stage in ("contracts", "routes") and not c.ok for c in graded):
            verification = min(verification, W_VERIFICATION * 0.3)
    else:
        verification = W_VERIFICATION * 0.5
    scores["verification"] = verification

    # -- tests ----------------------------------------------------------
    test_checks = [c for c in report.checks if c.stage == "tests"]
    if any(c.status == "fail" for c in test_checks):
        tests = 0.0
    elif any(c.ok for c in test_checks):
        tests = W_TESTS
    else:
        # No suite to run is not the agent's fault, but it is not evidence
        # either -- partial credit, and the report says so.
        tests = W_TESTS * 0.5
    scores["tests"] = tests

    report.scores = scores
    report.confidence = round(sum(scores.values()), 1)
