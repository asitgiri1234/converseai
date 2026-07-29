"""Repository Intelligence Engine: understand a repo before touching it.

One deterministic scan -- no LLM calls, no network -- producing a
`RepoIntelligence` object that answers the questions the agent used to spend
its first half-dozen turns discovering: language, framework, architecture,
package manager, database, ORM, entry points, file roles, API endpoints, and
a symbol index.

The object is built once per run by `agent.loop`, rendered compactly by
`render()`, and embedded in the *first user message*, where it survives for
every later phase (history elision only touches old tool results). Planning
therefore starts informed, and EXPLORE shrinks to targeted verification reads.

Honesty note: detection is heuristic. Symbols come from regexes, not an AST;
JavaScript and Python are covered well, other languages get counts only. The
prompt tells the model the summary is a pre-scan to verify, not gospel.

Standalone use (free -- burns no tokens):

    python -m agent.intel path/to/repo          # human-readable summary
    python -m agent.intel path/to/repo --json   # full object as JSON
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Directories never scanned. Superset of the tool layer's SKIP_DIRS.
SKIP_DIRS = {
    "node_modules", ".git", "dist", "build", "out", "coverage", "vendor",
    ".venv", "venv", "__pycache__", ".next", ".nuxt", "target", ".idea",
    ".vscode", "eggs", ".pytest_cache", ".mypy_cache",
}

MAX_FILES_SCANNED = 2_000
MAX_FILE_BYTES = 2 * 1024 * 1024

# Rendering caps, so the block stays cheap on a tight token budget.
MAX_LISTED = {"endpoints": 20, "symbols": 40, "files_per_role": 8, "deps": 25}

LANGUAGE_EXTENSIONS = {
    ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".jsx": "JavaScript",
    ".py": "Python", ".java": "Java", ".kt": "Kotlin", ".go": "Go",
    ".rb": "Ruby", ".php": "PHP", ".rs": "Rust", ".cs": "C#",
}

# dependency name -> (framework label, kind). Kind lets one dep imply
# several facts (mongoose is both an ODM and evidence of MongoDB).
DEP_SIGNALS = {
    # JavaScript
    "express": ("Express", "framework"),
    "next": ("Next.js", "framework"),
    "react": ("React", "framework"),
    "vue": ("Vue", "framework"),
    "@nestjs/core": ("NestJS", "framework"),
    "fastify": ("Fastify", "framework"),
    "koa": ("Koa", "framework"),
    "@hapi/hapi": ("Hapi", "framework"),
    "mongoose": ("Mongoose", "orm"),
    "sequelize": ("Sequelize", "orm"),
    "prisma": ("Prisma", "orm"),
    "@prisma/client": ("Prisma", "orm"),
    "typeorm": ("TypeORM", "orm"),
    "mongodb": ("MongoDB", "database"),
    "pg": ("PostgreSQL", "database"),
    "mysql": ("MySQL", "database"),
    "mysql2": ("MySQL", "database"),
    "sqlite3": ("SQLite", "database"),
    "better-sqlite3": ("SQLite", "database"),
    "redis": ("Redis", "database"),
    "ioredis": ("Redis", "database"),
    # Python
    "fastapi": ("FastAPI", "framework"),
    "django": ("Django", "framework"),
    "flask": ("Flask", "framework"),
    "sqlalchemy": ("SQLAlchemy", "orm"),
    "pymongo": ("MongoDB", "database"),
    "psycopg2": ("PostgreSQL", "database"),
    "psycopg2-binary": ("PostgreSQL", "database"),
    "motor": ("MongoDB", "database"),
    "peewee": ("Peewee", "orm"),
}

ORM_IMPLIES_DB = {
    "Mongoose": "MongoDB",
    "SQLAlchemy": "SQL (dialect from config)",
    "Sequelize": "SQL (dialect from config)",
    "Prisma": "per prisma/schema.prisma",
    "TypeORM": "SQL (dialect from config)",
}

# connection-string scheme -> database, for config-file evidence.
URL_SCHEMES = {
    "mongodb": "MongoDB",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "redis": "Redis",
}

CONFIG_NAMES = {
    "package.json", "tsconfig.json", "pyproject.toml", "requirements.txt",
    "setup.py", "setup.cfg", "pom.xml", "build.gradle", "go.mod",
    "cargo.toml", "gemfile", "composer.json", "dockerfile",
    "docker-compose.yml", "docker-compose.yaml", ".env.example",
    "manage.py", "settings.py",
}

# path-segment / filename evidence -> role
ROLE_HINTS = [
    ("models", "models"), (".model.", "models"),
    ("controllers", "controllers"), (".controller.", "controllers"),
    ("routes", "routes"), (".routes.", "routes"), ("urls.py", "routes"),
    ("services", "services"), (".service.", "services"),
    ("middleware", "middleware"), (".middleware.", "middleware"),
    ("views", "views"), ("utils", "utilities"), ("helpers", "utilities"),
    ("lib", "utilities"), ("config", "config"), ("settings", "config"),
]

_JS_ROUTE = re.compile(
    r"\b(?:app|router)\.(get|post|put|delete|patch|all|use)\(\s*['\"]([^'\"]+)['\"]"
)
_JS_EXPORT = re.compile(r"^(?:module\.)?exports\.(\w+)\s*=", re.M)
_JS_FUNCTION = re.compile(r"^(?:async\s+)?function\s+(\w+)\s*\(", re.M)
_JS_ARROW = re.compile(r"^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(", re.M)
_JS_CLASS = re.compile(r"^(?:export\s+)?class\s+(\w+)", re.M)
_JS_MODEL = re.compile(r"mongoose\.model\(\s*['\"](\w+)['\"]")
_PY_ROUTE = re.compile(r"^@(?:\w+)\.(get|post|put|delete|patch|route)\(\s*['\"]([^'\"]+)['\"]", re.M)
_PY_DEF = re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\(", re.M)
_PY_CLASS = re.compile(r"^class\s+(\w+)", re.M)
_URL_SCHEME = re.compile(r"\b(\w+)(?:\+\w+)?://")


@dataclass
class Symbol:
    """One named thing found in the repo."""

    name: str
    kind: str      # function | class | export | model | route | middleware
    file: str      # repo-relative, forward slashes
    line: int
    detail: str = ""  # e.g. "GET /notes" for routes


@dataclass
class RepoIntelligence:
    """Everything one scan learned about a repository."""

    root: str = ""
    languages: dict[str, int] = field(default_factory=dict)  # name -> file count
    primary_language: str = "Unknown"
    frameworks: list[str] = field(default_factory=list)
    architecture: str = "Unclassified"
    package_manager: str = "Unknown"
    database: str = "Not detected"
    orm: str = "Not detected"
    entry_points: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    file_roles: dict[str, list[str]] = field(default_factory=dict)
    endpoints: list[Symbol] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    files_scanned: int = 0
    notes: list[str] = field(default_factory=list)  # caveats worth surfacing

    # -- queries for later phases --------------------------------------

    def symbols_in(self, path: str) -> list[Symbol]:
        """All symbols found in one file."""
        return [s for s in self.symbols if s.file == path]

    def find_symbol(self, name: str) -> list[Symbol]:
        """Symbols matching a name, case-insensitive."""
        want = name.lower()
        return [s for s in self.symbols if s.name.lower() == want]

    def files_with_role(self, role: str) -> list[str]:
        """Files categorised under a role (models, routes, ...)."""
        return self.file_roles.get(role, [])

    def to_dict(self) -> dict:
        """The full object as plain data, for JSON output or storage."""
        as_sym = lambda s: {
            "name": s.name, "kind": s.kind, "file": s.file,
            "line": s.line, "detail": s.detail,
        }
        return {
            "root": self.root,
            "languages": self.languages,
            "primary_language": self.primary_language,
            "frameworks": self.frameworks,
            "architecture": self.architecture,
            "package_manager": self.package_manager,
            "database": self.database,
            "orm": self.orm,
            "entry_points": self.entry_points,
            "config_files": self.config_files,
            "dependencies": self.dependencies,
            "file_roles": self.file_roles,
            "endpoints": [as_sym(s) for s in self.endpoints],
            "symbols": [as_sym(s) for s in self.symbols],
            "files_scanned": self.files_scanned,
            "notes": self.notes,
        }

    # -- rendering ------------------------------------------------------

    def render(self) -> str:
        """Compact text block for the prompt. Capped per section."""
        lines: list[str] = ["REPOSITORY INTELLIGENCE (pre-scanned, heuristic -- verify before relying on it):"]

        langs = ", ".join(
            f"{name} ({count})" for name, count in
            sorted(self.languages.items(), key=lambda kv: -kv[1])
        ) or "none detected"
        lines.append(f"- Language: {self.primary_language} | all: {langs}")
        lines.append(f"- Framework: {', '.join(self.frameworks) or 'none detected'}")
        lines.append(f"- Architecture: {self.architecture}")
        lines.append(f"- Package manager: {self.package_manager}")
        lines.append(f"- Database: {self.database} | ORM/ODM: {self.orm}")
        lines.append(f"- Entry points: {', '.join(self.entry_points) or 'none detected'}")
        lines.append(f"- Config files: {', '.join(self.config_files[:10]) or 'none detected'}")

        deps = self.dependencies
        cap = MAX_LISTED["deps"]
        shown = ", ".join(deps[:cap]) + (f" (+{len(deps) - cap} more)" if len(deps) > cap else "")
        lines.append(f"- Dependencies: {shown or 'none detected'}")

        for role in ("models", "controllers", "routes", "services", "middleware", "views", "utilities"):
            files = self.file_roles.get(role, [])
            if not files:
                continue
            cap = MAX_LISTED["files_per_role"]
            shown = ", ".join(files[:cap]) + (f" (+{len(files) - cap} more)" if len(files) > cap else "")
            lines.append(f"- {role.capitalize()}: {shown}")

        if self.endpoints:
            lines.append("- API endpoints:")
            cap = MAX_LISTED["endpoints"]
            for ep in self.endpoints[:cap]:
                lines.append(f"    {ep.detail}  ({ep.file}:{ep.line})")
            if len(self.endpoints) > cap:
                lines.append(f"    (+{len(self.endpoints) - cap} more)")

        if self.symbols:
            lines.append("- Symbol index (name [kind] file:line):")
            cap = MAX_LISTED["symbols"]
            for sym in self.symbols[:cap]:
                lines.append(f"    {sym.name} [{sym.kind}] {sym.file}:{sym.line}")
            if len(self.symbols) > cap:
                lines.append(f"    (+{len(self.symbols) - cap} more)")

        for note in self.notes:
            lines.append(f"- Note: {note}")

        return "\n".join(lines)


# ----------------------------------------------------------------------
# Scanning
# ----------------------------------------------------------------------


def analyze(root: str | Path) -> RepoIntelligence:
    """Scan a repository and return its intelligence object.

    Pure function of the filesystem: no LLM, no network, deterministic for a
    given tree. Never raises for content it cannot parse -- unreadable or
    binary files are skipped and detection falls back to "Not detected".

    Args:
        root: Repository root directory.

    Returns:
        A populated `RepoIntelligence`.

    Raises:
        NotADirectoryError: If ``root`` is not an existing directory.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    intel = RepoIntelligence(root=str(root))
    files = _collect_files(root)
    intel.files_scanned = len(files)
    if len(files) >= MAX_FILES_SCANNED:
        intel.notes.append(f"scan capped at {MAX_FILES_SCANNED} files; index may be partial")

    rels = [f.relative_to(root).as_posix() for f in files]

    _detect_languages(intel, rels)
    _detect_manifest(intel, root, rels)
    _detect_roles_and_configs(intel, rels)
    _detect_structure(intel, rels)
    _index_symbols(intel, root, files, rels)
    _detect_database(intel, root, files, rels)
    _detect_entry_points(intel, root, rels)

    return intel


def _collect_files(root: Path) -> list[Path]:
    """Walk the tree, pruning SKIP_DIRS, capped at MAX_FILES_SCANNED."""
    found: list[Path] = []
    stack = [root]
    while stack and len(found) < MAX_FILES_SCANNED:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for entry in entries:
            if len(found) >= MAX_FILES_SCANNED:
                break
            if entry.is_dir():
                if entry.name not in SKIP_DIRS and not entry.is_symlink():
                    stack.append(entry)
            elif entry.is_file():
                found.append(entry)
    return found


def _read(path: Path) -> str:
    """Best-effort text read; empty string for binary/unreadable files."""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return ""
        raw = path.read_bytes()
    except OSError:
        return ""
    if b"\0" in raw[:8192]:
        return ""
    return raw.decode("utf-8", errors="replace")


def _detect_languages(intel: RepoIntelligence, rels: list[str]) -> None:
    counts: dict[str, int] = {}
    for rel in rels:
        lang = LANGUAGE_EXTENSIONS.get(Path(rel).suffix.lower())
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    intel.languages = counts
    if counts:
        intel.primary_language = max(counts.items(), key=lambda kv: kv[1])[0]


def _detect_manifest(intel: RepoIntelligence, root: Path, rels: list[str]) -> None:
    """Package manager, dependencies, and dep-implied framework/DB/ORM."""
    rel_set = {r.lower() for r in rels}
    deps: list[str] = []

    if "package.json" in rel_set:
        if "pnpm-lock.yaml" in rel_set:
            intel.package_manager = "pnpm"
        elif "yarn.lock" in rel_set:
            intel.package_manager = "yarn"
        else:
            intel.package_manager = "npm"
        try:
            manifest = json.loads(_read(root / "package.json") or "{}")
        except json.JSONDecodeError:
            manifest = {}
            intel.notes.append("package.json is not valid JSON")
        deps = sorted(
            list(manifest.get("dependencies", {}))
            + list(manifest.get("devDependencies", {}))
        )
    elif "pyproject.toml" in rel_set or "requirements.txt" in rel_set:
        if "poetry.lock" in rel_set:
            intel.package_manager = "poetry"
        elif "uv.lock" in rel_set:
            intel.package_manager = "uv"
        else:
            intel.package_manager = "pip"
        text = _read(root / "requirements.txt") + _read(root / "pyproject.toml")
        deps = sorted({
            m.group(1).lower()
            for m in re.finditer(r"^\s*([A-Za-z][A-Za-z0-9_.-]+)", text, re.M)
        } & set(DEP_SIGNALS))
        # requirements lines outside DEP_SIGNALS still matter as context:
        all_reqs = re.findall(r"^\s*([A-Za-z][A-Za-z0-9_.-]+)\s*[=<>!~;\[]?", _read(root / "requirements.txt"), re.M)
        deps = sorted(set(deps) | {d.lower() for d in all_reqs})
    elif "pom.xml" in rel_set:
        intel.package_manager = "maven"
        if "spring-boot" in _read(root / "pom.xml"):
            intel.frameworks.append("Spring Boot")
    elif "build.gradle" in rel_set or "build.gradle.kts" in rel_set:
        intel.package_manager = "gradle"
    elif "go.mod" in rel_set:
        intel.package_manager = "go modules"
    elif "cargo.toml" in rel_set:
        intel.package_manager = "cargo"

    intel.dependencies = deps
    for dep in deps:
        label, kind = DEP_SIGNALS.get(dep, (None, None))
        if not label:
            continue
        if kind == "framework" and label not in intel.frameworks:
            intel.frameworks.append(label)
        elif kind == "orm" and intel.orm == "Not detected":
            intel.orm = label
        elif kind == "database" and intel.database == "Not detected":
            intel.database = label


def _detect_roles_and_configs(intel: RepoIntelligence, rels: list[str]) -> None:
    roles: dict[str, list[str]] = {}
    configs: list[str] = []
    for rel in rels:
        lower = rel.lower()
        name = Path(lower).name
        if name in CONFIG_NAMES or name.endswith((".config.js", ".config.ts", ".config.mjs")):
            configs.append(rel)
        segments = lower.split("/")
        for hint, role in ROLE_HINTS:
            if hint in segments or hint in name:
                roles.setdefault(role, []).append(rel)
                break
    intel.file_roles = roles
    intel.config_files = configs


def _detect_structure(intel: RepoIntelligence, rels: list[str]) -> None:
    dirs = {seg for rel in rels for seg in rel.lower().split("/")[:-1]}
    if {"domain", "application", "infrastructure"} <= dirs:
        intel.architecture = "Clean Architecture (domain/application/infrastructure)"
    elif {"models", "controllers"} <= dirs and ({"routes"} & dirs or {"views"} & dirs):
        intel.architecture = "MVC (models / controllers / routes)"
    elif {"services"} <= dirs and ({"repositories"} & dirs or {"controllers"} & dirs):
        intel.architecture = "Layered (controllers / services / repositories)"
    elif {"features"} & dirs or {"modules"} & dirs:
        intel.architecture = "Feature-based (feature/module folders)"
    elif {"models"} & dirs or {"routes"} & dirs:
        intel.architecture = "Partial MVC (some conventional folders)"


def _index_symbols(
    intel: RepoIntelligence, root: Path, files: list[Path], rels: list[str]
) -> None:
    """Regex-based symbol and endpoint extraction for JS/TS and Python."""
    for path, rel in zip(files, rels):
        suffix = path.suffix.lower()
        if suffix in (".js", ".mjs", ".cjs", ".ts", ".jsx", ".tsx"):
            text = _read(path)
            if text:
                _index_js(intel, rel, text)
        elif suffix == ".py":
            text = _read(path)
            if text:
                _index_py(intel, rel, text)
    # Deduplicate endpoints (same method+path registered once per scan pass).
    seen: set[str] = set()
    unique: list[Symbol] = []
    for ep in intel.endpoints:
        if ep.detail not in seen:
            seen.add(ep.detail)
            unique.append(ep)
    intel.endpoints = unique


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _index_js(intel: RepoIntelligence, rel: str, text: str) -> None:
    for m in _JS_ROUTE.finditer(text):
        method, route = m.group(1).upper(), m.group(2)
        kind = "middleware" if m.group(1) == "use" else "route"
        detail = f"{method} {route}" if kind == "route" else f"USE {route}"
        intel.endpoints.append(Symbol(route, kind, rel, _line_of(text, m.start()), detail))
    for m in _JS_MODEL.finditer(text):
        intel.symbols.append(Symbol(m.group(1), "model", rel, _line_of(text, m.start())))
    for m in _JS_EXPORT.finditer(text):
        intel.symbols.append(Symbol(m.group(1), "export", rel, _line_of(text, m.start())))
    for m in _JS_FUNCTION.finditer(text):
        intel.symbols.append(Symbol(m.group(1), "function", rel, _line_of(text, m.start())))
    for m in _JS_ARROW.finditer(text):
        intel.symbols.append(Symbol(m.group(1), "function", rel, _line_of(text, m.start())))
    for m in _JS_CLASS.finditer(text):
        intel.symbols.append(Symbol(m.group(1), "class", rel, _line_of(text, m.start())))


def _index_py(intel: RepoIntelligence, rel: str, text: str) -> None:
    for m in _PY_ROUTE.finditer(text):
        method, route = m.group(1).upper(), m.group(2)
        intel.endpoints.append(
            Symbol(route, "route", rel, _line_of(text, m.start()), f"{method} {route}")
        )
    for m in _PY_DEF.finditer(text):
        intel.symbols.append(Symbol(m.group(1), "function", rel, _line_of(text, m.start())))
    for m in _PY_CLASS.finditer(text):
        intel.symbols.append(Symbol(m.group(1), "class", rel, _line_of(text, m.start())))


def _detect_database(
    intel: RepoIntelligence, root: Path, files: list[Path], rels: list[str]
) -> None:
    """Config-file connection strings beat dependency guesses."""
    if intel.database == "Not detected" and intel.orm in ORM_IMPLIES_DB:
        intel.database = ORM_IMPLIES_DB[intel.orm]

    candidates = [
        (p, r) for p, r in zip(files, rels)
        if "config" in r.lower() or r.lower().endswith((".env.example", "settings.py"))
    ]
    for path, rel in candidates[:50]:
        for m in _URL_SCHEME.finditer(_read(path)):
            db = URL_SCHEMES.get(m.group(1).lower())
            if db:
                intel.database = f"{db} (connection string in {rel})"
                return


def _detect_entry_points(intel: RepoIntelligence, root: Path, rels: list[str]) -> None:
    entries: list[str] = []
    rel_set = set(rels)

    if "package.json" in rel_set:
        try:
            manifest = json.loads(_read(root / "package.json") or "{}")
        except json.JSONDecodeError:
            manifest = {}
        main = manifest.get("main")
        if main and main in rel_set:
            entries.append(main)
        start = (manifest.get("scripts") or {}).get("start", "")
        m = re.search(r"(?:node|nodemon)\s+([\w./-]+\.[cm]?js)", start)
        if m and m.group(1) in rel_set and m.group(1) not in entries:
            entries.append(m.group(1))

    for conventional in ("manage.py", "main.py", "app.py", "server.js", "index.js", "src/index.js", "src/main.ts"):
        if conventional in rel_set and conventional not in entries:
            entries.append(conventional)

    intel.entry_points = entries


def reindex_file(intel: RepoIntelligence, root: str | Path, rel: str) -> None:
    """Refresh one file's symbols and endpoints after it changed on disk.

    Used by `agent.memory` when the agent writes a file mid-run, so the
    intelligence object keeps describing the repository as it now is rather
    than as it was at scan time.

    Args:
        intel: The object to update in place.
        root: Repository root.
        rel: Repo-relative path (forward slashes) of the changed file.
    """
    root = Path(root).expanduser().resolve()
    intel.symbols = [s for s in intel.symbols if s.file != rel]
    intel.endpoints = [e for e in intel.endpoints if e.file != rel]

    path = root / rel
    suffix = path.suffix.lower()
    text = _read(path) if path.is_file() else ""
    if text:
        if suffix in (".js", ".mjs", ".cjs", ".ts", ".jsx", ".tsx"):
            _index_js(intel, rel, text)
        elif suffix == ".py":
            _index_py(intel, rel, text)

    seen: set[str] = set()
    unique: list[Symbol] = []
    for ep in intel.endpoints:
        if ep.detail not in seen:
            seen.add(ep.detail)
            unique.append(ep)
    intel.endpoints = unique


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan a repository and print its intelligence summary."
    )
    parser.add_argument("repo", help="Repository root to analyze.")
    parser.add_argument("--json", action="store_true", help="Emit the full object as JSON.")
    args = parser.parse_args()

    result = analyze(args.repo)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.render())
        print(f"\n({result.files_scanned} files scanned)")
