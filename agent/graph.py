"""Repository Knowledge Graph: how the pieces of a repo connect.

Built on top of `agent.intel`'s scan, this module extracts *relationships*:
which module imports which, which route is handled by which controller
function, which functions call which, where the database is touched, and
which files read configuration.

Like the intelligence scan it is deterministic Python -- regex extraction
plus import-path resolution, no LLM, no network. And like the symbol index it
is honest about being an approximation: call attribution assumes a function's
body runs from its definition line to the next definition in the same file,
which fits flat controller-style code well and nested code loosely.

Node identity:
    modules  -> repo-relative path            "app/models/note.model.js"
    symbols  -> path::name                    "app/controllers/x.js::create"
    routes   -> METHOD path                   "GET /notes"

Edge kinds:
    imports      module -> module      (detail: local alias)
    reads_config module -> module      (import whose target is a config file)
    registers    route  -> symbol      (route handler wiring)
    calls        symbol -> symbol      (function body references)
    db_call      symbol -> module      (model method use; detail: method)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from agent.intel import RepoIntelligence, Symbol, _read

_JS_REQUIRE = re.compile(
    r"(?:const|let|var)\s+(\w+)\s*=\s*require\(\s*['\"]([^'\"]+)['\"]\s*\)"
)
_JS_REQUIRE_BARE = re.compile(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)")
_JS_IMPORT = re.compile(
    r"import\s+(?:(\w+)|\{[^}]*\}|\*\s+as\s+(\w+))\s+from\s+['\"]([^'\"]+)['\"]"
)
_JS_ROUTE_HANDLER = re.compile(
    r"\b(?:app|router)\.(get|post|put|delete|patch|all)\(\s*['\"]([^'\"]+)['\"]\s*,\s*([\w.$]+)\s*\)"
)
_PY_IMPORT = re.compile(r"^(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.M)

# Model-object methods that mean "this line talks to the database".
DB_METHODS = {
    "find", "findone", "findbyid", "findbyidandupdate", "findbyidandremove",
    "findbyidanddelete", "findoneandupdate", "deleteone", "deletemany",
    "updateone", "updatemany", "countdocuments", "estimateddocumentcount",
    "aggregate", "create", "insertmany", "save", "distinct", "exists",
    "filter", "get", "all", "count",  # common ORM spellings
}


@dataclass
class Edge:
    """One directed relationship."""

    src: str
    dst: str
    kind: str      # imports | reads_config | registers | calls | db_call
    detail: str = ""


@dataclass
class KnowledgeGraph:
    """The relationships, queryable and renderable."""

    edges: list[Edge] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)   # files that appear as nodes
    line_counts: dict[str, int] = field(default_factory=dict)

    # -- queries --------------------------------------------------------

    def imports_of(self, path: str) -> list[str]:
        """Files this module imports (imports + reads_config)."""
        return [
            e.dst for e in self.edges
            if e.src == path and e.kind in ("imports", "reads_config")
        ]

    def importers_of(self, path: str) -> list[str]:
        """Files that import this module -- its parents."""
        return [
            e.src for e in self.edges
            if e.dst == path and e.kind in ("imports", "reads_config")
        ]

    def handler_of(self, route_detail: str) -> str | None:
        """Symbol id handling a route like ``GET /notes``, if wired."""
        for e in self.edges:
            if e.kind == "registers" and e.src == route_detail:
                return e.dst
        return None

    def calls_from(self, symbol_id: str) -> list[Edge]:
        """Outgoing calls and db_calls of one function."""
        return [
            e for e in self.edges
            if e.src == symbol_id and e.kind in ("calls", "db_call")
        ]

    def db_methods_in(self, path: str) -> list[str]:
        """Database methods used anywhere in a file, deduplicated."""
        seen: list[str] = []
        for e in self.edges:
            if e.kind == "db_call" and e.src.startswith(path + "::"):
                if e.detail not in seen:
                    seen.append(e.detail)
        return seen

    def reference_counts(self) -> dict[str, int]:
        """Incoming calls/registers per symbol id -- who gets used most."""
        counts: dict[str, int] = {}
        for e in self.edges:
            if e.kind in ("calls", "registers"):
                counts[e.dst] = counts.get(e.dst, 0) + 1
        return counts

    # -- rendering ------------------------------------------------------

    def render(self, max_lines: int = 30) -> str:
        """Compact wiring map for the prompt."""
        lines: list[str] = []

        module_edges: dict[str, list[str]] = {}
        for e in self.edges:
            if e.kind in ("imports", "reads_config"):
                module_edges.setdefault(e.src, []).append(e.dst)
        if module_edges:
            lines.append("- Module graph (file -> imports):")
            for src in sorted(module_edges):
                lines.append(f"    {src} -> {', '.join(sorted(set(module_edges[src])))}")

        wired = [e for e in self.edges if e.kind == "registers"]
        if wired:
            lines.append("- Route wiring (route -> handler):")
            for e in wired:
                lines.append(f"    {e.src} -> {e.dst}")

        db = {}
        for e in self.edges:
            if e.kind == "db_call":
                file = e.src.split("::")[0]
                db.setdefault(file, set()).add(e.detail)
        if db:
            lines.append("- Database touchpoints (file -> methods):")
            for file in sorted(db):
                lines.append(f"    {file}: {', '.join(sorted(db[file]))}")

        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"    (+{len(lines) - max_lines} more lines)"]
        return "\n".join(lines)


def build_graph(root: str | Path, intel: RepoIntelligence) -> KnowledgeGraph:
    """Construct the knowledge graph for a scanned repository.

    Args:
        root: Repository root (same one `agent.intel.analyze` scanned).
        intel: The intelligence object whose symbols/endpoints to connect.

    Returns:
        A populated `KnowledgeGraph`. Never raises for unparsable content;
        what cannot be resolved simply produces no edge.
    """
    root = Path(root).expanduser().resolve()
    graph = KnowledgeGraph()
    config_set = {c.lower() for c in intel.config_files}

    # Symbols grouped per file, sorted by line, for body-span attribution.
    per_file: dict[str, list[Symbol]] = {}
    for s in intel.symbols:
        per_file.setdefault(s.file, []).append(s)
    for symbols in per_file.values():
        symbols.sort(key=lambda s: s.line)

    indexable = {".js", ".mjs", ".cjs", ".ts", ".jsx", ".tsx", ".py"}
    files = sorted(
        {s.file for s in intel.symbols}
        | {e.file for e in intel.endpoints}
        | {r for r in per_file}
        | {c for c in intel.config_files if Path(c).suffix.lower() in indexable}
        | set(intel.entry_points)
    )

    for rel in files:
        path = root / rel
        if path.suffix.lower() not in indexable:
            continue
        text = _read(path)
        if not text:
            continue
        graph.modules.append(rel)
        graph.line_counts[rel] = text.count("\n") + 1

        aliases = _import_edges(graph, rel, text, root, config_set)
        _route_edges(graph, rel, text, aliases, per_file)
        _call_edges(graph, rel, text, aliases, per_file, intel)

    return graph


def _resolve_js(base: Path, spec: str, root: Path) -> str | None:
    """Resolve a relative require/import spec to a repo-relative path."""
    if not spec.startswith("."):
        return None  # bare specifier -> node_modules, not a repo edge
    candidate = (base.parent / spec).resolve()
    for attempt in (candidate, candidate.with_suffix(candidate.suffix + ".js")
                    if candidate.suffix else candidate.with_suffix(".js"),
                    candidate / "index.js"):
        try:
            if attempt.is_file():
                return attempt.relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
    return None


def _resolve_py(spec: str, root: Path) -> str | None:
    """Resolve a Python dotted import to a repo file, if it is one."""
    tail = spec.replace(".", "/")
    for attempt in (root / f"{tail}.py", root / tail / "__init__.py"):
        if attempt.is_file():
            return attempt.relative_to(root).as_posix()
    return None


def _import_edges(
    graph: KnowledgeGraph, rel: str, text: str, root: Path, config_set: set[str]
) -> dict[str, str]:
    """Add import edges for one file; return alias -> target-path map."""
    aliases: dict[str, str] = {}
    path = root / rel

    if path.suffix.lower() == ".py":
        for m in _PY_IMPORT.finditer(text):
            target = _resolve_py(m.group(1) or m.group(2), root)
            if target and target != rel:
                kind = "reads_config" if target.lower() in config_set else "imports"
                graph.edges.append(Edge(rel, target, kind))
        return aliases

    for m in _JS_REQUIRE.finditer(text):
        target = _resolve_js(path, m.group(2), root)
        if target and target != rel:
            aliases[m.group(1)] = target
            kind = "reads_config" if target.lower() in config_set else "imports"
            graph.edges.append(Edge(rel, target, kind, detail=m.group(1)))
    for m in _JS_REQUIRE_BARE.finditer(text):
        target = _resolve_js(path, m.group(1), root)
        if target and target != rel and target not in aliases.values():
            kind = "reads_config" if target.lower() in config_set else "imports"
            graph.edges.append(Edge(rel, target, kind))
    for m in _JS_IMPORT.finditer(text):
        name = m.group(1) or m.group(2)
        target = _resolve_js(path, m.group(3), root)
        if target and target != rel:
            if name:
                aliases[name] = target
            kind = "reads_config" if target.lower() in config_set else "imports"
            graph.edges.append(Edge(rel, target, kind, detail=name or ""))
    return aliases


def _route_edges(
    graph: KnowledgeGraph, rel: str, text: str,
    aliases: dict[str, str], per_file: dict[str, list[Symbol]],
) -> None:
    """Wire ``app.get('/x', mod.handler)`` routes to their handler symbols."""
    for m in _JS_ROUTE_HANDLER.finditer(text):
        method, route, handler = m.group(1).upper(), m.group(2), m.group(3)
        route_id = f"{method} {route}"
        if "." in handler:
            alias, func = handler.split(".", 1)
            target_file = aliases.get(alias)
            if target_file:
                graph.edges.append(
                    Edge(route_id, f"{target_file}::{func}", "registers")
                )
                continue
        # Locally defined handler (function in the same file).
        if any(s.name == handler for s in per_file.get(rel, [])):
            graph.edges.append(Edge(route_id, f"{rel}::{handler}", "registers"))


def _call_edges(
    graph: KnowledgeGraph, rel: str, text: str,
    aliases: dict[str, str], per_file: dict[str, list[Symbol]],
    intel: RepoIntelligence,
) -> None:
    """Attribute calls to functions by body span (def line -> next def line)."""
    symbols = per_file.get(rel, [])
    if not symbols:
        return
    lines = text.splitlines()

    # Which alias points at a model file? Those member calls are db_calls.
    model_files = {
        s.file for s in intel.symbols if s.kind == "model"
    } | set(intel.file_roles.get("models", []))

    local_names = {s.name for s in symbols}

    for i, sym in enumerate(symbols):
        start = sym.line - 1
        end = symbols[i + 1].line - 1 if i + 1 < len(symbols) else len(lines)
        body = "\n".join(lines[start:end])
        src_id = f"{rel}::{sym.name}"

        # alias.method( ...) -- either a db call or a cross-module call.
        for m in re.finditer(r"\b(\w+)\.(\w+)\s*\(", body):
            alias, method = m.group(1), m.group(2)
            target_file = aliases.get(alias)
            if not target_file:
                continue
            if target_file in model_files and method.lower() in DB_METHODS:
                graph.edges.append(Edge(src_id, target_file, "db_call", detail=method))
            elif any(s.name == method for s in per_file.get(target_file, [])):
                graph.edges.append(
                    Edge(src_id, f"{target_file}::{method}", "calls")
                )

        # bare localName( ...) -- same-file function calls.
        for name in local_names:
            if name != sym.name and re.search(rf"\b{re.escape(name)}\s*\(", body):
                graph.edges.append(Edge(src_id, f"{rel}::{name}", "calls"))
