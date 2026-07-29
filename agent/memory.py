"""Repository Memory: what the run knows so far, and how to keep it cheap.

One `RepositoryMemory` object lives for the duration of a run. It wraps the
intelligence scan and the knowledge graph, and adds the run-time layer:

- architecture summary        (from `agent.intel`)
- module summaries            (deterministic, one per source file)
- dependency graph            (from `agent.graph`)
- frequently referenced symbols (incoming call/wiring edges)
- previously explored files   (every read_file / write_file the agent made)
- previously discovered APIs  (endpoints at scan time + any added by writes)

Its second job is **context compression** for `agent.loop`: when an old
read_file result would be elided from the request payload, the loop asks
memory for that file's *module summary* and sends that instead of a blank
placeholder -- the model keeps a compressed understanding of files it has
already seen, and re-opens one only when it needs exact text again.

After every write_file, the changed file is re-indexed and the graph rebuilt,
so summaries always describe the repository as it currently is.
"""

from __future__ import annotations

from pathlib import Path

from agent import graph as kg
from agent import intel as ri


class RepositoryMemory:
    """Everything the run has learned, queryable and current."""

    def __init__(self, root: str | Path, intelligence: ri.RepoIntelligence,
                 graph: kg.KnowledgeGraph) -> None:
        self.root = Path(root).expanduser().resolve()
        self.intel = intelligence
        self.graph = graph

        # path -> "read" | "written"
        self.explored_files: dict[str, str] = {}
        # endpoint details, scan-time plus anything discovered mid-run
        self.discovered_apis: list[str] = [e.detail for e in intelligence.endpoints]
        # tool_call_id -> repo-relative path, so elision can map an old
        # result back to the file it showed
        self.tool_call_file: dict[str, str] = {}
        # ids whose results have been compression-served at least once
        self.compressed_ids: set[str] = set()

        self.module_summaries: dict[str, str] = {}
        self._build_summaries()

    # -- summaries ------------------------------------------------------

    def _build_summaries(self) -> None:
        """One deterministic summary line per source module."""
        roles: dict[str, str] = {}
        for role, files in self.intel.file_roles.items():
            for f in files:
                roles.setdefault(f, role)

        for rel in self.graph.modules:
            parts: list[str] = []
            role = roles.get(rel)
            lines = self.graph.line_counts.get(rel)
            head = rel + (f" [{role}]" if role else "") + (f" ({lines} lines)" if lines else "")

            symbols = [s for s in self.intel.symbols if s.file == rel]
            exports = [s.name for s in symbols if s.kind in ("export", "function", "class", "model")]
            if exports:
                parts.append("defines " + ", ".join(dict.fromkeys(exports)))

            eps = [e.detail for e in self.intel.endpoints if e.file == rel and e.kind == "route"]
            if eps:
                parts.append("routes: " + ", ".join(eps))

            imports = self.graph.imports_of(rel)
            if imports:
                parts.append("imports " + ", ".join(sorted(set(imports))))

            db = self.graph.db_methods_in(rel)
            if db:
                parts.append("DB calls: " + ", ".join(db))

            self.module_summaries[rel] = head + (" -- " + "; ".join(parts) if parts else "")

    def summary_for(self, path: str) -> str | None:
        """The current one-line summary of a file, if it is a known module."""
        return self.module_summaries.get(path)

    # -- run-time recording ---------------------------------------------

    def record_read(self, path: str, tool_call_id: str) -> None:
        """Note that the agent has read a file."""
        self.explored_files.setdefault(path, "read")
        self.tool_call_file[tool_call_id] = path

    def record_write(self, path: str, tool_call_id: str) -> None:
        """Note a write and refresh everything derived from that file."""
        self.explored_files[path] = "written"
        self.tool_call_file[tool_call_id] = path
        self.refresh(path)

    def refresh(self, path: str) -> None:
        """Re-index one changed file and rebuild the graph and summaries.

        A full graph rebuild is deliberate: the repos this targets are small
        (the scan itself caps at 2,000 files), and correctness of edges into
        and out of the changed file matters more than saving milliseconds.
        """
        ri.reindex_file(self.intel, self.root, path)
        self.graph = kg.build_graph(self.root, self.intel)
        self._build_summaries()
        for ep in self.intel.endpoints:
            if ep.detail not in self.discovered_apis:
                self.discovered_apis.append(ep.detail)

    # -- derived views --------------------------------------------------

    def architecture_summary(self) -> str:
        """One line: what kind of repository this is."""
        i = self.intel
        return (
            f"{i.primary_language} / {', '.join(i.frameworks) or 'no framework'} / "
            f"{i.architecture}; db: {i.database}; orm: {i.orm}; "
            f"entry: {', '.join(i.entry_points) or 'unknown'}"
        )

    def frequent_symbols(self, top: int = 8) -> list[tuple[str, int]]:
        """Most-referenced symbols (incoming calls + route wirings)."""
        counts = self.graph.reference_counts()
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]

    # -- rendering ------------------------------------------------------

    def render_block(self) -> str:
        """Wiring map + hot symbols, appended to the first user message."""
        parts = [self.graph.render()]
        hot = self.frequent_symbols()
        if hot:
            parts.append(
                "- Frequently referenced symbols: "
                + ", ".join(f"{sym} ({n})" for sym, n in hot)
            )
        return "\n".join(p for p in parts if p)

    def stats_line(self) -> str:
        """One console line describing what memory did this run."""
        return (
            f"{len(self.explored_files)} files explored, "
            f"{len(self.module_summaries)} module summaries, "
            f"{len(self.compressed_ids)} results compressed, "
            f"{len(self.discovered_apis)} known endpoints"
        )


def build_memory(root: str | Path) -> RepositoryMemory:
    """Scan, graph, and wrap a repository in one call."""
    intelligence = ri.analyze(root)
    graph = kg.build_graph(root, intelligence)
    return RepositoryMemory(root, intelligence, graph)
