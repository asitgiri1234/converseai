"""Semantic repository search over the knowledge graph.

Answers questions rather than matching patterns: "where are notes created?",
"which endpoint deletes a note?", "which controller uses this model?",
"which functions update the database?".

Why not embeddings
------------------
Groq serves no embedding model (the account exposes `client.embeddings` but
the model list contains none), and a local encoder means torch plus a model
download -- roughly 2 GB against a project whose entire dependency list is
`groq` and `python-dotenv`. So this is a *structural* semantic search: it
understands the question's intent and resolves it against facts already
extracted by `agent.intel` and `agent.graph` -- routes, handlers, exports,
imports, database calls, file roles.

That trades open-vocabulary recall for precision on the vocabulary that
matters in a web codebase, and it costs no tokens, no network, and no
dependency. Where it genuinely cannot know -- a concept the repository never
names -- it returns nothing rather than a plausible guess.

Six capabilities:
    structural   file roles and architecture drive matches
    symbol-aware exports, functions, classes, models with file:line
    route-aware  HTTP method inferred from the verb in the question
    controller-  handler resolution through route wiring
      aware
    model-aware  importers of a model, and database methods used
    ranked       multi-signal scores, each with its evidence shown

Standalone (free -- no tokens, no key):

    python -m agent.search path/to/repo "which endpoint deletes a note?"
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

from agent.intel import _read

# Action families. Each maps to the words a codebase spells it with and the
# HTTP method it conventionally uses.
ACTIONS: dict[str, dict] = {
    "create": {
        "words": {"create", "creating", "created", "add", "adding", "new",
                  "insert", "save", "saving", "post", "register", "make"},
        "methods": {"POST"},
        "db": {"save", "create", "insertmany", "insert"},
    },
    "read": {
        "words": {"read", "get", "fetch", "fetching", "find", "finding",
                  "list", "listing", "retrieve", "retrieving", "show",
                  "query", "querying", "search", "view", "load"},
        "methods": {"GET"},
        "db": {"find", "findone", "findbyid", "aggregate", "countdocuments",
               "distinct", "exists", "estimateddocumentcount"},
    },
    "update": {
        "words": {"update", "updating", "updated", "edit", "editing",
                  "modify", "modifying", "change", "changing", "put",
                  "patch", "set"},
        "methods": {"PUT", "PATCH"},
        "db": {"findbyidandupdate", "updateone", "updatemany",
               "findoneandupdate", "save"},
    },
    "delete": {
        "words": {"delete", "deleting", "deleted", "remove", "removing",
                  "destroy", "drop", "erase", "del"},
        "methods": {"DELETE"},
        "db": {"findbyidandremove", "findbyidanddelete", "deleteone",
               "deletemany", "remove"},
    },
}

# What kind of thing the question is asking for.
ARTIFACTS = {
    "route": {"route", "routes", "endpoint", "endpoints", "api", "apis",
              "url", "path", "handler"},
    "controller": {"controller", "controllers"},
    "model": {"model", "models", "schema", "schemas", "collection", "entity"},
    "service": {"service", "services"},
    "middleware": {"middleware", "middlewares"},
    "config": {"config", "configuration", "settings", "env"},
    "function": {"function", "functions", "method", "methods", "export",
                 "exports"},
}

# Concepts that live in file *content* rather than in structure. Each maps to
# the tokens a codebase actually uses for it.
CONCEPTS = {
    "authentication": {"auth", "authenticate", "authentication", "login",
                       "logout", "signin", "signup", "jwt", "token",
                       "passport", "bcrypt", "password", "session",
                       "credential", "oauth", "authorize", "authorization"},
    "database": {"mongoose", "mongo", "db", "database", "schema", "model",
                 "collection", "query", "connection"},
    "validation": {"validate", "validation", "valid", "sanitize", "schema",
                   "required", "constraint"},
    "error handling": {"error", "err", "catch", "exception", "throw",
                       "status", "500", "404"},
    "logging": {"log", "logger", "logging", "console", "winston", "morgan"},
    "pagination": {"page", "paginate", "pagination", "limit", "skip",
                   "offset"},
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "where", "which", "what",
    "who", "how", "does", "do", "did", "in", "on", "at", "of", "for", "to",
    "this", "that", "these", "those", "it", "its", "and", "or", "we", "i",
    "my", "our", "be", "been", "can", "could", "should", "would", "there",
    "here", "when", "why", "with", "from", "by", "use", "used", "uses",
    "using", "implemented", "implement", "implementation", "handle",
    "handles", "handled", "handling", "code", "codebase", "repo",
    "repository", "app", "application",
}

# Score weights. Kept as named constants so ranking is tunable and the
# rendered evidence can explain exactly what fired.
W_ROUTE_METHOD = 6.0
W_ROUTE_ENTITY = 4.0
W_NAME_EXACT = 6.0
W_NAME_FUZZY = 3.0
W_ACTION_IN_NAME = 5.0
W_ROLE_MATCH = 4.0
W_DB_METHOD = 5.5
W_RELATION = 7.0
W_CONCEPT_HIT = 4.0
W_PATH_TOKEN = 2.0
W_SUMMARY_TOKEN = 1.5
W_ARTIFACT_KIND = 2.5
# Penalty when a symbol matches the question's subject but not its verb.
W_NAME_NOT_ACTION = 2.0

FUZZY_THRESHOLD = 0.82
MAX_CONTENT_FILES = 60


@dataclass
class Intent:
    """What a natural-language question is asking for."""

    raw: str
    tokens: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)

    @property
    def methods(self) -> set[str]:
        """HTTP methods implied by the question's verb."""
        out: set[str] = set()
        for action in self.actions:
            out |= ACTIONS[action]["methods"]
        return out

    @property
    def db_methods(self) -> set[str]:
        """Database methods implied by the question's verb."""
        out: set[str] = set()
        for action in self.actions:
            out |= ACTIONS[action]["db"]
        return out


@dataclass
class Hit:
    """One ranked search result."""

    kind: str          # route | symbol | file
    label: str         # what to show
    location: str      # file:line or file
    score: float
    reasons: list[str] = field(default_factory=list)

    def render(self) -> str:
        return (
            f"[{self.kind}] {self.label}  ({self.location})  score {self.score:.1f}\n"
            f"      why: {'; '.join(self.reasons)}"
        )


def tokenize(text: str) -> list[str]:
    """Split into lowercase word tokens, dropping stopwords."""
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def _split_identifier(name: str) -> set[str]:
    """Break camelCase / snake_case / kebab-case into lowercase parts."""
    parts = re.split(r"[_\-.]", name)
    out: set[str] = set()
    for part in parts:
        out |= {p.lower() for p in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", part)}
    out.add(name.lower())
    return {p for p in out if p}


def _variants(tokens: list[str]) -> set[str]:
    """Tokens plus crude singular forms.

    "deletes"/"creates"/"updates" are how questions are actually phrased, and
    a vocabulary listing only the bare verb silently misclassifies them as
    domain nouns -- which pushes the right route down the ranking.
    """
    out = set(tokens)
    for token in tokens:
        if token.endswith("es") and len(token) > 4:
            out.add(token[:-2])
        if token.endswith("s") and len(token) > 3:
            out.add(token[:-1])
    return out


def parse_query(query: str) -> Intent:
    """Extract actions, artifact types, concepts, and entities from a question.

    Args:
        query: The natural-language question.

    Returns:
        A populated `Intent`. Unmatched tokens become entities -- the domain
        nouns ("notes", "user") the question is about.
    """
    tokens = tokenize(query)
    token_set = _variants(tokens)
    intent = Intent(raw=query, tokens=tokens)

    for action, spec in ACTIONS.items():
        if token_set & spec["words"]:
            intent.actions.append(action)

    for artifact, words in ARTIFACTS.items():
        if token_set & words:
            intent.artifacts.append(artifact)

    for concept, words in CONCEPTS.items():
        if token_set & words:
            intent.concepts.append(concept)

    claimed: set[str] = set()
    for spec in ACTIONS.values():
        claimed |= spec["words"]
    for words in ARTIFACTS.values():
        claimed |= words
    for words in CONCEPTS.values():
        claimed |= words

    for token in tokens:
        singular = token[:-1] if token.endswith("s") and len(token) > 3 else token
        if token in claimed or singular in claimed:
            continue
        if singular not in intent.entities:
            intent.entities.append(singular)
    return intent


def _entity_match(intent: Intent, text: str) -> bool:
    """True if any domain noun from the question appears in ``text``."""
    lowered = text.lower()
    return any(entity in lowered for entity in intent.entities)


def _fuzzy(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def search(query: str, memory, limit: int = 8) -> list[Hit]:
    """Rank repository elements against a natural-language question.

    Args:
        query: The question.
        memory: An `agent.memory.RepositoryMemory`.
        limit: Maximum hits to return.

    Returns:
        Hits sorted by descending score. Empty when nothing scores above
        zero -- an honest "not found" rather than the least-bad match.
    """
    intent = parse_query(query)
    intel, graph = memory.intel, memory.graph
    hits: list[Hit] = []

    hits += _score_routes(intent, intel, graph)
    hits += _score_symbols(intent, intel, graph, memory)
    hits += _score_relations(intent, intel, graph, memory)
    if intent.concepts:
        hits += _score_concepts(intent, memory)

    # Collapse duplicates (a route and its handler can both surface), keeping
    # the higher score and merging the evidence.
    best: dict[str, Hit] = {}
    for hit in hits:
        key = f"{hit.kind}|{hit.label}"
        existing = best.get(key)
        if existing is None or hit.score > existing.score:
            if existing:
                hit.reasons = list(dict.fromkeys(hit.reasons + existing.reasons))
            best[key] = hit
        else:
            existing.reasons = list(dict.fromkeys(existing.reasons + hit.reasons))

    ranked = sorted(best.values(), key=lambda h: (-h.score, h.label))
    return [h for h in ranked if h.score > 0][:limit]


def _score_routes(intent: Intent, intel, graph) -> list[Hit]:
    """Route-aware scoring: HTTP method from the verb, path from the noun."""
    hits: list[Hit] = []
    wants_route = "route" in intent.artifacts or not intent.artifacts

    for endpoint in intel.endpoints:
        if endpoint.kind != "route":
            continue
        score = 0.0
        reasons: list[str] = []
        method = endpoint.detail.split(" ", 1)[0]

        if method in intent.methods:
            score += W_ROUTE_METHOD
            reasons.append(f"HTTP {method} matches '{intent.actions[0]}'")
        if intent.entities and _entity_match(intent, endpoint.name):
            score += W_ROUTE_ENTITY
            reasons.append(f"path mentions {', '.join(intent.entities)}")
        if wants_route and score > 0:
            score += W_ARTIFACT_KIND
            reasons.append("question asks for an endpoint")

        handler = graph.handler_of(endpoint.detail)
        if score > 0:
            label = endpoint.detail + (f" -> {handler}" if handler else "")
            hits.append(Hit("route", label, f"{endpoint.file}:{endpoint.line}",
                            score, reasons))
    return hits


def _score_symbols(intent: Intent, intel, graph, memory) -> list[Hit]:
    """Symbol-aware scoring over exports, functions, classes, and models."""
    hits: list[Hit] = []
    roles: dict[str, str] = {}
    for role, files in intel.file_roles.items():
        for f in files:
            roles.setdefault(f, role)

    action_words: set[str] = set()
    for action in intent.actions:
        action_words |= ACTIONS[action]["words"]

    for symbol in intel.symbols:
        score = 0.0
        reasons: list[str] = []
        parts = _split_identifier(symbol.name)

        if action_words & parts:
            score += W_ACTION_IN_NAME
            reasons.append(f"name '{symbol.name}' means {intent.actions[0]}")
        entity_overlap = set(intent.entities) & parts
        if entity_overlap:
            score += W_NAME_EXACT
            reasons.append(f"name matches {', '.join(sorted(entity_overlap))}")
            # Named the right subject but not the action asked about: a Note
            # model is not the answer to "where are notes created".
            if intent.actions and not (action_words & parts):
                score -= W_NAME_NOT_ACTION
        else:
            for entity in intent.entities:
                if _fuzzy(entity, symbol.name.lower()) >= FUZZY_THRESHOLD:
                    score += W_NAME_FUZZY
                    reasons.append(f"name ~ '{entity}'")
                    break

        role = roles.get(symbol.file)
        if role and role.rstrip("s") in {a.rstrip("s") for a in intent.artifacts}:
            score += W_ROLE_MATCH
            reasons.append(f"lives in the {role} layer")
        if symbol.kind == "model" and "model" in intent.artifacts:
            score += W_ARTIFACT_KIND
            reasons.append("is a model definition")

        path_tokens = _split_identifier(symbol.file.replace("/", "_"))
        overlap = set(intent.entities) & path_tokens
        if overlap:
            score += W_PATH_TOKEN
            reasons.append(f"path mentions {', '.join(overlap)}")

        # Database-aware: does this function actually touch the DB the way
        # the question describes?
        if intent.db_methods or "database" in intent.concepts:
            sym_id = f"{symbol.file}::{symbol.name}"
            used = [e.detail for e in graph.calls_from(sym_id) if e.kind == "db_call"]
            if used:
                matching = [m for m in used if m.lower() in intent.db_methods]
                if matching:
                    score += W_DB_METHOD
                    reasons.append(f"calls {', '.join(matching)}")
                elif "database" in intent.concepts:
                    score += W_DB_METHOD / 2
                    reasons.append(f"touches the database ({', '.join(used)})")

        if score > 0:
            hits.append(Hit("symbol", f"{symbol.name} [{symbol.kind}]",
                            f"{symbol.file}:{symbol.line}", score, reasons))
    return hits


def _score_relations(intent: Intent, intel, graph, memory) -> list[Hit]:
    """Graph-aware answers: which module uses which, via import edges.

    Handles "which controller uses this model" and its siblings by resolving
    the mentioned artifact to a file, then walking importers.
    """
    hits: list[Hit] = []
    if not intent.artifacts:
        return hits

    # Which file is the question *about*? Prefer a model when one is named.
    targets: list[str] = []
    for role in ("model", "service", "config"):
        if role in intent.artifacts:
            targets += intel.file_roles.get(role + "s", []) + intel.file_roles.get(role, [])
    if intent.entities:
        targets = [
            t for t in targets
            if _entity_match(intent, t) or not intent.entities
        ] or targets

    asking_for = [a for a in intent.artifacts if a in ("controller", "service", "route", "middleware")]
    if not targets or not asking_for:
        return hits

    roles: dict[str, str] = {}
    for role, files in intel.file_roles.items():
        for f in files:
            roles.setdefault(f, role)

    for target in dict.fromkeys(targets):
        for importer in dict.fromkeys(graph.importers_of(target)):
            role = roles.get(importer, "")
            if not any(role.startswith(a) for a in asking_for):
                continue
            summary = memory.summary_for(importer) or importer
            hits.append(Hit(
                "file", importer, importer, W_RELATION,
                [f"imports {target}", f"is in the {role} layer",
                 summary.split(" -- ")[-1][:90]],
            ))
    return hits


def _score_concepts(intent: Intent, memory) -> list[Hit]:
    """Content scan for concepts that live in code text, not structure.

    Only runs when the question names a concept, and only over indexed source
    files, so it stays cheap. Returns nothing when the repository genuinely
    does not implement the concept -- which is the correct answer.
    """
    hits: list[Hit] = []
    root = Path(memory.root)

    for concept in intent.concepts:
        if concept == "database":
            continue  # already covered structurally by db_call edges
        vocabulary = CONCEPTS[concept]
        for rel in list(memory.graph.modules)[:MAX_CONTENT_FILES]:
            text = _read(root / rel)
            if not text:
                continue
            found = sorted({
                w for w in vocabulary
                if re.search(rf"\b{re.escape(w)}\b", text, re.I)
            })
            if not found:
                continue
            line = 1
            match = re.search(rf"\b{re.escape(found[0])}\b", text, re.I)
            if match:
                line = text.count("\n", 0, match.start()) + 1
            hits.append(Hit(
                "file", f"{rel} ({concept})", f"{rel}:{line}",
                W_CONCEPT_HIT + min(len(found), 4),
                [f"mentions {', '.join(found[:5])}"],
            ))
    return hits


def render(query: str, hits: list[Hit], intent: Intent | None = None) -> str:
    """Format ranked hits for the agent or the terminal."""
    intent = intent or parse_query(query)
    lines = [f"SEMANTIC SEARCH: {query}"]
    detected = []
    if intent.actions:
        detected.append("action=" + "/".join(intent.actions))
    if intent.artifacts:
        detected.append("looking for=" + "/".join(intent.artifacts))
    if intent.entities:
        detected.append("about=" + "/".join(intent.entities))
    if intent.concepts:
        detected.append("concept=" + "/".join(intent.concepts))
    lines.append("  interpreted as: " + (", ".join(detected) or "no structural intent detected"))

    if not hits:
        lines.append(
            "  No matches. This repository does not appear to contain that -- "
            "do not assume it exists; use search_code for a literal string."
        )
        return "\n".join(lines)

    for rank, hit in enumerate(hits, 1):
        lines.append(f"  {rank}. {hit.render()}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    from agent.memory import build_memory

    parser = argparse.ArgumentParser(description="Semantic search over a repository.")
    parser.add_argument("repo", help="Repository root.")
    parser.add_argument("question", help="Natural-language question.")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    print(render(args.question, search(args.question, build_memory(args.repo), args.limit)))
