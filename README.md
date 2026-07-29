# converseai

An AI coding agent for existing **Node.js / Express / MongoDB** codebases,
running on **Groq**. Give it a vague product request — *"let users organise
their notes better"* — and it explores the repository, writes a plan, makes the
edits, verifies them, and reports what it changed.

It is a single-file-per-concern, dependency-light implementation: five Python
modules, two third-party packages, no framework. Everything the agent does is
narrated to the terminal as it happens.

---

## Contents

- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [The agent workflow](#the-agent-workflow)
- [Repository Intelligence Engine](#repository-intelligence-engine)
- [The planning stage](#the-planning-stage)
- [Knowledge graph and repository memory](#knowledge-graph-and-repository-memory)
- [Context compression and token benchmark](#context-compression-and-token-benchmark)
- [How repository exploration works](#how-repository-exploration-works)
- [Semantic search](#semantic-search)
- [Patch-based editing](#patch-based-editing)
- [Tool reference](#tool-reference)
- [Message formats](#message-formats)
- [Safety model](#safety-model)
- [Console output](#console-output)
- [Context and rate-limit management](#context-and-rate-limit-management)
- [Verified run](#verified-run)
- [Running the result locally](#running-the-result-locally)
- [Frontend](#frontend)
- [Troubleshooting](#troubleshooting)
- [Extending the agent](#extending-the-agent)
- [Development and testing](#development-and-testing)
- [Design trade-offs](#design-trade-offs)
- [Limitations](#limitations)
- [Project layout](#project-layout)

---

## Quick start

Requires Python 3.11+ and a [Groq API key](https://console.groq.com/keys).
Node.js is needed only if you want the agent's `node --check` verification step
to work.

```bash
git clone https://github.com/asitgiri1234/converseai
cd converseai

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt

cp .env.example .env              # then edit .env and paste your key
```

Clone something for it to work on, and run:

```bash
git clone https://github.com/callicoder/node-easy-notes-app ./target-repo

python main.py \
  --repo ./target-repo \
  --task "Add a GET /notes/count endpoint that returns the total number of notes as JSON."
```

Review what it did with `git -C target-repo diff`. The agent never commits, so
`git -C target-repo checkout -- .` throws the run away.

> **Work on a throwaway clone.** The agent writes files and runs shell commands
> in the directory you point it at. See [Safety model](#safety-model).

---

## CLI reference

```
python main.py --repo PATH --task TEXT [--model ID] [--temperature F] [--max-turns N]
```

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--repo` | yes | — | Repository the agent may read and modify. Every tool is confined to it. |
| `--task` | yes | — | The product request, in plain language. Vague is fine. |
| `--model` | no | `$GROQ_MODEL`, else `llama-3.3-70b-versatile` | Any Groq model with tool-calling support. |
| `--temperature` | no | `0.2` | Sampling temperature. Low suits a coding agent. |
| `--max-turns` | no | `40` | Hard cap on model turns before the run aborts. |
| `--no-intel` | no | off | Skip the repository pre-scan; explore with tools only. |

**Exit codes**

| Code | Meaning |
| --- | --- |
| `0` | The agent finished and emitted its summary. |
| `1` | The run failed — turn cap hit, or a Groq API error that survived retries. |
| `2` | Bad input — `--repo` is not a directory, or `GROQ_API_KEY` is unset. |
| `130` | Interrupted with Ctrl-C. |

---

## Configuration

Read from the environment, or from a `.env` file loaded automatically at import
(`.env` is gitignored; `.env.example` is the template).

| Variable | Required | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | yes | Groq credentials. |
| `GROQ_MODEL` | no | Overrides the default model; `--model` overrides this. |

### Choosing a model

Any Groq model with tool-calling support works. These three were tested against
the notes app and all drive the loop correctly:

| Model | Free-tier TPM | Notes |
| --- | --- | --- |
| `llama-3.3-70b-versatile` | 12,000 | **Default.** Most token headroom, which matters most. Follows the phase structure loosely. |
| `openai/gpt-oss-120b` | 8,000 | Best judgement of the three in testing — the [verified run](#verified-run) used it. Tighter token budget. |
| `qwen/qwen3.6-27b` | 8,000 | Works; least tested. |

Token-per-minute headroom matters more than raw capability here, because an
agent replays its whole conversation on every single turn. See
[Context and rate-limit management](#context-and-rate-limit-management).

---

## Architecture

Five modules, one direction of control. `loop` is the only stateful piece: the
chat-completions API is stateless, so the loop owns the conversation history and
replays it every turn.

```
  main.py ──── parses --repo/--task, binds the repo root, maps exit codes
     │
     ▼
  agent/loop.py ───────────────────────────────────────────┐
     │  history: [user task + intel, assistant, tools, …]  │
     │                                                     │
     ├──▶ agent/memory.py     RepositoryMemory ───────────┐│
     │      ├─ agent/intel.py   one deterministic pre-scan ││
     │      └─ agent/graph.py   relationships + call edges ││
     │      (no LLM calls)    module summaries feed both   ││
     │                        the first message and the    ││
     │                        loop's context compression ◀─┘│
     ├──▶ agent/prompts.py    system prompt (phases, rules)│
     │                                                     │
     ├──▶ agent/llm.py ──────▶ Groq chat completions       │
     │      one request         (OpenAI-compatible)         │
     │    ◀── message.content + message.tool_calls          │
     │                                                     │
     └──▶ agent/tools.py ──▶ target repo (filesystem, shell)
            dispatch(name, input) → str ────────────────────┘
                         role="tool" messages re-enter history
```

### Module responsibilities

| Module | Owns | Deliberately does *not* |
| --- | --- | --- |
| `main.py` | Argument parsing, `--repo` validation, binding the tool root, exit codes | Know anything about turns or tools |
| `agent/intel.py` | The pre-scan: architecture detection, file roles, symbol index, rendering | Call the LLM, the network, or mutate anything |
| `agent/graph.py` | Relationships: imports, route wiring, call edges, DB touchpoints | Read the filesystem twice — it works from the scan |
| `agent/search.py` | Question intent, multi-signal ranking, evidence strings | Call a model or an embedding service |
| `agent/patch.py` | Locating an edit region, applying it, proving nothing else moved | Touch the filesystem — it is pure string work |
| `agent/memory.py` | Module summaries, explored files, discovered APIs, compression source | Decide when to compress — the loop asks it |
| `agent/llm.py` | Client construction, model defaults, one request/response | Loop, retry, or interpret the response |
| `agent/prompts.py` | The stable system prompt and the opening user message | Know about tools or the API |
| `agent/tools.py` | Tool schemas, filesystem/shell execution, path and command safety | Know that an LLM exists |
| `agent/loop.py` | History, turn sequencing, retries, context elision, console output | Implement any tool itself |

`agent/llm.py` returns the raw `ChatCompletion` unmodified, so the loop can
inspect `choices[0].message.tool_calls` and `finish_reason` itself. Swapping
providers means rewriting `llm.py` and the two format-handling helpers in
`loop.py` — nothing else.

---

## The agent workflow

The system prompt drives five explicit phases.

| Phase | What the agent does |
| --- | --- |
| **1. EXPLORE** | With the intelligence block present: targeted reads of only the files the task touches, located via the symbol index. Without it: `list_files`, then `package.json`, then following `require()` chains. |
| **2. PLAN** | Emits a nine-section `PLAN:` block — understanding, interpretation, strategies, trade-offs, choice, files, risks, verification, impact — *before* the first edit, cited from the intelligence object. See [The planning stage](#the-planning-stage). |
| **3. IMPLEMENT** | `write_file`, preserving existing routes, exports, response shapes, and field names, matching the surrounding style. Additive and backwards-compatible. |
| **4. VERIFY** | Re-reads every modified file, runs `node --check` on each changed file, runs `npm test` / `npm run lint` if they exist. |
| **5. SUMMARIZE** | A `SUMMARY:` section listing every file changed and why. |

### Handling ambiguity

The agent runs unattended, so it is instructed never to ask a clarifying
question. Given *"let users organise their notes better"* it picks the smallest
reasonable interpretation, writes it into the plan as an explicit
`Assumption:`, and builds that. You review the assumption in the output rather
than answering a prompt mid-run.

### Why `SUMMARY:` is the stop condition

A naive loop stops when the model replies with text instead of a tool call. That
breaks here: `PLAN:` *is* a text-only turn, so the run would end right after
planning and before any edit.

So a text-only turn ends the run **only** when it contains `SUMMARY:`.
Otherwise the loop appends a short nudge and continues — capped at
`MAX_NUDGES = 3`, so a model that never emits the marker still terminates
rather than burning all 40 turns.

`SUMMARY_MARKER` is defined once in `prompts.py` and imported by `loop.py`.
Renaming it in one file and not the other silently breaks termination.

---

## Repository Intelligence Engine

Before the first model turn, `agent/intel.py` runs **one deterministic scan**
of the repository — pure Python, no LLM calls, no network, zero API tokens —
and builds a `RepoIntelligence` object:

| Capability | How it is detected |
| --- | --- |
| Primary language | File-extension counts (JS/TS, Python, Java, Go, Ruby, PHP, Rust, C#, Kotlin) |
| Framework | Dependency signals — `express`, `next`, `@nestjs/core`, `fastapi`, `django`, `flask`, Spring Boot via `pom.xml`, … |
| Architecture | Directory shape — `models+controllers+routes` → MVC, `services+repositories` → Layered, `features/modules` → Feature-based, `domain+application+infrastructure` → Clean |
| Package manager | Manifest + lockfile — npm/yarn/pnpm, pip/poetry/uv, maven, gradle, go modules, cargo |
| Database | Connection-string schemes in config files (`mongodb://`, `postgres://`, …) — which **beat** dependency guesses — else ORM implication |
| ORM / ODM | Dependency signals — Mongoose, Sequelize, Prisma, TypeORM, SQLAlchemy, Peewee, … |
| Entry points | `package.json` `main` + `scripts.start`, then conventional names (`server.js`, `manage.py`, `main.py`, …) |
| File roles | Path/filename hints → models, controllers, routes, services, middleware, views, utilities, config |
| API endpoints | `app.get('/path', …)` / `router.post(…)` in JS, `@app.get("/path")` decorators in Python — with `file:line` |
| Symbol index | Functions, classes, exports, Mongoose models — regex-extracted, each with `file:line` |

The object is **reusable**: `find_symbol(name)`, `symbols_in(file)`,
`files_with_role(role)`, and `to_dict()` let later phases query it instead of
re-scanning. Its `render()` output (compact, per-section caps, ~350 tokens for
the notes app) is embedded in the **first user message** — deliberately, for
two reasons: the system prompt must stay byte-identical across runs, and the
first user message is never touched by history elision, so the map stays
visible to PLAN, IMPLEMENT, and VERIFY alike.

Run it standalone (free, no key needed):

```bash
python -m agent.intel ./target-repo          # human-readable summary
python -m agent.intel ./target-repo --json   # full object as JSON
```

Output for the notes app:

```
REPOSITORY INTELLIGENCE (pre-scanned, heuristic -- verify before relying on it):
- Language: JavaScript | all: JavaScript (5)
- Framework: Express
- Architecture: MVC (models / controllers / routes)
- Package manager: npm
- Database: MongoDB (connection string in config/database.config.js) | ORM/ODM: Mongoose
- Entry points: server.js
- API endpoints:
    GET /notes/count  (app/routes/note.routes.js:11)
    ...
- Symbol index (name [kind] file:line):
    count [export] app/controllers/note.controller.js:124
    ...
```

**Honesty:** detection is heuristic. Symbols come from regexes, not an AST;
JavaScript and Python are indexed well, other languages get language/manifest
detection only. The block is labelled "pre-scanned, heuristic — verify" and
the prompt tells the model to treat it as a map, not gospel. A scan failure
never kills a run — the agent falls back to tool-driven exploration, and
`--no-intel` forces that mode.

---

## The planning stage

PLAN is not "list the files you'll touch" — it is a nine-section engineering
document the agent must produce before its first edit, grounded in the
intelligence object rather than in assumptions.

| # | Section | What it must contain |
| --- | --- | --- |
| 1 | `UNDERSTANDING` | What the repo is, and the parts this task touches — cited as `file:line` symbol entries, `route -> handler` wirings, or module summaries |
| 2 | `INTERPRETATION` | What the request concretely means here, what it excludes, and any `Assumption:` |
| 3 | `STRATEGIES` | At least two genuinely different approaches, labelled A and B |
| 4 | `TRADE-OFFS` | For each: what it costs and buys — effort, blast radius, performance, fit with conventions |
| 5 | `CHOSEN` | Which one, and why |
| 6 | `FILES` | Every file to create or modify, with a reason, using exact paths from the scan |
| 7 | `RISKS` | Specific routes, exports, response shapes or schema fields at risk — including which modules import the file being changed |
| 8 | `VERIFICATION` | The exact checks to run afterwards |
| 9 | `IMPACT` | What changes for existing behaviour, and what is guaranteed unchanged |

**The grounding rule** is the point of the redesign: every factual claim about
the repository must come from the intelligence block, the module graph, a
module summary, or a file actually read this run. If the agent cannot support
a claim it must read the file first or mark the line `unverified`.

### Enforcement

`PLAN_SECTIONS` in `agent/prompts.py` is the single source of truth — the
test suite asserts every entry also appears in `SYSTEM_PROMPT`, so the spec
and the validator cannot drift. The loop scores each plan and logs it:

```
  ✓ plan: all 9 sections present
  ✗ plan: 3/9 sections -- missing STRATEGIES, TRADE-OFFS, CHOSEN, RISKS, VERIFICATION, IMPACT
```

An incomplete plan gets **one** targeted repair asking only for the missing
sections — and it rides on the nudge that a text-only PLAN turn triggers
anyway, so it costs no extra request. Matching is deliberately forgiving
(`TRADE-OFFS`, `Trade offs:`, `**TRADEOFFS**` all count); rejecting a good
plan over punctuation would waste a turn on a tight token budget.

### Plan condensation

A nine-section plan is a large *assistant* message, and assistant messages are
never elided — so an old plan would otherwise cost its full size on every
later request. Once `CHOSEN` is written, the deliberation has done its job, so
a plan older than six messages is condensed to `CHOSEN`, `FILES`, `RISKS`,
`VERIFICATION`, `IMPACT`:

```
PLAN (condensed -- strategy chosen, deliberation dropped):
5. CHOSEN: Strategy A, one round trip.
6. FILES: app/controllers/note.controller.js - add stats; ...
```

Measured at ~46% smaller on a real plan. This was not optional: the first live
run of the nine-section format **413'd at turn 10** with a 9,648-token request.
With condensation the same point in the run was 7,887 tokens, under the 8k
limit.

### What it does and does not buy

Verified live: `gpt-oss-120b` and `llama-3.3-70b-versatile` both produced all
nine sections, with genuinely distinct strategies (a Mongoose aggregation
pipeline versus fetching documents and computing in application code) and
trade-offs naming round-trips and memory cost.

**It does not make a weak model safe.** In one run `llama-3.3-70b` wrote a
plan whose `IMPACT` section claimed existing behaviour was untouched, then
converted `note.routes.js` from `module.exports = (app) => {...}` to an
`express.Router`. `node --check` passed — the file was valid syntax — and the
app crashed at startup, because `server.js` calls that module as a function.
The plan structure did not prevent it. Two prompt guards were added in
response:

| Guard | Result |
| --- | --- |
| Never change a module's export shape; check the graph's importers first | **Worked** — the crash stopped reproducing, app boots, existing endpoints 200 |
| Register literal routes before parameterised ones on the same prefix | **Did not work** on `llama-3.3-70b` — it still appended `/notes/stats` after `/notes/:noteId`, so the endpoint 404s. `gpt-oss-120b` gets this right unprompted |

The honest reading: structured planning improves the *reasoning that is
visible to you* and catches some classes of error, but it is not a substitute
for a capable model, and a plan asserting safety is not evidence of safety.
Read the diff.

---

## Knowledge graph and repository memory

The intelligence scan says *what exists*. `agent/graph.py` and
`agent/memory.py` add *how it connects* and *what this run has learned*.

### Knowledge graph (`agent/graph.py`)

A directed graph over three node types — modules (`app/models/note.model.js`),
symbols (`path::name`), and routes (`GET /notes`) — with five edge kinds:

| Edge | Meaning | Extracted from |
| --- | --- | --- |
| `imports` | module → module | `require()` / `import` with the specifier resolved to a real repo file |
| `reads_config` | module → config file | an import whose target is a detected config file |
| `registers` | route → handler symbol | `app.get('/notes', notes.findAll)` resolved through the import alias |
| `calls` | symbol → symbol | identifiers referenced inside a function's body span |
| `db_call` | symbol → model file | model-object methods (`find`, `countDocuments`, `findByIdAndUpdate`, …) |

Queries: `imports_of`, `importers_of` (parent modules), `handler_of(route)`,
`calls_from(symbol)`, `db_methods_in(file)`, `reference_counts()`.

Built on the notes app it resolves the whole request path —
`server.js → routes → controller → model`, plus `server.js → config` — and
wires all eight routes to their exact handler symbols:

```
- Module graph (file -> imports):
    app/controllers/note.controller.js -> app/models/note.model.js
    app/routes/note.routes.js -> app/controllers/note.controller.js
    server.js -> app/routes/note.routes.js, config/database.config.js
- Route wiring (route -> handler):
    GET /notes/count -> app/controllers/note.controller.js::count
    GET /notes/:noteId -> app/controllers/note.controller.js::findOne
- Database touchpoints (file -> methods):
    app/controllers/note.controller.js: countDocuments, find, findById, …
```

### Repository memory (`agent/memory.py`)

One `RepositoryMemory` per run, holding the architecture summary, per-module
summaries, the dependency graph, frequently referenced symbols, previously
explored files, and previously discovered APIs.

A **module summary** is deterministic, one line, and always current:

```
app/controllers/note.controller.js [controllers] (169 lines) -- defines create,
findAll, findOne, update, delete, count, search, recent; imports
app/models/note.model.js; DB calls: find, findById, findByIdAndUpdate,
findByIdAndRemove, countDocuments
```

After every `write_file`, memory re-indexes that file and rebuilds the graph,
so a summary reflects the repository as it *now* is — a handler the agent
added this run shows up in its own summary immediately.

---

## Context compression and token benchmark

The loop's elision pass now has two tiers. When memory can attribute a stale
`read_file` result to a file, the contents are replaced by that file's
**current module summary** rather than a blank placeholder:

```
[compressed -- current summary of this file: app/controllers/note.controller.js
[controllers] (169 lines) -- defines create, findAll, …; DB calls: find, …
Re-read the file only if you need its exact contents.]
```

The system prompt tells the model these blocks are current and that it should
re-read only when it needs exact text to edit. Results memory cannot
attribute fall back to the generic notice.

### Benchmark

Replaying a 12-call run (the real read/write/verify sequence against the notes
app, real file contents, 13 requests, ~4 chars/token):

| Policy | Total sent | Peak request |
| --- | --- | --- |
| A — no elision, no intel (naive) | ~62,900 tok | ~8,200 tok |
| B — generic elision, no intel | ~41,900 tok | ~4,400 tok |
| C — generic elision + intel block | ~51,300 tok | ~5,100 tok |
| D — **summary compression + graph** | ~52,900 tok | ~5,400 tok |

**Read this honestly.** Against the naive baseline the current design sends
**16% fewer total tokens and a 34% smaller peak request** — and peak is what
413s an 8k-TPM tier. But against policy B, the pre-intelligence design, D is
**26% larger**: the intelligence block, graph, and summaries are real tokens
that a blank placeholder does not spend.

What that buys is fewer *turns*. Live runs on the notes app went from 20 turns
/ 82k prompt tokens (no intel) to 11 turns / 42–48k across three separate
tasks, because the agent stops spending its opening turns rediscovering
layout. The per-request overhead is paid back several times over by the
requests never made — but on a repo where the agent would only ever open one
file, policy B would genuinely be cheaper. `--no-intel` gives you that.

---

## How repository exploration works

Exploration is **intelligence-first, tool-verified**. The pre-scan answers the
questions the agent used to spend its opening turns discovering; the tools
then verify the specific files the task touches. The LLM still decides every
step — there is just a map in its hand now.

Measured on the notes app (same model, comparable small-endpoint tasks):

| | Without intel | With intel |
| --- | --- | --- |
| First edit at turn | 8 | 4 |
| `PLAN:` emitted at turn | 7 | 3 |
| Opening discovery calls | `list_files`, `package.json`, model, controller, routes, `server.js` | controller, routes only |
| Total turns / prompt tokens | 20 / 82,482 | 11 / 41,295 |

With `--no-intel` the old behaviour returns: `list_files`, then
`package.json`, then following `require()` chains — each step a round trip.
That mode remains the fallback whenever the scan fails or the flag is set.

---

## Semantic search

`agent/search.py` answers questions instead of matching patterns:

```bash
python -m agent.search ./target-repo "which endpoint deletes a note?"
```

```
SEMANTIC SEARCH: which endpoint deletes a note?
  interpreted as: action=delete, looking for=route, about=note
  1. [route] DELETE /notes/:noteId -> app/controllers/note.controller.js::delete
        (app/routes/note.routes.js:17)  score 12.5
      why: HTTP DELETE matches 'delete'; path mentions note; question asks for an endpoint
  2. [symbol] delete [export]  (app/controllers/note.controller.js:97)  score 12.5
      why: name 'delete' means delete; path mentions note; calls findByIdAndRemove
```

### Why not embeddings

Groq serves **no embedding model** — the account exposes `client.embeddings`
but the model list contains none, so there is nothing to call. The
alternative is a local encoder, which means torch plus a model download:
roughly 2 GB against a project whose entire dependency list is `groq` and
`python-dotenv`.

So this is *structural* semantic search. It parses the question's intent and
resolves it against facts already extracted by the intelligence scan and the
knowledge graph — routes, handlers, exports, imports, database calls, file
roles. That trades open-vocabulary recall for precision on the vocabulary a
web codebase actually uses, and it costs no tokens, no network, and no
dependency.

### The six capabilities

| Capability | How it works |
| --- | --- |
| Structural | File roles and architecture from the scan drive matches |
| Symbol-aware | Exports, functions, classes and models, each with `file:line` |
| Route-aware | The verb in the question maps to an HTTP method: *deletes* → `DELETE` |
| Controller-aware | Routes resolve through the graph's `registers` edges to their handler symbol |
| Model-aware | "which controller uses this model" walks `importers_of` and filters by role |
| Ranked | Multi-signal scores, each hit carrying the evidence that produced it |

### How ranking works

Intent parsing splits the question into **actions** (create/read/update/
delete, with synonyms and third-person forms — *creates*, *deletes*),
**artifacts** (route, controller, model, service, middleware, config,
function), **concepts** (authentication, database, validation, error
handling, logging, pagination), and **entities** — whatever domain nouns are
left over.

Scoring then combines named, tunable signals: HTTP-method match (6.0), graph
relation (7.0), exact name match (6.0), action-in-name (5.0), database method
match (5.5), role match (4.0), fuzzy name via stdlib `difflib` (3.0), and
path/summary token overlap. A symbol that matches the question's *subject*
but not its *verb* is penalised — a `Note` model is not the answer to "where
are notes created", the `create` handler is.

### `search_repo` vs `search_code`

Both are available; they answer different questions.

| | `search_repo` | `search_code` |
| --- | --- | --- |
| Input | Plain-English question | Regex |
| Answers | *Where does X happen?* | *Where does this string appear?* |
| Output | Ranked hits with evidence | `file:line:match` |
| Use when | Locating behaviour | You need a literal string or exact pattern |

The tool reuses the run's existing scan (the loop injects it via
`set_search_index`, so it also sees files written during the run) and builds
its own lazily when used standalone.

### It says "no" when the answer is no

Asked *"where is authentication implemented?"* against the notes app — which
has no auth — it returns:

```
  No matches. This repository does not appear to contain that -- do not
  assume it exists; use search_code for a literal string.
```

That wording is deliberate. A search tool that always returns its least-bad
match teaches the model to hallucinate a subsystem; the same query for *error
handling*, which the app does implement, correctly returns the controller.

---

## Patch-based editing

The agent used to change code by retyping whole files. That was the single
largest source of diff noise in this project: a model asked to add one
endpoint would rewrite the file and "tidy" everything it retyped. One recorded
run produced **105 insertions and 90 deletions** for a change needing about a
dozen lines. Worse, a retyped file can silently drop a function or convert a
module's export shape — which
[happened, and killed the app at startup](#the-planning-stage).

`agent/patch.py` replaces that with anchored replacement. The caller supplies
a snippet of the *existing* text plus its replacement; everything outside the
matched span is preserved byte for byte.

### The strategy

**1. Locate the smallest region.** The model supplies the smallest snippet
that is unique in the file. Matching is strict, in two tiers:

| Situation | Result |
| --- | --- |
| Anchor appears verbatim, exactly once | apply (`exact`) |
| Appears once ignoring trailing whitespace | apply (`whitespace`) |
| Appears more than once | **refuse** — "appears 3 times… include more context" |
| Does not appear | **refuse** — names the closest line in the file |

Refusals are errors the model can act on, never silent guesses. Applying an
edit to the wrong one of three identical blocks is worse than not editing.

**2. Preserve everything else.** The new file is
`before[:start] + new_text + before[end:]`. Untouched code is not reformatted,
reordered, or lost — not by policy but by construction.

**3. Verify only the intended change.** `verify_untouched` asserts that every
byte outside the replaced span is identical. It holds by construction, so it
guards against a future refactor quietly breaking the guarantee the engine
rests on.

**4. Show the diff back.** Every patch returns its own unified diff, so the
model sees exactly what it did and can catch a wrong-anchor edit immediately
rather than at verification.

`insert_after` handles pure additions — a route beside the existing ones, a
handler after the last one — without retyping the anchor. It always inserts at
a **line boundary**: an anchor ending mid-line would otherwise splice new text
into the middle of a statement. (That bug was caught by the test suite, not by
review.)

### `write_file` is now guarded

Overwriting an existing file requires `overwrite=true`. The default path for
an existing file is `edit_file`, and the refusal says so:

```
Error: app/routes/note.routes.js already exists. Use edit_file to change part
of it -- that keeps the rest of the file byte-identical and the diff small.
```

Creating new files is unaffected.

### Measured: before and after

Same feature (`GET /notes/count`), same repository. The whole-file numbers are
from **real recorded agent runs**, not estimates:

| Approach | Diff | Churn |
| --- | --- | --- |
| Whole-file rewrite (`llama-3.3-70b`, recorded) | +105 / −90 | 195 |
| Whole-file rewrite (`gpt-oss-120b`, recorded) | +15 / −0 | 15 |
| **Patch-based** (through the real loop and tools) | **+13 / −0** | **13** |

93% smaller than the bad case, 13% smaller than the good one. The honest
reading: when a model already writes a clean additive rewrite, patching wins
little. Its value is that it makes the catastrophic case **structurally
impossible** rather than dependent on model discipline.

The resulting diff is two pure-addition hunks, and the patched app was
verified running: `GET /notes/count` → `{"count":10}`, `GET /notes` → 200,
`node --check` clean on both files, route registered before `/notes/:noteId`,
and `module.exports = (app) => {` preserved.

---

## Tool reference

Tools are declared in OpenAI function format — Groq speaks the OpenAI
chat-completions dialect — and executed client-side by `dispatch(name, input)`,
which **always returns a string and never raises**.

| Tool | Arguments | Behaviour | Limits |
| --- | --- | --- | --- |
| `list_files` | `path` (optional) | Indented recursive tree, directories before files | Skips `node_modules` and `.git`; 1,000 entries; won't follow symlinked dirs |
| `read_file` | `path` | Contents with line numbers | 500 lines; refuses binaries (NUL-byte check) |
| `search_code` | `pattern` | Regex across all text files → `file:line:match` | 50 results; skips binaries and files > 2 MB; match lines clipped to 200 chars |
| `search_repo` | `question` | Plain-English question → ranked hits with locations and evidence, see [Semantic search](#semantic-search) | 8 hits; returns nothing rather than a weak guess |
| `edit_file` | `path`, `old_text`, `new_text` | Replace one located region; everything else stays byte-identical | Anchor must be unique; returns the diff. See [Patch-based editing](#patch-based-editing) |
| `insert_after` | `path`, `anchor`, `text` | Insert lines after a unique anchor, at a line boundary | For pure additions |
| `write_file` | `path`, `content`, `overwrite` | Create a **new** file | Refuses an existing file unless `overwrite=true` |
| `run_shell` | `command` | Runs via the platform shell with `cwd` = repo root | 30 s timeout; output capped at 20,000 chars; returns exit code + stdout + stderr |

Failures are returned as strings prefixed `Error:` — unknown tool, bad
arguments, missing file, path escape, refused command — so the model reads what
went wrong and adjusts instead of the run crashing.

---

## Message formats

Groq speaks the OpenAI chat-completions dialect. Three shapes matter, and
getting any of them wrong produces a confusing API error rather than a clean
failure — so they are worth stating explicitly.

**1. Tool declarations** (`agent/tools.py`, built by the `_function` helper):

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read a text file and return its contents with line numbers…",
    "parameters": {
      "type": "object",
      "properties": { "path": { "type": "string", "description": "…" } },
      "required": ["path"]
    }
  }
}
```

**2. The assistant turn**, rebuilt field by field before it re-enters history.
`content` may be empty when the model only calls tools, and `arguments` is a
**JSON string**, not an object:

```json
{
  "role": "assistant",
  "content": "",
  "tool_calls": [
    {
      "id": "fc_6febfae0",
      "type": "function",
      "function": { "name": "read_file", "arguments": "{\"path\":\"package.json\"}" }
    }
  ]
}
```

**3. The tool reply** — one per call, carrying the matching id:

```json
{ "role": "tool", "tool_call_id": "fc_6febfae0", "content": "     1\t{\n…" }
```

Three rules the loop enforces:

- **Every call gets exactly one reply.** Leave a `tool_call_id` unanswered and
  the next request is rejected.
- **The assistant message must keep its `tool_calls`.** Appending only the text
  leaves the tool replies answering nothing.
- **Arguments are model-generated JSON and may be malformed.** They are parsed
  with `json.loads`; on failure the loop returns a `role: "tool"` message
  describing the parse error, so the model can re-issue the call.

The assistant entry is reconstructed rather than dumped from the SDK model, so
provider-specific fields (`reasoning`, `refusal`, null `function_call`) never
leak back into the next request.

---

## Safety model

**Path containment.** Every model-supplied path is resolved — symlinks included
— and rejected unless it lands inside the repo root. `../` traversal, absolute
paths pointing elsewhere, and symlinks aimed outward all fail with
`Error: path escapes the repository root`.

**Command screening.** `run_shell` refuses:

- privilege escalation — `sudo`, `su`, `doas`, `runas`
- recursive/forced deletes whose target resolves outside the repo
- `mkfs`, `dd of=/dev/…`, `shutdown`, `reboot`, and fork bombs

> ### This is a guardrail, not a sandbox
>
> Anything that passes the screen runs with your full user privileges. It does
> not stop `curl … | sh`, a `git push --force`, or a Python one-liner that
> deletes files. **Run the agent against a throwaway clone**, or inside a
> container, if that matters. The screen exists to catch obvious accidents, not
> a determined adversary.

---

## Console output

Each turn prints the model's reasoning preview, any assistant text, and every
tool call with truncated input and a one-line result preview.

```
────────────────────────────────────────────────────────────────────────
  converseai  →  groq / openai/gpt-oss-120b  (temp=0.2)
  repo: /home/me/target-repo
  task: Add a GET /notes/count endpoint that returns the total numb…
────────────────────────────────────────────────────────────────────────

[turn 4/40]
  We need to add GET /notes/count endpoint returning total number of notes.
  · read_file(path="app/controllers/note.controller.js")
    ✓ 1 const Note = require('../models/note.model.js');  [116 lines]

[turn 5/40]
  PLAN:
  1. Feature: Add a GET `/notes/count` endpoint returning `{ count: <number> }`.
  2. Files to modify:
  - `app/routes/note.routes.js` – register the route before `/:noteId`.
  - `app/controllers/note.controller.js` – add a `count` export.
  3. Assumption: the response is an object with a single `count` field.
  · no SUMMARY: yet -- continuing

[turn 12/40]
  · run_shell(command="node --check app/controllers/note.controller.js")
    ✓ exit code: 0  [2 lines]

────────────────────────────────────────────────────────────────────────
  ✓ done
────────────────────────────────────────────────────────────────────────
SUMMARY:
- app/routes/note.routes.js – registered GET /notes/count before /:noteId.
- app/controllers/note.controller.js – added a count handler using countDocuments().
────────────────────────────────────────────────────────────────────────
  20 turns  ·  18 tool calls  ·  558s  ·  82,482 in / 4,600 out tokens
────────────────────────────────────────────────────────────────────────
```

Long string arguments are clipped to 60 characters, so a `write_file` carrying a
whole file cannot flood the screen. Tool timings appear only for calls slower
than 0.5 s. The box-drawing glyphs fall back to ASCII when the terminal cannot
encode them — otherwise piping the output to a file would crash on Windows.

---

## Context and rate-limit management

This is the part that decides whether a run finishes, so it is worth
understanding.

**The problem.** An agent replays its entire history every turn. Read one
120-line file twice and most of a small tier's per-minute token budget is gone.

**Groq bills `max_completion_tokens` up front**, against the same per-minute
budget as the prompt. So `prompt + max_completion_tokens` must clear the TPM
ceiling, and the ceiling is squeezed from both directions:

| `max_completion_tokens` | Failure |
| --- | --- |
| Too high | `413` — request larger than the TPM limit, before the model runs |
| Too low | `write_file`'s whole-file argument truncates mid-JSON → `400 tool_use_failed` |

`DEFAULT_MAX_TOKENS = 1800` threads that gap for a file of roughly 120 lines.

**History elision.** Tool results older than the last `KEEP_TOOL_RESULTS = 4`
are replaced with a placeholder in the *request* — the real history is kept
intact. Only the `content` is swapped, never the message, because each
`role: "tool"` entry must keep its `tool_call_id` to answer the call that
requested it. The placeholder tells the model it may re-read the file.

**Retries.** Three attempts. Rate-limited responses honour the `retry-after`
header (exponential backoff would retry long before a per-minute window rolls
over); `tool_use_failed` is retried as a generation artifact; other 4xx errors
raise immediately, since they will fail identically.

---

## Verified run

Against `callicoder/node-easy-notes-app` on `openai/gpt-oss-120b`:

| | |
| --- | --- |
| Turns | 20 of 40 |
| Tool calls | 18 |
| Wall clock | 558 s |
| Tokens | 82,482 prompt / 4,600 completion |
| Result | Exit 0, ended with `SUMMARY:` |

All five phases fired. The agent got the non-obvious part right: it registered
`/notes/count` **before** `/notes/:noteId`, so Express would not shadow the new
route with the parameterised one. Both edited files passed `node --check`.

**The diff was still poor.** Alongside the ~12 lines the feature needed, it
reformatted the whole controller — `if(` → `if (`, promise chains reindented —
for a total of 84 insertions and 63 deletions. Whole-file writes let a model
tidy everything it retypes, and these models take the invitation.

On the vaguer task (*"better organise and search their notes"*) with
`llama-3.3-70b-versatile`, it re-emitted its `PLAN:` six times without
progressing, then implemented a `findByTag` handler but exhausted the daily
token budget before registering its route — leaving unreachable dead code.

---

## Running the result locally

To see the agent's changes running rather than just reading the diff:

```bash
# 1. MongoDB — 4.4 matches the app's old Mongoose 5.2 driver
docker run -d --name easy-notes-mongo -p 27017:27017 mongo:4.4

# 2. The notes app (with the agent's edits) on http://localhost:3000
cd target-repo
npm install
node server.js
```

Then exercise it — the agent-written endpoint included:

```bash
curl -X POST http://localhost:3000/notes \
     -H "Content-Type: application/json" \
     -d '{"title":"Groceries","content":"milk, eggs"}'
curl http://localhost:3000/notes/count        # → {"count":1}
```

Teardown: stop the Node process and `docker rm -f easy-notes-mongo`.

---

## Frontend

`frontend/` is a browser UI for the notes app — list, create, edit, delete,
live search with match highlighting, and a note-count badge fed by the
agent-written `/notes/count` endpoint.

```bash
# with the backend from the previous section running:
python frontend/serve.py            # → http://localhost:8080
```

Two files, no build step, no dependencies:

| File | Role |
| --- | --- |
| `frontend/index.html` | The whole UI — vanilla JS, single file, light/dark via `prefers-color-scheme` |
| `frontend/serve.py` | Stdlib-only static server that also proxies `/notes*` to the Express app |

**Why the proxy exists:** the Express app ships no CORS headers, so a page
served from another origin cannot fetch it — and the app lives in
`target-repo/`, which this project deliberately does not modify by hand. The
proxy makes the browser talk to one origin (`:8080`) and relays `/notes*` to
`:3000` (`--api` to point elsewhere, `--port` to move it). Express error
responses (400/404/500) are relayed with status and JSON body intact, so the
UI can show the API's own error messages.

The UI treats the server as the source of truth: every mutation refreshes the
list and the count from the API rather than patching local state. Search is
client-side filtering over the fetched list — fine at this scale, and it keeps
the backend untouched.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `error: GROQ_API_KEY is not set` | No key in env or `.env` | `cp .env.example .env`, add the key |
| `413 … Request too large … tokens per minute` | `prompt + max_completion_tokens` exceeds TPM | Use a higher-TPM model, or lower `DEFAULT_MAX_TOKENS` in `agent/llm.py` |
| `400 … tool_use_failed` | Model's tool-call JSON truncated mid-write | Raise `DEFAULT_MAX_TOKENS`; retried automatically once |
| `429 … tokens per day (TPD)` | Daily free-tier budget exhausted for that model | Switch `--model`, or wait for the reset |
| Run ends at the turn cap | Task too large, or the model is looping | Narrow the task, or raise `--max-turns` |
| Agent re-plans repeatedly | Weaker model following the phase prompt loosely | Try `--model openai/gpt-oss-120b` |
| `npm test` always fails | The notes app defines `test` as `exit 1` | Expected; `node --check` is the meaningful signal |
| Mojibake instead of `─ · → ✓` | Terminal cannot encode the glyphs | Handled automatically — it falls back to ASCII |

---

## Extending the agent

### Adding a tool

Four steps, all in `agent/tools.py`:

1. **Write the function.** It takes plain keyword arguments and returns a
   `str`. Report failures as a returned string starting with `Error:` — never
   raise, and never return a non-string.

   ```python
   def git_diff(path: str = ".") -> str:
       """Show uncommitted changes under path."""
       try:
           target = _resolve(path)      # rejects anything outside the repo root
       except PathEscapeError as exc:
           return f"Error: {exc}"
       ...
       return output
   ```

2. **Declare it** in `TOOLS` via the `_function` helper. The description is
   what the model reads to decide *when* to reach for it, so say when it
   applies, not just what it does.

3. **Register it** in `_IMPLEMENTATIONS`, mapping the schema name to the
   function. `dispatch` routes on that dict and reports unknown names against
   it.

4. **Resolve every path through `_resolve`.** That is the only thing keeping a
   model-supplied path inside the repo root.

No change to `loop.py` is needed — it discovers tools through `TOOLS` and
`dispatch`.

### Changing the workflow

The phases live entirely in `SYSTEM_PROMPT` in `agent/prompts.py`; there is no
phase state machine in the loop. Editing that string changes how the agent
works. Two couplings to respect:

- `SUMMARY_MARKER` is the loop's stop condition. Change the marker in
  `prompts.py` and the loop follows automatically — but change the wording *in
  the prompt body only* and runs will never terminate cleanly.
- Keep run-specific detail out of `SYSTEM_PROMPT`. Repo path and task belong in
  `build_task_message`, which produces the first user message.

### Switching providers

`agent/llm.py` is the provider boundary. Moving off Groq means rewriting that
module plus `_assistant_message` and `_handle_tool_calls` in `loop.py` — the
two places that know the wire format. Tools, prompts, safety, and console
output are provider-agnostic.

---

## Development and testing

The loop can be exercised without spending a single token. `run()` accepts any
object with `.model`, `.temperature`, and a `.complete()` returning something
shaped like a `ChatCompletion`, so a scripted stub drives a whole run offline:

```python
import types
from agent import loop

def msg(content, tool_calls=None, finish="stop"):
    message = types.SimpleNamespace(
        content=content, tool_calls=tool_calls, reasoning=None
    )
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message, finish_reason=finish)],
        usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )

class FakeLLM:
    model, temperature = "fake", 0.0
    def __init__(self, script): self.script = list(script)
    def complete(self, messages, tools=None, system=None, **kw):
        return self.script.pop(0)

loop.run(repo_path, "add an endpoint", llm=FakeLLM([msg("SUMMARY: done")]))
```

Worth covering with scripted turns: a text-only `PLAN:` turn (it must *not* end
the run), the nudge cap, the turn cap raising `RuntimeError`, retry-then-succeed
versus retries-exhausted, and malformed tool-call JSON. Assert that every
`tool_call_id` is answered exactly once and that the assistant entry keeps its
`tool_calls`.

The tools are independently testable with no model at all — bind a temporary
directory and call `dispatch` directly:

```python
from agent.tools import set_repo_root, dispatch

set_repo_root("/tmp/scratch-repo")
print(dispatch("list_files", {}))
print(dispatch("read_file", {"path": "../../etc/passwd"}))   # Error: path escapes…
print(dispatch("run_shell", {"command": "sudo rm -rf /"}))   # Error: refusing…
```

There is no committed test suite yet — these are the patterns the code was
verified with, not something you can run with `pytest` today.

**Compatibility note.** The code targets Python 3.11+ (`X | None` annotations,
`Path.is_relative_to`) and has been exercised on 3.13.

---

## Design trade-offs

**Single-agent loop, not multi-agent.** One conversation, one context window,
tools executed in sequence. A planner/worker/reviewer split would parallelise
wide tasks and give an independent opinion on the diff, at the cost of
cross-agent coordination and roughly N× the tokens. For a six-file repo the
coordination overhead would exceed the benefit. The cost: no independent review
— the VERIFY phase is the same model marking its own homework.

**Anchored patches, not whole-file writes or unified diffs.** The agent
supplies a unique snippet and its replacement rather than a `@@` hunk. A hunk
carries line numbers that go stale the moment anything above shifts, and a
model that miscounts context lines produces a patch that will not apply;
anchored text has no line numbers to get wrong. The cost is that the anchor
must be unique — hence the ambiguity refusal — and that a model must copy
existing text exactly. See [Patch-based editing](#patch-based-editing).

**Deterministic pre-scan, not embeddings.** The Repository Intelligence Engine
is regex-and-heuristic Python: free, instant, reproducible, and wrong in
predictable ways. Vector indexing would add an embedding step, a dependency,
and a staleness problem to solve a retrieval issue a six-file repo does not
have — semantic search starts paying off somewhere in the thousands-of-files
range, where "which file handles billing?" stops being greppable.

**Read-before-write is instructed, not enforced.** The prompt requires it;
nothing in the harness blocks a `write_file` on an unread path.

**History is compressed to deterministic summaries, not model-written ones.**
An old file read collapses to a generated one-liner (exports, routes,
imports, DB calls) that is free, instant, and always current. It is not a
semantic summary — it will not tell the model *why* a function exists, only
what the file contains. An LLM-written summary would say more and cost an
extra call per file.

**Git is untouched.** No branch, no commit, no rollback. Review with `git diff`
and revert by hand.

---

## Limitations

- **Model quality dominates outcomes.** The phase structure is followed loosely,
  not reliably. Weaker models re-plan in circles, reformat files they were told
  to leave alone, and occasionally emit malformed tool-call JSON.
- **A complete plan is not a safe change.** All nine sections can be present,
  including `RISKS` and `IMPACT` asserting nothing breaks, while the edit
  silently breaks the app — observed and documented in
  [The planning stage](#the-planning-stage). `node --check` proves a file
  parses, nothing more. Review the diff and run the app.
- **Free-tier token limits are the practical ceiling.** 8,000 TPM on most Groq
  models, 12,000 on `llama-3.3-70b-versatile`, plus a daily cap. A repo much
  larger than the notes app needs a paid tier.
- **No independent verification.** The model checks its own work.
- **No checkpointing.** A run that fails at turn 30 starts over at turn 1.
- **Tool calls execute sequentially**, even when the model requests several at
  once.
- **The 40-turn cap is a blunt stop.** Hitting it leaves the repository in
  whatever half-edited state the agent reached.
- **The symbol index is regex-based, not an AST.** Unusual code layouts
  (multi-line signatures, dynamic route registration, computed exports) are
  missed; JS/TS and Python are indexed, other languages get detection only.
  The scan caps at 2,000 files.
- **Call edges are attributed by body span**, meaning a function is assumed to
  run from its definition line to the next definition in the same file. That
  fits flat controller-style modules; deeply nested closures will misattribute.
  Route wiring only resolves `app.get('/x', mod.handler)` shapes — routers
  built dynamically are missed.
- **Memory is per-run, not persistent.** Nothing is cached between
  invocations; every run re-scans from scratch.
- **Semantic search is vocabulary-bound, not embedding-based.** It knows the
  words in its action, artifact, and concept tables plus whatever the code
  names itself. A question phrased entirely outside that vocabulary
  ("where does the system fan out work?") will find nothing, where an
  embedding model might. Extending it means adding words to
  `ACTIONS`/`CONCEPTS` in `agent/search.py`, not retraining anything.
- **Patching shifts the failure mode rather than removing it.** A model can
  still choose a wrong-but-unique anchor and patch the wrong function, and it
  must copy existing text exactly — a near-miss anchor costs a turn. The
  engine is fully covered by tests and benchmarked through the real loop and
  tools, but a *model-driven* patch has not yet completed a live run: the
  daily token budget ran out at the implementation turn, twice.
- **`search_repo` is verified offline but not yet inside a completed live
  run.** Its engine, ranking, and tool dispatch are covered by tests, and a
  live model did choose it with a sensible question — but that run failed on
  Groq's tool-call serialisation (`<function=search_repo {...}</function>`
  instead of JSON) and the daily token budget ran out before a clean
  end-to-end demonstration. The temperature-varied resample added in
  response is likewise untested live.
- **Tested on one repository.** Behaviour on larger or differently structured
  Express projects is unmeasured.

---

## Project layout

```
converseai/
├── main.py              CLI entry point
├── agent/
│   ├── __init__.py
│   ├── graph.py         Knowledge graph: imports, routes, calls, DB edges
│   ├── intel.py         Repository Intelligence Engine (pre-scan, no LLM)
│   ├── memory.py        Repository memory + module summaries (compression)
│   ├── patch.py         Patch engine: locate, apply, verify, diff
│   ├── search.py        Semantic search: intent parsing, ranking, evidence
│   ├── llm.py           Groq client wrapper — one request, one response
│   ├── loop.py          Turn loop, retries, context elision, console output
│   ├── prompts.py       System prompt (five phases) and the opening message
│   └── tools.py         Tool schemas, dispatch, path and command safety
├── frontend/
│   ├── index.html       Browser UI for the notes app (single file, no build)
│   └── serve.py         Static server + /notes* proxy (stdlib only)
├── requirements.txt     groq, python-dotenv
├── .env.example         Template for .env (gitignored)
└── target-repo/         Whatever you clone to work on (gitignored)
```
