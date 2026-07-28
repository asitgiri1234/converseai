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
- [How repository exploration works](#how-repository-exploration-works)
- [Tool reference](#tool-reference)
- [Message formats](#message-formats)
- [Safety model](#safety-model)
- [Console output](#console-output)
- [Context and rate-limit management](#context-and-rate-limit-management)
- [Verified run](#verified-run)
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
     │  history: [user task, assistant, tool results, …]   │
     │                                                     │
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
| **1. EXPLORE** | `list_files`, then `package.json` (entry point, scripts, deps, CommonJS vs ESM), then routes, models, controllers. Following `require()` chains rather than guessing at paths. |
| **2. PLAN** | Emits a numbered `PLAN:` block — the feature chosen, every file it will touch with a reason, and any `Assumption:` — *before* the first edit. |
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

## How repository exploration works

Exploration is **tool-driven and LLM-decided**. There is no pre-indexing pass,
no repo map built ahead of time, no embeddings, and no hardcoded traversal
order. The agent gets five tools and a repo root, and picks its next move from
what the last result told it.

A real trace against the notes app:

```
list_files(".")                      → sees app/, config/, server.js
read_file("package.json")            → main: server.js, deps: express, mongoose
read_file("app/models/note.model.js")→ schema: title, content, timestamps
read_file("app/controllers/…")       → five exports, promise chains, 4-space indent
read_file("app/routes/note.routes.js")→ how routes are registered
```

Five calls, and it knows the module system, the conventions, and the request
path. The trade-off is turn count: each step is a round trip, so exploring costs
latency and tokens that a pre-built index would not. On a repo this size that is
the right trade (see [Design trade-offs](#design-trade-offs)).

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
| `write_file` | `path`, `content` | Create or overwrite, creating parent directories | Reports created-vs-overwrote plus line and byte counts |
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

**Whole-file writes, not diff edits.** `write_file` replaces the entire file, so
the agent must reproduce everything it is not changing. Simple, unambiguous, and
it cannot produce a malformed hunk. But it costs output tokens proportional to
file size, risks truncation on large files, and invites gratuitous reformatting
— all three were observed. A `str_replace`-style edit tool is the right call
above a few hundred lines.

**No vector indexing or embeddings.** The target is six source files;
`list_files` plus one regex `search_code` finds anything in a call or two.
Embedding a repo this size adds an indexing step, a dependency, and a staleness
problem to solve a retrieval issue that does not exist. It starts paying off
somewhere in the thousands-of-files range.

**Read-before-write is instructed, not enforced.** The prompt requires it;
nothing in the harness blocks a `write_file` on an unread path.

**History is elided, not summarised.** Cheap and lossless-by-reference — the
agent can re-read — but it does cause repeat reads. Real compaction would
summarise instead.

**Git is untouched.** No branch, no commit, no rollback. Review with `git diff`
and revert by hand.

---

## Limitations

- **Model quality dominates outcomes.** The phase structure is followed loosely,
  not reliably. Weaker models re-plan in circles, reformat files they were told
  to leave alone, and occasionally emit malformed tool-call JSON.
- **Free-tier token limits are the practical ceiling.** 8,000 TPM on most Groq
  models, 12,000 on `llama-3.3-70b-versatile`, plus a daily cap. A repo much
  larger than the notes app needs a paid tier.
- **No independent verification.** The model checks its own work.
- **No checkpointing.** A run that fails at turn 30 starts over at turn 1.
- **Tool calls execute sequentially**, even when the model requests several at
  once.
- **The 40-turn cap is a blunt stop.** Hitting it leaves the repository in
  whatever half-edited state the agent reached.
- **Tested on one repository.** Behaviour on larger or differently structured
  Express projects is unmeasured.

---

## Project layout

```
converseai/
├── main.py              CLI entry point
├── agent/
│   ├── __init__.py
│   ├── llm.py           Groq client wrapper — one request, one response
│   ├── loop.py          Turn loop, retries, context elision, console output
│   ├── prompts.py       System prompt (five phases) and the opening message
│   └── tools.py         Tool schemas, dispatch, path and command safety
├── requirements.txt     groq, python-dotenv
├── .env.example         Template for .env (gitignored)
└── target-repo/         Whatever you clone to work on (gitignored)
```
