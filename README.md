# converseai

An AI coding agent for existing Node.js / Express / MongoDB codebases. Give it
a vague product request; it explores the repo, plans, edits, and verifies.

Python 3.11+. Dependencies: `anthropic`, `python-dotenv`.

```bash
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY
python main.py --repo ./target-repo --task "Improve the application so users can better organise and search their notes."
```

## Architecture

Five modules, one direction of control. `loop` is the only stateful piece: the
Messages API is stateless, so it owns the conversation history and replays it
on every request.

```
  main.py ──── parses --repo/--task, binds the repo root, exits cleanly
     │
     ▼
  agent/loop.py ───────────────────────────────────────────┐
     │  history: [user task, assistant, tool_results, …]   │
     │                                                     │
     ├──▶ agent/prompts.py    system prompt (phases, rules)│
     │                                                     │
     ├──▶ agent/llm.py ──────▶ Anthropic Messages API      │
     │      one request         (streamed, one Message back)│
     │    ◀── content: [thinking, text, tool_use …]        │
     │                                                     │
     └──▶ agent/tools.py ──▶ target repo (filesystem, shell)
            dispatch(name, input) → str ────────────────────┘
                                    tool_result blocks re-enter history
```

- **`main.py`** — CLI surface. Validates `--repo`, binds it as the tool root,
  maps exceptions to exit codes (2 = bad input, 1 = run failed, 130 = Ctrl-C).
- **`agent/llm.py`** — thin Messages API wrapper. One call, one `Message`
  returned unmodified. Streams internally so long turns don't hit the HTTP
  timeout. Defaults to `claude-opus-5` with adaptive thinking.
- **`agent/prompts.py`** — the stable system prompt. Kept free of
  interpolation so it stays a cacheable prefix; run-specific context (repo
  path, task) goes in the first user message.
- **`agent/tools.py`** — five tool schemas plus `dispatch(name, input)`, which
  always returns a string and never raises.
- **`agent/loop.py`** — the turn loop, retry policy, and console output.

## Agent workflow phases

The system prompt drives five phases; the console narrates each turn.

1. **EXPLORE** — `list_files`, then `package.json` (entry point, scripts,
   deps, CommonJS vs ESM), then routes, models, controllers. No path guessing.
2. **PLAN** — a numbered `PLAN:` block naming the feature chosen, every file
   to be touched with a reason, and any `Assumption:` — emitted *before* the
   first edit. Ambiguity is resolved by picking the smallest reasonable
   reading and stating it, never by asking (nothing is watching to answer).
3. **IMPLEMENT** — `write_file`, preserving existing routes, exports, response
   shapes, and field names, matching the surrounding style. Additive and
   backwards-compatible; no rewriting in another language or framework.
4. **VERIFY** — re-read every modified file, `node --check` each changed file,
   run `npm test` / `npm run lint` where they exist.
5. **SUMMARIZE** — a `SUMMARY:` section listing every file changed and why.

`PLAN:` is text with no tool call, which a naive loop reads as "finished". So
a text-only turn ends the run **only** when it contains `SUMMARY:`; otherwise
the agent is nudged onward, capped at 3 nudges. The marker is defined once in
`prompts.py` and imported by `loop.py`.

## How repository exploration works

Exploration is **tool-driven and LLM-decided**. There is no pre-indexing pass,
no repo map built ahead of time, and no hardcoded traversal order. The agent
receives five tools and a repo root, and chooses what to look at next based on
what the last result told it — read `package.json`, discover
`require('./app/routes/note.routes.js')`, read that, follow it to the
controller, and so on.

| Tool | Behaviour |
| --- | --- |
| `list_files(path)` | Indented tree; skips `node_modules` and `.git`; caps at 1000 entries |
| `read_file(path)` | Line-numbered; truncates past 500 lines; refuses binaries |
| `search_code(pattern)` | Regex → `file:line:match`; caps at 50 results |
| `write_file(path, content)` | Create or overwrite, making parent dirs |
| `run_shell(command)` | `cwd=repo`, 30 s timeout, returns exit code + stdout + stderr |

Every path is resolved (symlinks included) and rejected unless it lands inside
the repo root, so `../`, absolute paths, and outward symlinks all fail.
`run_shell` refuses privilege escalation (`sudo`, `su`, `doas`, `runas`),
recursive deletes aimed outside the repo, and `mkfs` / `dd of=/dev/…` /
`shutdown` / fork bombs. **This is a guardrail, not a sandbox** — anything that
passes runs with your full privileges. Use a throwaway checkout.

## Assumptions and trade-offs

- **Single-agent loop, not multi-agent.** One conversation, one context
  window, tools executed in sequence. A planner/worker/reviewer split would
  parallelise wide tasks and give a second opinion on the diff, at the cost of
  cross-agent coordination and roughly N× the tokens. For a five-file repo the
  coordination overhead would exceed the benefit. This does mean no
  independent review of the agent's own output — the VERIFY phase is the same
  model marking its own homework.
- **Whole-file writes, not diff edits.** `write_file` replaces the entire
  file, so the agent must reproduce everything it isn't changing. Simple and
  unambiguous, and it can't produce a malformed hunk — but it costs output
  tokens proportional to file size and risks silent truncation on large files.
  A `str_replace`-style edit tool would be the right call above a few hundred
  lines.
- **No vector indexing or embeddings.** `node-easy-notes-app` is six source
  files; `list_files` plus a regex `search_code` finds anything in one or two
  calls. Embedding a repo this size would add an indexing step, a dependency,
  and a staleness problem to solve a retrieval issue that doesn't exist here.
  It would become worth it somewhere in the thousands-of-files range.
- **Read-before-write is instructed, not enforced.** The prompt requires it;
  nothing in the harness blocks a `write_file` on an unread path.
- **Git is untouched.** No branch, no commit, no rollback. Review with
  `git diff` in the target repo and revert by hand.

## Limitations

- **Not yet run against a live model.** The loop, tools, and prompt were
  verified with scripted responses and by running the tools against the cloned
  `target-repo` — but no end-to-end run has happened, because this environment
  has no `ANTHROPIC_API_KEY`. Real-model adherence to the phase structure is
  unproven.
- `node-easy-notes-app` defines `"test": "echo \"Error: no test specified\" &&
  exit 1"`. VERIFY's `npm test` step will *always* fail there, and the agent
  may waste turns trying to fix it. `node --check` is the meaningful signal.
- Tool calls execute sequentially even when the model requests several at once.
- No checkpointing: a run that fails at turn 30 starts over from turn 1.
- History grows unbounded; a long run can approach the context window with no
  compaction.
- The 40-turn cap is a blunt stop. Hitting it leaves the repo in whatever
  half-edited state the agent reached.
