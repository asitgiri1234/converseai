# converseai

An AI coding agent built on the Anthropic Messages API. Point it at a repo,
give it a task, and it works through the task using tools you control.

Python 3.11+. Dependencies: `anthropic`, `python-dotenv`.

## Status

Working end to end. Point it at a repo, give it a task, and it explores,
edits, and verifies until it's done.

```
────────────────────────────────────────────────────────────────────────
  converseai  →  claude-opus-5  (effort=high)
  repo: /home/me/demo
  task: Add a health check endpoint
────────────────────────────────────────────────────────────────────────

[turn 1/40]
  I should look at the repo layout before editing anything.
  · list_files(path=".")
    ✓ ./  [12 lines]
  · read_file(path="src/app.py")
    ✓ 1 from flask import Flask  [34 lines]

[turn 2/40]
  · write_file(path="src/health.py", content="def health(): return {'stat…")
    ✓ Created src/health.py (4 lines, 61 bytes).
  · run_shell(command="pytest -q")
    ✓ exit code: 0  [3 lines]  (2.4s)

────────────────────────────────────────────────────────────────────────
  ✓ done
────────────────────────────────────────────────────────────────────────
Added src/health.py with a health() endpoint. Tests pass.
────────────────────────────────────────────────────────────────────────
  2 turns  ·  4 tool calls  ·  31s  ·  18,204 in / 1,960 out tokens
────────────────────────────────────────────────────────────────────────
```

The loop stops when the model replies with text instead of tool calls, and
hard-stops at 40 turns (`--max-turns`). Transient API failures get one retry
with exponential backoff; 4xx errors fail immediately. Box-drawing glyphs fall
back to ASCII when the terminal can't encode them.

## Tools

`agent/tools.py` exports `TOOLS` (the schemas sent to the model) and
`dispatch(name, input)` (the client-side executor, which always returns a
string and never raises).

| Tool | Behaviour |
| --- | --- |
| `list_files(path)` | Indented recursive tree; skips `node_modules` and `.git`; caps at 1000 entries |
| `read_file(path)` | Contents with line numbers; truncates past 500 lines; refuses binaries |
| `search_code(pattern)` | Regex grep returning `file:line:match`; caps at 50 results |
| `write_file(path, content)` | Create or overwrite, making parent dirs |
| `run_shell(command)` | Runs with `cwd=repo`, 30s timeout, returns exit code + stdout + stderr |

Call `set_repo_root(path)` once before any tool runs — `main.py` does this
after validating `--repo`.

### Safety model

Every path is resolved (symlinks included) and rejected unless it lands inside
the repo root, so `../`, absolute paths, and symlinks pointing out all fail.
`run_shell` additionally refuses privilege escalation (`sudo`, `su`, `doas`,
`runas`), recursive deletes aimed outside the repo, and a short list of
machine-destroying commands (`mkfs`, `dd of=/dev/…`, `shutdown`, `reboot`,
fork bombs).

This is a guardrail, not a sandbox: a command that passes the check runs with
your full privileges. Run the agent against a throwaway checkout, or in a
container, if that matters.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env          # then add your ANTHROPIC_API_KEY
```

The key is read from the `ANTHROPIC_API_KEY` environment variable;
`python-dotenv` loads `.env` automatically at import time.

## Usage

```bash
python main.py --repo /path/to/project --task "Add a health check endpoint"
```

Options: `--model` (default `claude-opus-5`, overridable via `$ANTHROPIC_MODEL`),
`--effort` (`low`…`max`, default `high`), `--max-turns` (default 50).

## Layout

| File | Role |
| --- | --- |
| `main.py` | CLI: argument parsing, validation, exit codes |
| `agent/llm.py` | Thin Messages API wrapper — one request, one `Message` back |
| `agent/loop.py` | The agent loop, retry policy, and console output |
| `agent/tools.py` | Tool schemas and client-side dispatch |
| `agent/prompts.py` | System prompt and prompt assembly |

## Implementation notes

- **Append the whole `response.content`** to the message history, not just the
  text. Dropping `tool_use` blocks breaks the next request.
- **Return every tool result in a single user message.** Splitting them across
  messages trains the model out of making parallel tool calls.
- **Report tool failures as `tool_result` blocks with `is_error: true`**, not as
  exceptions — the model can then correct itself.
- **Validate model-supplied paths** against the repo root before touching the
  filesystem. `..`, symlinks, and absolute paths all need rejecting.
- **Keep the system prompt stable.** It sits at the front of the cached prefix,
  so interpolating a timestamp or the task into it defeats prompt caching. Put
  run-specific context in the first user message instead.
