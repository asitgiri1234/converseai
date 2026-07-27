"""CLI entry point for the coding agent.

    python main.py --repo /path/to/project --task "Add a health check endpoint"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import groq

from agent.llm import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    LLM,
    MissingAPIKeyError,
)
from agent.loop import MAX_TURNS, run
from agent.tools import set_repo_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="converseai",
        description="An AI coding agent that works on a repository you point it at.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        type=Path,
        help="Path to the repository the agent may read and modify.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="What the agent should do, in plain language.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Groq model id. Defaults to $GROQ_MODEL, then {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature. Default: {DEFAULT_TEMPERATURE}.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=MAX_TURNS,
        help=f"Stop after this many model turns. Default: {MAX_TURNS}.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the agent and return a process exit code."""
    args = parse_args(argv)

    repo = args.repo.expanduser().resolve()
    if not repo.is_dir():
        print(f"error: --repo is not a directory: {repo}", file=sys.stderr)
        return 2

    # Confines every tool to this directory for the rest of the process.
    set_repo_root(repo)

    try:
        llm = LLM(model=args.model, temperature=args.temperature)
    except MissingAPIKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        run(repo, args.task, llm=llm, max_turns=args.max_turns)
    except RuntimeError as exc:
        # The loop already printed the detail; this is just the exit code.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except groq.APIError as exc:
        print(f"error: Groq API request failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
