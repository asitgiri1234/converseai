"""CLI entry point for the coding agent.

    python main.py --repo /path/to/project --task "Add a health check endpoint"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.llm import LLM, MissingAPIKeyError
from agent.loop import run


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
        help="Model id to use. Defaults to $ANTHROPIC_MODEL, then claude-opus-5.",
    )
    parser.add_argument(
        "--effort",
        default="high",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="How hard the model works per turn. Default: high.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Stop after this many model turns. Default: 50.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the agent and return a process exit code."""
    args = parse_args(argv)

    repo = args.repo.expanduser().resolve()
    if not repo.is_dir():
        print(f"error: --repo is not a directory: {repo}", file=sys.stderr)
        return 2

    try:
        llm = LLM(model=args.model, effort=args.effort)
    except MissingAPIKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        result = run(repo, args.task, llm=llm, max_turns=args.max_turns)
    except NotImplementedError:
        print(
            "error: the agent loop is not implemented yet "
            "(see agent/loop.py and agent/tools.py).",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
