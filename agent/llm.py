"""Thin wrapper around the Anthropic Messages API with tool-use support.

This module is deliberately small: it owns client construction, model defaults,
and the single request/response call. It does *not* own the agent loop or tool
execution -- those live in `agent.loop` and `agent.tools`.

Usage:

    from agent.llm import LLM

    llm = LLM()
    response = llm.complete(
        messages=[{"role": "user", "content": "List the files in this repo."}],
        tools=[{"name": "list_files", "description": "...", "input_schema": {...}}],
        system="You are a coding agent.",
    )
    if response.stop_reason == "tool_use":
        ...  # dispatch the tool_use blocks, then call complete() again
"""

from __future__ import annotations

import os
from typing import Any, Iterable

import anthropic
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16_000
DEFAULT_EFFORT = "high"


class MissingAPIKeyError(RuntimeError):
    """Raised when ANTHROPIC_API_KEY is not set in the environment."""


class LLM:
    """A single-call wrapper around `client.messages.create`.

    Args:
        model: Model id to use. Defaults to the ``ANTHROPIC_MODEL`` env var,
            falling back to ``claude-opus-5``.
        max_tokens: Ceiling on tokens generated per response (thinking tokens
            count against this too).
        effort: One of ``low``, ``medium``, ``high``, ``xhigh``, ``max``.
            Controls how much the model thinks and how thoroughly it works.
        api_key: Overrides the ``ANTHROPIC_API_KEY`` env var. Prefer the env
            var; this exists for tests and for callers that manage their own
            secrets.
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = DEFAULT_EFFORT,
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingAPIKeyError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
                "add your key, or export it in your shell."
            )

        self.model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.effort = effort
        self.client = anthropic.Anthropic(api_key=key)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> anthropic.types.Message:
        """Send one turn to the model and return the raw ``Message``.

        The response is returned unmodified so the caller can inspect
        ``stop_reason`` (``end_turn`` vs ``tool_use``) and walk ``content``
        blocks itself. Nothing here executes tools or mutates ``messages``.

        Args:
            messages: Full conversation history, oldest first. The API is
                stateless, so this must include every prior turn.
            tools: Tool definitions the model may call this turn. Omit or pass
                an empty list to disable tool use.
            system: System prompt. See `agent.prompts`.
            max_tokens: Per-call override of the instance default.
            effort: Per-call override of the instance default.

        Returns:
            The ``Message`` object, including any ``tool_use`` content blocks.

        Raises:
            anthropic.APIStatusError: For non-2xx responses (rate limits,
                invalid requests, server errors). The SDK already retries
                429s and 5xxs with backoff.
        """
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": messages,
            "output_config": {"effort": effort or self.effort},
            "thinking": {"type": "adaptive", "display": "summarized"},
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = list(tools)

        # Stream and collect: keeps long tool-heavy turns from tripping the
        # SDK's HTTP timeout, while still handing back one complete Message.
        with self.client.messages.stream(**request) as stream:
            return stream.get_final_message()
