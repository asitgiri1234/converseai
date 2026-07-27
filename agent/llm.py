"""Thin wrapper around the Groq chat completions API with tool-use support.

This module is deliberately small: it owns client construction, model defaults,
and the single request/response call. It does *not* own the agent loop or tool
execution -- those live in `agent.loop` and `agent.tools`.

Groq speaks the OpenAI chat-completions dialect, so tools go out as a list of
``{"type": "function", "function": {...}}`` objects and come back on
``message.tool_calls`` with their arguments as a JSON *string*.

Usage:

    from agent.llm import LLM

    llm = LLM()
    response = llm.complete(
        messages=[{"role": "user", "content": "List the files in this repo."}],
        tools=TOOLS,
        system="You are a coding agent.",
    )
    message = response.choices[0].message
    if message.tool_calls:
        ...  # run each call, append role="tool" results, call complete() again
"""

from __future__ import annotations

import os
from typing import Any, Iterable

import groq
from dotenv import load_dotenv

load_dotenv()

# Handles the tool loop correctly and carries the most generous free-tier
# token budget (12k TPM vs 8k for the gpt-oss and qwen models), which matters
# because an agent replays its whole history every turn. Override with
# --model or the GROQ_MODEL env var.
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Squeezed from both sides. Groq bills this against the per-minute token
# budget *up front*, on top of the prompt, so `prompt + max_completion_tokens`
# must fit under the TPM limit or the request 413s before the model ever runs.
# But write_file sends a whole file as a tool argument, and a budget that
# cannot fit the largest file truncates the arguments mid-JSON, which Groq
# rejects with `tool_use_failed`. This value clears a ~120-line file while
# leaving room for history on a 12k TPM tier.
DEFAULT_MAX_TOKENS = 1_800
DEFAULT_TEMPERATURE = 0.2


class MissingAPIKeyError(RuntimeError):
    """Raised when GROQ_API_KEY is not set in the environment."""


class LLM:
    """A single-call wrapper around ``client.chat.completions.create``.

    Args:
        model: Model id to use. Defaults to the ``GROQ_MODEL`` env var,
            falling back to ``openai/gpt-oss-120b``.
        max_tokens: Ceiling on tokens generated per response.
        temperature: Sampling temperature. Low by default -- this is a coding
            agent, not a brainstorming one.
        api_key: Overrides the ``GROQ_API_KEY`` env var. Prefer the env var;
            this exists for tests and for callers that manage their own
            secrets.
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise MissingAPIKeyError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and "
                "add your key, or export it in your shell."
            )

        self.model = model or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = groq.Groq(api_key=key)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Any:
        """Send one turn to the model and return the raw completion.

        The response is returned unmodified so the caller can inspect
        ``choices[0].message.tool_calls`` and ``finish_reason`` itself.
        Nothing here executes tools or mutates ``messages``.

        Args:
            messages: Conversation history, oldest first, *without* the system
                message -- it is prepended here so it stays out of the mutable
                history and can't be truncated away.
            tools: Tool definitions in OpenAI function format. Omit or pass an
                empty list to disable tool use.
            system: System prompt. See `agent.prompts`.
            max_tokens: Per-call override of the instance default.
            temperature: Per-call override of the instance default.

        Returns:
            The ``ChatCompletion`` object.

        Raises:
            groq.APIStatusError: For non-2xx responses (rate limits, invalid
                requests, server errors).
        """
        payload: list[dict[str, Any]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(messages)

        request: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "max_completion_tokens": max_tokens or self.max_tokens,
            "temperature": (
                self.temperature if temperature is None else temperature
            ),
        }
        if tools:
            request["tools"] = list(tools)

        return self.client.chat.completions.create(**request)
