"""Shared LLM client with provider dispatch (Claude or Gemini).

Every LLM call in Darwin goes through this module so we have one place
to tune concurrency, retries, and rate-limit handling. The provider is
selected by ``settings.llm_provider``:

    LLM_PROVIDER=claude   (default — uses ANTHROPIC_API_KEY)
    LLM_PROVIDER=gemini   (uses GOOGLE_API_KEY)

Callers (strategist, builder, baseline engine) do NOT branch on the
provider. `complete()` returns a list of content blocks with the same
shape regardless of backend:

    block.type in {"text", "tool_use"}
    block.text                           # when type == "text"
    block.name, block.input (dict)       # when type == "tool_use"

For Gemini we wrap response parts in ``SimpleNamespace`` so agent code
that iterates Anthropic ``ContentBlock`` objects keeps working without
change.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from typing import Any

from darwin.config import settings

log = logging.getLogger("darwin.llm")

_sem = asyncio.Semaphore(30)

# Lazy provider clients — only the selected provider is instantiated,
# so users without one of the keys set don't see a startup error.
_anthropic_client = None
_gemini_client = None


def _get_anthropic():
    """Lazy-init the Anthropic async client."""
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import AsyncAnthropic

        _anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


def _get_gemini():
    """Lazy-init the Google GenAI client.

    The SDK exposes both sync and async methods on one client; we use the
    async surface via ``client.aio.models.generate_content``.
    """
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        _gemini_client = genai.Client(api_key=settings.google_api_key)
    return _gemini_client


# ──────────────────────────────────────────────────────────────────────
# Gemini → Anthropic adapter helpers
# ──────────────────────────────────────────────────────────────────────


def _anthropic_tools_to_gemini(tools: list[dict]):
    """Translate Anthropic-style tool specs into Gemini function declarations.

    Anthropic tool shape:  ``{name, description, input_schema}``
    Gemini tool shape:     ``Tool(function_declarations=[FunctionDeclaration(...)])``

    Darwin's tool schemas are JSON Schema, which Gemini's ``parameters``
    field accepts directly — no structural translation needed.
    """
    from google.genai import types

    decls = [
        types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=t["input_schema"],
        )
        for t in tools
    ]
    return [types.Tool(function_declarations=decls)]


def _gemini_response_to_blocks(response) -> list:
    """Normalize a Gemini response into Anthropic-style content blocks.

    Each block is a ``SimpleNamespace`` quacking like an Anthropic
    ``ContentBlock``: attributes ``type``, ``text`` (for text blocks), or
    ``name`` + ``input`` (for tool_use blocks).
    """
    blocks: list[SimpleNamespace] = []
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return blocks
    parts = getattr(candidates[0].content, "parts", None) or []
    for part in parts:
        fc = getattr(part, "function_call", None)
        if fc is not None:
            args = dict(fc.args) if fc.args else {}
            blocks.append(SimpleNamespace(type="tool_use", name=fc.name, input=args))
            continue
        text = getattr(part, "text", None)
        if text:
            blocks.append(SimpleNamespace(type="text", text=text))
    return blocks


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


async def complete(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 256,
    tools: list[dict] | None = None,
    provider: str | None = None,
) -> Any:
    """One-shot chat call routed through a provider.

    ``provider`` selects the SDK ("claude" or "gemini"). When ``None``,
    falls back to ``settings.llm_provider`` — this preserves the
    pre-multi-provider behavior for callers that don't care which
    backend handles their call. Roles that want explicit control
    (strategist, builder, player) should pass ``provider`` directly so
    a single generation can fan out to multiple providers in parallel.

    Returns a list of content blocks. For text replies, read
    ``content[0].text``. For tool-use replies, look for a block where
    ``block.type == "tool_use"`` and read ``block.name`` / ``block.input``.
    The same return shape is produced regardless of provider.
    """
    resolved = provider or settings.llm_provider
    tool_names = [t["name"] for t in tools] if tools else []
    log.info(
        "complete provider=%s model=%s prompt_chars=%d max_tokens=%d tools=%s",
        resolved, model, len(user), max_tokens, tool_names,
    )
    t0 = time.monotonic()
    try:
        if resolved == "gemini":
            blocks = await _complete_gemini(model, system, user, max_tokens, tools)
        elif resolved == "claude":
            blocks = await _complete_claude(model, system, user, max_tokens, tools)
        else:
            raise ValueError(f"unknown provider: {resolved!r}")
    except Exception:
        log.exception(
            "complete failed after %.1fs provider=%s model=%s",
            time.monotonic() - t0, resolved, model,
        )
        raise

    summary = _summarize_blocks(blocks)
    log.info(
        "complete ok in %.1fs provider=%s model=%s blocks=%s",
        time.monotonic() - t0, resolved, model, summary,
    )
    return blocks


def _summarize_blocks(blocks: Any) -> list[str]:
    out: list[str] = []
    for b in blocks or []:
        t = getattr(b, "type", "?")
        if t == "text":
            text = getattr(b, "text", "") or ""
            out.append(f"text({len(text)}ch)")
        elif t == "tool_use":
            out.append(f"tool_use(name={getattr(b, 'name', '?')})")
        else:
            out.append(t)
    return out


async def complete_text(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 256,
    provider: str | None = None,
) -> str:
    """Convenience wrapper for plain-text replies.

    Returns the first text block's content, or ``""`` if no text block
    came back. ``provider`` is forwarded to ``complete()``; ``None``
    means fall back to the global default.
    """
    content = await complete(
        model, system, user, max_tokens=max_tokens, provider=provider
    )
    for block in content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


# ──────────────────────────────────────────────────────────────────────
# Provider implementations
# ──────────────────────────────────────────────────────────────────────


async def _complete_claude(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    tools: list[dict] | None,
) -> Any:
    from anthropic._exceptions import APIError, RateLimitError

    client = _get_anthropic()
    backoff = 1.0
    async with _sem:
        for attempt in range(5):
            try:
                msg = await client.messages.create(
                    model=model,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    tools=tools or [],
                )
                return msg.content
            except RateLimitError:
                await asyncio.sleep(backoff)
                backoff *= 2
            except APIError:
                if attempt == 4:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2
    raise RuntimeError("unreachable")


async def _complete_gemini(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    tools: list[dict] | None,
) -> Any:
    from google.genai import errors as genai_errors
    from google.genai import types

    client = _get_gemini()
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        # Gemini 2.5 Flash/Pro enable thinking by default, which consumes
        # output-token budget BEFORE any function_call is emitted. For a
        # builder that needs to return ~1-2k tokens of Python code, thinking
        # can eat the entire budget and we get an empty response. Disable it.
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    if tools:
        config.tools = _anthropic_tools_to_gemini(tools)
        # Force the model to emit a function_call rather than free text.
        config.tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        )

    backoff = 1.0
    last_error: Exception | None = None
    async with _sem:
        for attempt in range(5):
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=user,
                    config=config,
                )
                blocks = _gemini_response_to_blocks(response)
                if not blocks:
                    # Help diagnose "did not return tool_use" vs. truncation,
                    # safety blocks, or other silent empty-response states.
                    cand = (response.candidates[0]
                            if getattr(response, "candidates", None) else None)
                    fr = getattr(cand, "finish_reason", None)
                    safety = getattr(cand, "safety_ratings", None)
                    usage = getattr(response, "usage_metadata", None)
                    log.warning(
                        "gemini empty response model=%s finish_reason=%r "
                        "safety=%r usage=%r",
                        model, fr, safety, usage,
                    )
                return blocks
            except genai_errors.APIError as e:
                last_error = e
                status = getattr(e, "code", None)
                # Log every retry so the operator sees *which* failure mode
                # exhausted retries. Without this, a string of 429s and a
                # string of 503s look identical from outside.
                log.warning(
                    "gemini retry attempt=%d/5 status=%r model=%s err=%s",
                    attempt + 1, status, model, str(e)[:200],
                )
                if status == 429 or (attempt < 4):
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise
    # All 5 attempts errored. Surface the actual API failure rather than a
    # vestigial RuntimeError("unreachable") — the previous wording made it
    # impossible to tell rate-limit (429) from upstream-overload (503).
    raise RuntimeError(
        f"gemini call failed after 5 retries (model={model}, "
        f"last_status={getattr(last_error, 'code', None)!r}): "
        f"{type(last_error).__name__}: {str(last_error)[:200]}"
    ) from last_error
