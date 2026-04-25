"""Shared LLM client with provider dispatch (Claude, Gemini, OpenAI).

Every LLM call in Darwin goes through this module so we have one place
to tune concurrency, retries, and rate-limit handling. The provider is
selected by ``settings.llm_provider``:

    LLM_PROVIDER=claude   (default — uses ANTHROPIC_API_KEY)
    LLM_PROVIDER=gemini   (uses GOOGLE_API_KEY)
    LLM_PROVIDER=openai   (uses OPENAI_API_KEY)

Callers (strategist, builder, baseline engine) do NOT branch on the
provider. `complete()` returns a list of content blocks with the same
shape regardless of backend:

    block.type in {"text", "tool_use"}
    block.text                           # when type == "text"
    block.name, block.input (dict)       # when type == "tool_use"

For Gemini and OpenAI we wrap response parts in ``SimpleNamespace`` so
agent code that iterates Anthropic ``ContentBlock`` objects keeps
working without change.

Tool emission is forced on every backend when ``tools`` is provided —
Anthropic via ``tool_choice={"type": "any"}``, Gemini via
``mode="ANY"``, OpenAI via ``tool_choice="required"``. Builder code
relies on a tool_use block coming back; without forcing, free-text
replies are an avoidable failure mode.

Prompt caching: pass ``cache_prefix`` to mark a stable user-content
prefix as cacheable. Anthropic honours it via ``cache_control:
ephemeral``; Gemini and OpenAI silently concatenate (no equivalent
public-cache primitive exposed by the SDK we want to commit to here).
"""

from __future__ import annotations

import asyncio
import json
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
_openai_client = None


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


def _get_openai():
    """Lazy-init the OpenAI async client."""
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI

        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


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
# OpenAI → Anthropic adapter helpers
# ──────────────────────────────────────────────────────────────────────


def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Translate Anthropic-style tool specs into OpenAI's chat.completions shape.

    Anthropic: ``{name, description, input_schema}``
    OpenAI:    ``{"type": "function", "function":
                  {"name", "description", "parameters"}}``

    The JSON Schema in ``input_schema`` maps directly onto OpenAI's
    ``parameters`` field.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _openai_response_to_blocks(message) -> list:
    """Normalize an OpenAI ChatCompletionMessage to Anthropic-style blocks.

    Tool-call arguments arrive as a JSON-encoded string on
    ``call.function.arguments`` — we ``json.loads`` so callers see a
    dict on ``block.input`` exactly like Anthropic and our Gemini
    adapter. Malformed JSON degrades to ``input={}`` so the builder's
    static gates are the layer that surfaces the failure rather than a
    cryptic ValueError here.
    """
    blocks: list[SimpleNamespace] = []
    text = getattr(message, "content", None)
    if text:
        blocks.append(SimpleNamespace(type="text", text=text))

    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        fn = getattr(call, "function", None)
        if fn is None:
            continue
        raw_args = getattr(fn, "arguments", None) or "{}"
        try:
            args = json.loads(raw_args)
        except (TypeError, ValueError) as e:
            log.warning(
                "openai tool-call arguments not valid JSON: %s — falling back to {}",
                str(e)[:120],
            )
            args = {}
        blocks.append(SimpleNamespace(type="tool_use", name=fn.name, input=args))
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
    cache_prefix: str | None = None,
) -> Any:
    """One-shot chat call routed through a provider.

    ``provider`` selects the SDK ("claude", "gemini", or "openai"). When
    ``None``, falls back to ``settings.llm_provider`` — this preserves
    the pre-multi-provider behavior for callers that don't care which
    backend handles their call. Roles that want explicit control
    (strategist, builder, player) should pass ``provider`` directly so
    a single generation can fan out to multiple providers in parallel.

    ``cache_prefix`` is an optional stable prefix of the user prompt
    that the caller wants cached across calls. On Anthropic this is
    sent as a separate text block with ``cache_control: ephemeral``,
    so subsequent calls within the 5-minute TTL hit the cache. On
    Gemini and OpenAI the prefix is prepended to ``user`` and no
    explicit caching is configured — the call still works, it just
    pays full token cost.

    Returns a list of content blocks. For text replies, read
    ``content[0].text``. For tool-use replies, look for a block where
    ``block.type == "tool_use"`` and read ``block.name`` / ``block.input``.
    The same return shape is produced regardless of provider.
    """
    resolved = provider or settings.llm_provider
    tool_names = [t["name"] for t in tools] if tools else []
    log.info(
        "complete provider=%s model=%s prompt_chars=%d max_tokens=%d "
        "tools=%s cache_prefix_chars=%d",
        resolved, model, len(user) + len(cache_prefix or ""),
        max_tokens, tool_names, len(cache_prefix or ""),
    )
    t0 = time.monotonic()
    try:
        if resolved == "gemini":
            blocks = await _complete_gemini(
                model, system, user, max_tokens, tools, cache_prefix
            )
        elif resolved == "claude":
            blocks = await _complete_claude(
                model, system, user, max_tokens, tools, cache_prefix
            )
        elif resolved == "openai":
            blocks = await _complete_openai(
                model, system, user, max_tokens, tools, cache_prefix
            )
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
    cache_prefix: str | None = None,
) -> str:
    """Convenience wrapper for plain-text replies.

    Returns the first text block's content, or ``""`` if no text block
    came back. ``provider`` is forwarded to ``complete()``; ``None``
    means fall back to the global default.
    """
    content = await complete(
        model, system, user,
        max_tokens=max_tokens,
        provider=provider,
        cache_prefix=cache_prefix,
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
    cache_prefix: str | None,
) -> Any:
    # Public exception names — ``anthropic._exceptions`` is internal and has
    # been reshuffled between SDK minor versions. Importing from the package
    # root is the supported path.
    from anthropic import APIError, RateLimitError

    client = _get_anthropic()

    # Build the user content. When the caller marks a stable prefix with
    # ``cache_prefix``, the message becomes a two-block list: the prefix
    # carries ``cache_control: ephemeral`` so a 5-minute write-through
    # cache covers its tokens, and the dynamic suffix is sent as a plain
    # text block. Without ``cache_prefix`` we keep the simpler string
    # form so traffic that doesn't benefit from caching doesn't pay the
    # extra structuring overhead.
    if cache_prefix:
        user_content: Any = [
            {
                "type": "text",
                "text": cache_prefix,
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": user},
        ]
    else:
        user_content = user

    # Build kwargs conditionally — passing ``tools=[]`` is sketchy across
    # SDK versions and ``tool_choice`` without ``tools`` is rejected.
    # Forcing ``tool_choice={"type": "any"}`` whenever tools are present
    # mirrors what the Gemini path does (``mode="ANY"``) so Claude can't
    # silently reply with prose for a builder request.
    kwargs: dict[str, Any] = {
        "model": model,
        "system": system,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = {"type": "any"}

    backoff = 1.0
    async with _sem:
        for attempt in range(5):
            try:
                msg = await client.messages.create(**kwargs)
            except RateLimitError:
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            except APIError:
                if attempt == 4:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2
                continue

            # Surface truncation explicitly. Without this, a builder
            # response that ran out of room mid-tool_use lands as a
            # malformed ``block.input`` dict and the failure shows up as
            # a confusing parse error two layers downstream rather than
            # a clean "we hit max_tokens" log line.
            stop_reason = getattr(msg, "stop_reason", None)
            if stop_reason == "max_tokens":
                log.warning(
                    "claude response truncated stop_reason=max_tokens "
                    "model=%s max_tokens=%d — caller may receive a "
                    "partial tool_use block",
                    model, max_tokens,
                )
            elif stop_reason in ("refusal", "pause_turn"):
                log.warning(
                    "claude response stop_reason=%s model=%s — content "
                    "may be empty or partial",
                    stop_reason, model,
                )

            return msg.content
    raise RuntimeError("unreachable")


async def _complete_gemini(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    tools: list[dict] | None,
    cache_prefix: str | None,
) -> Any:
    from google.genai import errors as genai_errors
    from google.genai import types

    client = _get_gemini()

    # Gemini does not expose an ephemeral-cache primitive on the
    # generate_content path that's stable enough to commit to here, so
    # we inline the prefix into the user content and accept full token
    # cost. Behaviour stays identical from the caller's viewpoint.
    contents = (cache_prefix + user) if cache_prefix else user

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
                    contents=contents,
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


async def _complete_openai(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    tools: list[dict] | None,
    cache_prefix: str | None,
) -> Any:
    """OpenAI chat-completions adapter normalised to Anthropic block shape.

    Differences from the Anthropic path that this function papers over:

      - System prompt goes inside ``messages`` as a ``role=system``
        entry, not as a top-level kwarg.
      - Tool spec wraps in ``{"type": "function", "function": {...}}``
        — one extra level vs Anthropic's flat shape.
      - Tool-call arguments come back as a JSON-encoded string on
        ``call.function.arguments``; the response adapter parses them.
      - Forcing tool emission uses ``tool_choice="required"`` (rather
        than Anthropic's ``{"type": "any"}`` or Gemini's ``mode="ANY"``).
      - ``max_completion_tokens`` is the post-o1 spelling of
        ``max_tokens``; it is accepted by all 4.x chat models too, so
        we standardise on it here.

    Caching is not configured — OpenAI's automatic prefix cache
    (which kicks in on prompts ≥ 1024 tokens) does not need explicit
    markers. The ``cache_prefix`` argument is simply prepended to the
    user content and ridden along on the request.
    """
    from openai import APIError, RateLimitError

    client = _get_openai()

    user_text = (cache_prefix + user) if cache_prefix else user
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = _anthropic_tools_to_openai(tools)
        kwargs["tool_choice"] = "required"

    backoff = 1.0
    async with _sem:
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(**kwargs)
            except RateLimitError:
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            except APIError:
                if attempt == 4:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2
                continue

            choice = response.choices[0]
            finish = getattr(choice, "finish_reason", None)
            if finish == "length":
                log.warning(
                    "openai response truncated finish_reason=length "
                    "model=%s max_tokens=%d — caller may receive a "
                    "partial tool_use block",
                    model, max_tokens,
                )
            elif finish == "content_filter":
                log.warning(
                    "openai response stopped by content filter model=%s "
                    "— content may be empty",
                    model,
                )

            blocks = _openai_response_to_blocks(choice.message)
            if not blocks:
                log.warning(
                    "openai empty response model=%s finish_reason=%r",
                    model, finish,
                )
            return blocks
    raise RuntimeError("unreachable")
