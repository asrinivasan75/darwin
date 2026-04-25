"""Shared LLM client. Person E owns; A and C use it.

Single shared async client + global semaphore + retry. Every LLM call in
the system goes through here so we have one place to tune concurrency and
rate-limit handling.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic
from anthropic._exceptions import APIError as AnthropicAPIError
from anthropic._exceptions import RateLimitError
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from cubist.config import settings

log = logging.getLogger("cubist.llm")

_anthropic_client: AsyncAnthropic | None = None
_gemini_client: genai.Client | None = None
_sem = asyncio.Semaphore(settings.resolved_llm_max_concurrency)


@dataclass(frozen=True)
class TextBlock:
    text: str
    type: str = "text"


@dataclass(frozen=True)
class ToolUseBlock:
    name: str
    input: dict[str, Any]
    id: str | None = None
    type: str = "tool_use"


def _provider_for_model(model: str) -> str:
    if settings.llm_provider != "auto":
        return settings.llm_provider

    normalized = model.removeprefix("models/")
    if normalized.startswith("gemini-"):
        return "gemini"
    if normalized.startswith("claude-"):
        return "anthropic"
    return settings.resolved_llm_provider


def _get_anthropic_client() -> AsyncAnthropic:
    global _anthropic_client
    if not settings.anthropic_api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY or switch LLM_PROVIDER=gemini.")
    if _anthropic_client is None:
        _anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    api_key = settings.gemini_api_key or settings.google_api_key
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY for Gemini.")
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _has_long_string_field(schema: dict) -> bool:
    """Detect tool schemas Gemini 3 will fail to fill (MALFORMED_FUNCTION_CALL).

    Gemini consistently produces empty MALFORMED_FUNCTION_CALL responses
    when asked to emit a function call whose argument is a multi-KB string
    (e.g. submit_engine.code with minLength >= 100). We treat that as a
    signal to skip tool-calling for Gemini and rely on text fallback.
    """
    properties = (schema or {}).get("properties", {})
    for prop in properties.values():
        if isinstance(prop, dict) and prop.get("type") == "string" and prop.get("minLength", 0) >= 100:
            return True
    return False


def _gemini_tools(tools: list[dict] | None) -> list[genai_types.Tool] | None:
    if not tools:
        return None

    declarations = []
    for tool in tools:
        if _has_long_string_field(tool.get("input_schema", {})):
            log.info(
                "gemini stripping tool=%s before send: long-string args cause empty "
                "MALFORMED_FUNCTION_CALL responses; relying on text fallback",
                tool["name"],
            )
            continue
        declarations.append(
            genai_types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description"),
                parameters_json_schema=tool.get("input_schema", {"type": "object"}),
            )
        )
    if not declarations:
        return None
    return [genai_types.Tool(function_declarations=declarations)]


def _gemini_tool_config(tools: list[dict] | None) -> genai_types.ToolConfig | None:
    safe_tools = [t for t in (tools or []) if not _has_long_string_field(t.get("input_schema", {}))]
    if not safe_tools:
        return None

    return genai_types.ToolConfig(
        function_calling_config=genai_types.FunctionCallingConfig(
            mode=genai_types.FunctionCallingConfigMode.ANY,
            allowed_function_names=[tool["name"] for tool in safe_tools],
        )
    )


def _gemini_blocks(response: genai_types.GenerateContentResponse) -> list[TextBlock | ToolUseBlock]:
    blocks: list[TextBlock | ToolUseBlock] = []
    for candidate in response.candidates or []:
        if not candidate.content:
            continue
        for part in candidate.content.parts or []:
            if part.text:
                blocks.append(TextBlock(part.text))
            if part.function_call:
                function_call = part.function_call
                blocks.append(
                    ToolUseBlock(
                        name=function_call.name or "",
                        input=dict(function_call.args or {}),
                        id=getattr(function_call, "id", None),
                    )
                )
    try:
        fallback_text = response.text if not blocks else None
    except (ValueError, AttributeError):
        # response.text raises when no text part exists (e.g. MALFORMED_FUNCTION_CALL).
        fallback_text = None
    if fallback_text:
        blocks.append(TextBlock(fallback_text))
    return blocks


def _gemini_diagnostics(response: genai_types.GenerateContentResponse) -> str:
    finish = [str(getattr(c, "finish_reason", None)) for c in (response.candidates or [])]
    usage = response.usage_metadata
    parts = [f"finish_reason={finish}"]
    if usage is not None:
        parts.append(
            f"prompt_tokens={usage.prompt_token_count} "
            f"candidates_tokens={usage.candidates_token_count} "
            f"thoughts_tokens={usage.thoughts_token_count}"
        )
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None:
        parts.append(f"prompt_feedback={feedback}")
    return " ".join(parts)


def _retry_gemini_error(error: genai_errors.APIError) -> bool:
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    return isinstance(error, genai_errors.ServerError) or code == 429


async def complete(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 256,
    tools: list[dict] | None = None,
) -> Any:
    """One-shot chat call. Returns content blocks with Anthropic-compatible attrs.

    For text replies, take `content[0].text`. For tool-use replies, look for
    a `tool_use` block and read its `input` dict.
    """
    provider = _provider_for_model(model)
    if provider == "gemini":
        return await _complete_gemini(model, system, user, max_tokens, tools)
    return await _complete_anthropic(model, system, user, max_tokens, tools)


async def _complete_anthropic(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    tools: list[dict] | None,
) -> Any:
    client = _get_anthropic_client()
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
            except AnthropicAPIError:
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
) -> list[TextBlock | ToolUseBlock]:
    client = _get_gemini_client()

    # Gemini 3 counts thinking tokens against max_output_tokens. Even at
    # thinking_level=LOW the model often allocates most of the budget to
    # thinking and runs out of room for actual output (player engines with
    # max_tokens=256 saw thoughts_tokens=243, candidates_tokens=10, then
    # MAX_TOKENS empty). Our prompts don't materially benefit from thinking
    # — strategist/builder produce solid output without it — so disable it
    # globally for predictable behavior.
    effective_max = max(max_tokens, 16)
    config = genai_types.GenerateContentConfig(
        max_output_tokens=effective_max,
        system_instruction=system or None,
        tools=_gemini_tools(tools),
        tool_config=_gemini_tool_config(tools),
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    )

    tool_names = [tool["name"] for tool in (tools or [])]
    log.info(
        "gemini call model=%s tools=%s prompt_chars=%d max_output_tokens=%d",
        model,
        tool_names,
        len(user),
        effective_max,
    )

    backoff = 1.0
    async with _sem:
        for attempt in range(5):
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=user,
                    config=config,
                )
                blocks = _gemini_blocks(response)
                diag = _gemini_diagnostics(response)
                kinds = [getattr(b, "type", "?") for b in blocks]
                if blocks:
                    log.info("gemini ok model=%s blocks=%s %s", model, kinds, diag)
                    return blocks
                # Empty response. MAX_TOKENS / SAFETY are recoverable —
                # callers (e.g. in-game LLMs) typically have a fallback,
                # and raising would just convert "shrug" into a noisy
                # exception loop. Reserve the raise for unexpected empties
                # like MALFORMED_FUNCTION_CALL where there's a real bug to
                # surface.
                finish_reasons = {
                    str(getattr(c, "finish_reason", None))
                    for c in (response.candidates or [])
                }
                soft_reasons = {
                    "FinishReason.MAX_TOKENS",
                    "FinishReason.SAFETY",
                    "FinishReason.RECITATION",
                }
                if finish_reasons & soft_reasons:
                    log.warning("gemini empty (recoverable) model=%s %s", model, diag)
                    return []
                log.warning("gemini empty response model=%s %s", model, diag)
                raise RuntimeError(f"gemini returned no content ({diag})")
            except genai_errors.APIError as error:
                if attempt == 4 or not _retry_gemini_error(error):
                    log.exception("gemini api error model=%s attempt=%d", model, attempt)
                    raise
                log.warning(
                    "gemini retry model=%s attempt=%d backoff=%.1fs error=%r",
                    model,
                    attempt,
                    backoff,
                    error,
                )
                await asyncio.sleep(backoff)
                backoff *= 2
    raise RuntimeError("unreachable")


async def complete_text(model: str, system: str, user: str, max_tokens: int = 256) -> str:
    """Convenience wrapper for plain-text replies."""
    content = await complete(model, system, user, max_tokens=max_tokens)
    for block in content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""
