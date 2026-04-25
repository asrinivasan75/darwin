"""Adversary agent — critiques builder-generated engine code.

Sits between the builder and the validator in the per-candidate chain:

    strategist (propose) → builder (code) → adversary (critique) → fixer (revise) → validator

The adversary reads the builder's source and the originating question
and returns a focused critique paragraph: what's likely to forfeit a
game, what drifted off the question's category, what the validator
would reject. The fixer then runs a second builder-style call with the
critique baked into its prompt.

The adversary is intentionally a different model role from the builder
— pairing the same model family on both sides tends to rubber-stamp
its own output. ``settings.adversary_provider`` and
``settings.adversary_model`` exist so the operator can pin the
adversary to a different provider (e.g. builder=gemini, adversary=
claude) without restarting other roles.

Failure mode: if the LLM call errors out, returns ``""``. The
orchestrator treats an empty critique as "no fixes needed" and skips
the fixer step, so an adversary outage degrades cleanly to the
pre-adversary pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from darwin.agents.strategist import Question
from darwin.config import settings
from darwin.llm import complete_text

logger = logging.getLogger("darwin.agents.adversary")

PROMPT = (Path(__file__).parent / "prompts" / "adversary_v1.md").read_text()


async def critique_engine(question: Question, code: str, engine_name: str) -> str:
    """Return an adversarial critique of ``code`` for ``question``.

    Empty string on any failure (LLM error, empty response, exception).
    Callers should treat empty as "skip the fixer pass" rather than
    blocking the candidate.
    """
    user = PROMPT.format(
        category=question.category,
        question_text=question.text,
        engine_name=engine_name,
        engine_code=code,
    )

    logger.info(
        "critique_engine starting engine=%s category=%s",
        engine_name, question.category,
    )

    try:
        text = await complete_text(
            model=settings.adversary_model,
            system=(
                "You are a critical reviewer of classical chess-engine code. "
                "Be specific, terse, and grounded in the code in front of you."
            ),
            user=user,
            max_tokens=600,
            provider=settings.provider_for("adversary"),
        )
    except Exception as exc:
        logger.warning(
            "adversary LLM call for engine=%s failed, skipping critique: %s",
            engine_name, exc,
        )
        return ""

    text = (text or "").strip()
    if len(text) < 20:
        logger.info(
            "critique_engine produced short/empty critique engine=%s len=%d — skipping fixer",
            engine_name, len(text),
        )
        return ""

    logger.info(
        "critique_engine ok engine=%s critique_chars=%d",
        engine_name, len(text),
    )
    return text
