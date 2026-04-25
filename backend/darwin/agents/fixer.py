"""Fixer agent — revises builder code in response to the adversary's critique.

Third stage in the propose / code / critique / fix chain. The fixer is
*the same role* as the builder (it writes engine code) but runs a
second LLM call with the original code and the adversary's critique
both included in its prompt. The same provider/model as the builder
is used by default, since "fix this code" should be done by the same
model family that wrote it — we want a revision, not a rewrite.

Reuses the builder's static gates, ``submit_engine`` tool schema, and
on-disk layout. A successful fix overwrites the original engine file
in place; a failed fix logs the failure and leaves the original file
untouched, so the candidate degrades gracefully to the unfixed
version rather than being dropped from the cohort.
"""

from __future__ import annotations

import logging
from pathlib import Path

from darwin.agents.builder import (
    GENERATED_DIR,
    TOOL,
    _save_failed_response,
    _static_check_source,
)
from darwin.agents.strategist import Question
from darwin.config import settings
from darwin.llm import complete

logger = logging.getLogger("darwin.agents.fixer")

PROMPT = (Path(__file__).parent / "prompts" / "fixer_v1.md").read_text()


async def fix_engine(
    path: Path,
    question: Question,
    critique: str,
    champion_code: str,
    champion_name: str,
    generation: int,
) -> Path:
    """Apply ``critique`` to the engine at ``path`` and overwrite in place.

    Args:
        path: file written by ``build_engine``. Read-modify-write.
        question: the strategist question this engine answers.
        critique: adversary output. If empty/whitespace, the fixer is
            a no-op and returns ``path`` unchanged.
        champion_code: source of the engine being modified — included
            in the fixer's prompt so the model can re-anchor if the
            adversary said the candidate drifted off-scope.
        champion_name: ``engine.name`` of the champion (lineage anchor).
        generation: the candidate's generation number.

    Returns:
        ``path`` (always the same path — fix is in-place when it
        succeeds, no-op when it fails).
    """
    if not critique or not critique.strip():
        logger.info("fix_engine no-op (empty critique) path=%s", path)
        return path

    engine_name = path.stem.replace("_", "-")

    try:
        original_code = path.read_text()
    except Exception as exc:
        logger.error(
            "fix_engine read failed engine=%s path=%s err=%r — keeping original",
            engine_name, path, exc,
        )
        return path

    user = PROMPT.format(
        category=question.category,
        question_text=question.text,
        critique=critique,
        original_code=original_code,
        champion_code=champion_code,
        engine_name=engine_name,
        generation=generation,
        champion_name=champion_name,
    )

    logger.info(
        "fix_engine starting engine=%s category=%s critique_chars=%d",
        engine_name, question.category, len(critique),
    )

    try:
        content = await complete(
            model=settings.builder_model,
            system="You revise Python chess engines based on critique.",
            user=user,
            max_tokens=16384,
            tools=[TOOL],
            provider=settings.provider_for("builder"),
        )
    except Exception as exc:
        logger.warning(
            "fix_engine LLM call failed engine=%s err=%r — keeping original",
            engine_name, exc,
        )
        return path

    block_summary: list[str] = []
    revised_code: str | None = None
    for block in content:
        bt = getattr(block, "type", "?")
        block_summary.append(bt)
        if bt == "tool_use" and getattr(block, "name", None) == "submit_engine":
            revised_code = block.input.get("code", "")

    if revised_code is None:
        raw = "\n\n".join(
            getattr(b, "text", "") or ""
            for b in content
            if getattr(b, "type", None) == "text"
        )
        _save_failed_response(
            f"{engine_name}_fix",
            raw,
            "fixer: no submit_engine tool_use block",
        )
        logger.warning(
            "fix_engine no tool_use engine=%s blocks=%s — keeping original "
            "(raw saved to engines/generated/_failures/%s_fix.txt)",
            engine_name, block_summary, engine_name,
        )
        return path

    reason = _static_check_source(revised_code)
    if reason is not None:
        _save_failed_response(f"{engine_name}_fix", revised_code, reason)
        logger.warning(
            "fix_engine rejected engine=%s reason=%s — keeping original "
            "(revised source saved to engines/generated/_failures/%s_fix.txt)",
            engine_name, reason, engine_name,
        )
        return path

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(revised_code)
    logger.info(
        "fix_engine wrote engine=%s lines=%d chars=%d",
        engine_name, revised_code.count("\n") + 1, len(revised_code),
    )
    return path
