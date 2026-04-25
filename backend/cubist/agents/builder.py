"""Person C — builder + validator.

The builder calls the builder model (default ``claude-sonnet-4-6``) with
a ``submit_engine`` tool, validates the returned source against a
forbidden-imports regex, writes it to ``engines/generated/<name>.py``,
and returns the path. The validator imports that path through Person A's
registry and plays one short game vs ``RandomEngine`` to confirm the
engine doesn't crash.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from cubist.agents.strategist import Question
from cubist.config import settings
from cubist.llm import complete

PROMPT = (Path(__file__).parent / "prompts" / "builder_v1.md").read_text()

# Builder output goes here. We do NOT import GENERATED_DIR from
# cubist.engines.registry to avoid a circular dependency at module load
# time (registry is Person A's territory and may grow imports from us).
GENERATED_DIR = Path(__file__).parent.parent / "engines" / "generated"

TOOL = {
    "name": "submit_engine",
    "description": (
        "Submit the new engine module as a single Python source string. "
        "Must subclass cubist.engines.base.BaseLLMEngine, end with "
        "`engine = YourEngineClass()`, and use only the allowed imports."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "minLength": 100},
        },
        "required": ["code"],
    },
}

# Backstop against imports the prompt forbids. The regex is a
# minimum-bar check — the prompt is the primary contract — but a builder
# that slips through this regex is something we want to know about
# immediately, not after it runs.
# Each alternative carries its own word-boundary: the final ``\b`` from a
# single outer group fails on patterns ending in ``(`` (eval/exec) because
# both the ``(`` and the next char are non-word, so no boundary fires.
FORBIDDEN = re.compile(
    r"(?:"
    r"\bsubprocess\b|\bos\.system\b|\bsocket\b|"
    r"\beval\s*\(|\bexec\s*\(|"
    r"\bimportlib\b|\burllib\b|\brequests\b|\bhttpx\b|"
    r"\basyncio\.subprocess\b|\bpty\b|\bfcntl\b"
    r")"
)


async def build_engine(
    champion_code: str,
    champion_name: str,
    generation: int,
    question: Question,
) -> Path:
    """Generate a candidate engine module and return its path.

    Args:
        champion_code: source of the current champion module.
        champion_name: ``engine.name`` of the current champion (used to
            derive a unique filename and to populate ``lineage``).
        generation: the new candidate's generation number.
        question: the strategist question this candidate is answering.

    Returns:
        Path to the written ``.py`` module under ``engines/generated/``.

    Raises:
        ValueError: if the returned source contains a forbidden import.
        RuntimeError: if the model never produced a ``tool_use`` block.
    """
    short = hashlib.sha1(question.text.encode()).hexdigest()[:6]
    engine_name = f"gen{generation}-{question.category}-{short}"
    safe_filename = engine_name.replace("-", "_") + ".py"

    user = PROMPT.format(
        category=question.category,
        question_text=question.text,
        champion_code=champion_code,
        engine_name=engine_name,
        generation=generation,
        champion_name=champion_name,
    )

    content = await complete(
        model=settings.builder_model,
        system="You write Python chess engines.",
        user=user,
        max_tokens=4096,
        tools=[TOOL],
    )

    for block in content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_engine":
            code = block.input["code"]
            if FORBIDDEN.search(code):
                raise ValueError(
                    f"builder code contains forbidden import / call "
                    f"(engine={engine_name})"
                )
            GENERATED_DIR.mkdir(parents=True, exist_ok=True)
            out = GENERATED_DIR / safe_filename
            out.write_text(code)
            return out

    raise RuntimeError("builder did not return tool_use")


async def validate_engine(module_path: Path) -> tuple[bool, str | None]:
    """Smoke-test a candidate.

    Loads the module via ``cubist.engines.registry.load_engine`` and
    plays one short game vs ``RandomEngine`` using
    ``cubist.tournament.referee.play_game``. Any exception during load,
    play, or termination ``error`` adjudication counts as a failed
    validation.

    Returns:
        ``(True, None)`` on success, ``(False, reason)`` on failure.
    """
    # Lazy imports — registry / referee may not yet be implemented when
    # Track A and Track B haven't merged. The validator is only called by
    # Person E's orchestrator after those merges land.
    try:
        from cubist.engines.random_engine import RandomEngine
        from cubist.engines.registry import load_engine
        from cubist.tournament.referee import play_game
    except Exception as e:  # pragma: no cover — import-time failure
        return False, f"import: {e!r}"

    try:
        eng = load_engine(str(module_path))
    except Exception as e:
        return False, f"load: {e!r}"

    try:
        opp = RandomEngine(seed=0)
        # Short per-move budget so a misbehaving builder doesn't burn time.
        # The referee API may evolve; we take the conservative subset of
        # parameters everyone has agreed on (white, black, time_per_move_ms).
        result = await play_game(eng, opp, time_per_move_ms=10_000)
    except Exception as e:
        return False, f"play: {e!r}"

    if getattr(result, "termination", None) == "error":
        return False, "engine crashed during smoke game"

    return True, None
