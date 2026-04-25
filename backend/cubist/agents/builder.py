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
import json
import logging
import re
from pathlib import Path

from cubist.agents.strategist import Question
from cubist.config import settings
from cubist.llm import complete

log = logging.getLogger("cubist.builder")

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


def _json_candidates(text: str) -> list[str]:
    candidates = [text.strip()]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.I)
    )

    first_obj, last_obj = text.find("{"), text.rfind("}")
    if first_obj != -1 and last_obj != -1 and first_obj < last_obj:
        candidates.append(text[first_obj : last_obj + 1])

    return candidates


def _code_from_text(text: str) -> str | None:
    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("code"), str):
            return data["code"]

    for match in re.finditer(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.I):
        code = match.group(1).strip()
        if "engine =" in code:
            return code

    stripped = text.strip()
    if "engine =" in stripped:
        starts = [
            index
            for token in ("from ", "import ", "class ")
            if (index := stripped.find(token)) != -1
        ]
        if starts:
            return stripped[min(starts) :]

    return None


def _write_engine(code: str, engine_name: str, safe_filename: str) -> Path:
    if FORBIDDEN.search(code):
        raise ValueError(
            f"builder code contains forbidden import / call "
            f"(engine={engine_name})"
        )
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out = GENERATED_DIR / safe_filename
    out.write_text(code)
    return out


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
    user += (
        "\n\nReturn the result by calling the `submit_engine` tool. "
        "If tool calling is unavailable, return exactly one fenced Python "
        "code block containing the full engine module and no extra prose."
    )

    log.info(
        "builder start engine=%s gen=%d category=%s model=%s",
        engine_name,
        generation,
        question.category,
        settings.builder_model,
    )
    content = await complete(
        model=settings.builder_model,
        system="You write Python chess engines.",
        user=user,
        max_tokens=16384,
        tools=[TOOL],
    )

    block_kinds = [getattr(b, "type", "?") for b in content]
    log.info("builder response engine=%s blocks=%s", engine_name, block_kinds)

    for block in content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_engine":
            log.info("builder tool_use ok engine=%s code_chars=%d", engine_name, len(block.input["code"]))
            return _write_engine(block.input["code"], engine_name, safe_filename)

    text = "\n".join(
        block.text for block in content if getattr(block, "type", None) == "text" and block.text
    )
    log.info("builder falling back to text parse engine=%s text_chars=%d", engine_name, len(text))
    code = _code_from_text(text)
    if code is not None:
        log.info("builder text parse ok engine=%s code_chars=%d", engine_name, len(code))
        return _write_engine(code, engine_name, safe_filename)

    excerpt = text[:200].replace("\n", " ")
    log.error("builder parse failed engine=%s excerpt=%r", engine_name, excerpt)
    raise RuntimeError(f"builder did not return tool_use or parseable code: {excerpt!r}")


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
        detail = getattr(result, "error", None) or "no detail"
        return False, f"engine crashed during smoke game: {detail}"
    if getattr(result, "termination", None) == "illegal_move":
        detail = getattr(result, "error", None) or "no detail"
        return False, f"engine made illegal move: {detail}"

    return True, None
