"""Person C — builder + validator.

The builder calls the builder model (default ``claude-sonnet-4-6``) with
a ``submit_engine`` tool, validates the returned source against several
static gates, writes it to ``engines/generated/<name>.py``, and returns
the path. The validator imports that path through Person A's registry,
runs another static-source pass, and plays one short game vs
``RandomEngine`` to confirm the engine doesn't crash.

Failure modes this module guards against, in order from cheapest gate
to most expensive:

  1. **Forbidden import** (``FORBIDDEN`` regex) — `subprocess`,
     `os.system`, `eval(`, etc. Build-time refusal.
  2. **No tool_use** — model replied with prose. Build-time refusal.
  3. **Missing required structure** (``REQUIRED_PATTERNS``) — no
     ``engine = ...`` symbol, no ``async def select_move``, no LLM
     call. Build-time refusal. These were the silent-zero-games modes
     before this gate was added: the builder would write a file, the
     validator would load it, but ``round_robin`` ended up with
     ``[champion]`` alone and scheduled zero games.
  4. **Static-source check at validate time** — same gates re-run
     against whatever's on disk, in case a hand-edited file in
     ``engines/generated/`` skipped the build path.
  5. **Module load** via ``cubist.engines.registry.load_engine``.
  6. **Smoke game** vs ``RandomEngine``. We reject any termination in
     ``REJECT_TERMINATIONS`` — not just ``error`` — so engines that
     return illegal moves or time out are dropped before they can
     pollute a real tournament.

Every gate emits a structured log line via ``logger`` so the operator
running the orchestrator can see exactly which gate killed each
candidate. When a build fails before writing, the raw model response
is persisted to ``engines/generated/_failed_<name>.txt`` so the failure
mode can be reverse-engineered later.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from cubist.agents.strategist import Question
from cubist.config import settings
from cubist.llm import complete

logger = logging.getLogger("cubist.agents.builder")

PROMPT = (Path(__file__).parent / "prompts" / "builder_v1.md").read_text()

# Builder output goes here. We do NOT import GENERATED_DIR from
# cubist.engines.registry to avoid a circular dependency at module load
# time (registry is Person A's territory and may grow imports from us).
GENERATED_DIR = Path(__file__).parent.parent / "engines" / "generated"

# Where to dump raw LLM responses that we couldn't accept — useful for
# post-mortems when "why didn't any candidate validate this generation?"
FAILED_DIR = GENERATED_DIR / "_failures"

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


# Patterns the source MUST contain. These exist because before this gate
# the builder happily produced engines that loaded fine but never called
# the LLM (or had no engine symbol), so round_robin ended up with the
# champion alone and scheduled zero games. Every requirement is keyed by
# a name we can show in the failure log.
REQUIRED_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "engine_symbol",
        re.compile(r"(?m)^\s*engine\s*=\s*\w[\w.]*\s*\("),
        "module is missing a top-level `engine = YourEngineClass()` line — "
        "registry can't find the engine entry point",
    ),
    (
        "async_select_move",
        re.compile(
            r"async\s+def\s+select_move\s*\(\s*self\s*,\s*board\s*,\s*time_remaining_ms",
        ),
        "engine has no `async def select_move(self, board, time_remaining_ms)` — "
        "referee will await select_move and crash on a non-coroutine return",
    ),
    (
        "llm_call",
        re.compile(r"\bcomplete(?:_text)?\s*\("),
        "engine never calls `complete(...)` or `complete_text(...)` — "
        "this candidate would not actually use the LLM, so promoting it "
        "would be a regression vs the baseline",
    ),
]

# Terminations that the validator rejects. The previous version only
# caught ``error`` — meaning an engine that returns illegal moves or
# times out every move would PASS validation and then bleed games in
# the real tournament. We catch all three.
REJECT_TERMINATIONS = frozenset({"error", "illegal_move", "time"})


def _static_check_source(source: str) -> str | None:
    """Run all static gates against ``source``. Return reason on failure."""
    if FORBIDDEN.search(source):
        return "forbidden import / call in source"
    for name, pattern, reason in REQUIRED_PATTERNS:
        if not pattern.search(source):
            return f"{name}: {reason}"
    return None


def _save_failed_response(engine_name: str, raw: str, reason: str) -> Path | None:
    """Persist a rejected response so we can inspect it later.

    Returns the path written, or None if persisting itself failed
    (we never want diagnostic plumbing to bring down a generation).
    """
    try:
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        out = FAILED_DIR / f"{engine_name}.txt"
        out.write_text(f"# rejection reason: {reason}\n\n{raw}")
        return out
    except Exception as e:  # pragma: no cover — best-effort logging
        logger.warning("failed to save rejected response for %s: %r", engine_name, e)
        return None


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
        ValueError: if any static gate rejects the returned source.
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

    logger.info(
        "build_engine starting engine=%s category=%s gen=%d",
        engine_name, question.category, generation,
    )

    content = await complete(
        model=settings.builder_model,
        system="You write Python chess engines.",
        user=user,
        max_tokens=8192,
        tools=[TOOL],
    )

    # Capture every block we got so a non-tool_use response is loggable.
    block_summary = []
    chosen_code: str | None = None
    for block in content:
        bt = getattr(block, "type", "?")
        block_summary.append(bt)
        if bt == "tool_use" and getattr(block, "name", None) == "submit_engine":
            chosen_code = block.input.get("code", "")

    if chosen_code is None:
        # Save the prose so we can inspect WHY the model refused the tool.
        raw = "\n\n".join(
            getattr(b, "text", "") or "" for b in content if getattr(b, "type", None) == "text"
        )
        _save_failed_response(engine_name, raw, "no submit_engine tool_use block")
        logger.error(
            "build_engine no tool_use engine=%s blocks=%s "
            "(raw saved to engines/generated/_failures/%s.txt)",
            engine_name, block_summary, engine_name,
        )
        raise RuntimeError(
            f"builder did not return tool_use (engine={engine_name}, blocks={block_summary})"
        )

    # All static gates run BEFORE we touch the filesystem so a bad source
    # never exists at engines/generated/<name>.py.
    reason = _static_check_source(chosen_code)
    if reason is not None:
        _save_failed_response(engine_name, chosen_code, reason)
        logger.error(
            "build_engine rejected engine=%s reason=%s "
            "(raw saved to engines/generated/_failures/%s.txt)",
            engine_name, reason, engine_name,
        )
        raise ValueError(
            f"builder code rejected: {reason} (engine={engine_name})"
        )

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out = GENERATED_DIR / safe_filename
    out.write_text(chosen_code)
    logger.info(
        "build_engine wrote engine=%s path=%s lines=%d chars=%d",
        engine_name, out, chosen_code.count("\n") + 1, len(chosen_code),
    )
    return out


async def validate_engine(module_path: Path) -> tuple[bool, str | None]:
    """Smoke-test a candidate.

    Three-phase validation:

      1. **Static source check.** Read the module source and re-run every
         gate from ``_static_check_source``. Catches hand-edited files
         in ``engines/generated/`` that bypassed ``build_engine``.

      2. **Module load** via ``cubist.engines.registry.load_engine``.
         This is a runtime check that the module imports, the
         ``engine`` symbol exists, and ``isinstance(engine, Engine)``.

      3. **Smoke game** vs ``RandomEngine`` via ``play_game``. We reject
         any termination in ``REJECT_TERMINATIONS`` — error, illegal
         move, or time-loss — not just ``error``.

    Returns:
        ``(True, None)`` on success, ``(False, reason)`` on failure.
    """
    name_hint = Path(module_path).stem

    # Phase 1: static source check
    try:
        source = Path(module_path).read_text()
    except Exception as e:
        logger.error("validate_engine read failed engine=%s err=%r", name_hint, e)
        return False, f"read: {e!r}"

    reason = _static_check_source(source)
    if reason is not None:
        logger.error("validate_engine static reject engine=%s reason=%s", name_hint, reason)
        return False, f"static: {reason}"

    # Phase 2: module load (lazy imports so tests can run before A/B merge).
    try:
        from cubist.engines.random_engine import RandomEngine
        from cubist.engines.registry import load_engine
        from cubist.tournament.referee import play_game
    except Exception as e:  # pragma: no cover — import-time failure
        logger.error("validate_engine import-deps failed err=%r", e)
        return False, f"import: {e!r}"

    try:
        eng = load_engine(str(module_path))
    except Exception as e:
        logger.error("validate_engine load failed engine=%s err=%r", name_hint, e)
        return False, f"load: {e!r}"

    # Phase 3: smoke game.
    try:
        opp = RandomEngine(seed=0)
        # Short per-move budget so a misbehaving builder doesn't burn
        # the validator's wall-clock. The referee API may evolve; we
        # pass the conservative subset everyone has agreed on.
        result = await play_game(eng, opp, time_per_move_ms=10_000)
    except Exception as e:
        logger.error("validate_engine play raised engine=%s err=%r", name_hint, e)
        return False, f"play: {e!r}"

    term = getattr(result, "termination", None)
    if term in REJECT_TERMINATIONS:
        logger.error(
            "validate_engine smoke-game termination=%r engine=%s — rejecting",
            term, name_hint,
        )
        return False, f"smoke termination: {term}"

    logger.info(
        "validate_engine ok engine=%s smoke=%s/%s termination=%s",
        name_hint, result.white, result.black, term,
    )
    return True, None
