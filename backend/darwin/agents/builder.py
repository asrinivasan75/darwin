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
  5. **Module load** via ``darwin.engines.registry.load_engine``.
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

import ast
import asyncio
import hashlib
import logging
import re
from pathlib import Path

import chess

from darwin.agents.strategist import Question
from darwin.config import settings
from darwin.llm import complete

logger = logging.getLogger("darwin.agents.builder")

PROMPT = (Path(__file__).parent / "prompts" / "builder_v1.md").read_text()

# Builder output goes here. We do NOT import GENERATED_DIR from
# darwin.engines.registry to avoid a circular dependency at module load
# time (registry is Person A's territory and may grow imports from us).
GENERATED_DIR = Path(__file__).parent.parent / "engines" / "generated"

# Where to dump raw LLM responses that we couldn't accept — useful for
# post-mortems when "why didn't any candidate validate this generation?"
FAILED_DIR = GENERATED_DIR / "_failures"

TOOL = {
    "name": "submit_engine",
    "description": (
        "Submit the new engine module as a single Python source string. "
        "Must subclass darwin.engines.base.BaseLLMEngine, end with "
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
        # Match `async def select_move(` then anywhere within the param list
        # (which may span multiple lines and include `: chess.Board` style
        # type annotations) the names ``board`` and ``time_remaining_ms``,
        # then the closing ``)``. The previous version required the three
        # params on one line with no annotations and false-rejected every
        # well-formatted engine Gemini emits.
        re.compile(
            r"async\s+def\s+select_move\s*\("
            r"[^)]*\bboard\b"
            r"[^)]*\btime_remaining_ms\b"
            r"[^)]*\)",
        ),
        "engine has no `async def select_move(self, board, time_remaining_ms)` — "
        "referee will await select_move and crash on a non-coroutine return",
    ),
    # EXPERIMENT (branch only): llm_call requirement dropped. We now allow
    # candidates that don't call the LLM at runtime — i.e. pure-Python
    # chess engines (alpha-beta, evaluation heuristics, MCTS, etc.) that
    # the LLM *wrote* but that play independently. Vastly faster (~ms
    # per move vs ~1s) and no quota at play time.
]

# Terminations that the validator rejects. The previous version only
# caught ``error`` — meaning an engine that returns illegal moves or
# times out every move would PASS validation and then bleed games in
# the real tournament. We catch all three.
REJECT_TERMINATIONS = frozenset({"error", "illegal_move", "time"})


# Valid python-chess module attributes — discovered at module load via
# ``dir(chess)``. We use this to catch builder hallucinations like
# ``chess.NAVY`` (should be chess.KNIGHT) or ``chess.between(...)``
# (function that doesn't exist) BEFORE the engine module is loaded —
# otherwise the failure is just an AttributeError on engine import or,
# worse, mid-game during the smoke run.
_VALID_CHESS_ATTRS = frozenset(dir(chess))

# ``chess.<NAME>`` references in source. We only care about the first
# attribute lookup (``chess.X``); deeper accesses (``chess.X.Y``) are
# resolved at runtime by Python and not relevant to the static check.
_CHESS_REF_RE = re.compile(r"\bchess\.([A-Za-z_]\w*)")


def _check_hallucinated_chess_attrs(source: str) -> str | None:
    """Detect ``chess.<NAME>`` references where ``<NAME>`` is not a real
    attribute of the python-chess module.

    Examples this catches:
      - ``chess.NAVY`` (should be ``chess.KNIGHT``)
      - ``chess.between(a, b, c)`` — function doesn't exist; the
        canonical helper is ``chess.SquareSet.between(a, b)``
      - ``chess.distance`` — doesn't exist
      - ``chess.legal_uci_moves`` — doesn't exist (it's a method on
        ``board.legal_moves``)

    The check inspects ``dir(chess)`` at module-load time so it picks up
    anything python-chess ships, including dynamically-added attributes.
    Any mismatch is reported with the offending name(s) so the
    rejection log line is actionable.
    """
    used = set(_CHESS_REF_RE.findall(source))
    bogus = sorted(used - _VALID_CHESS_ATTRS)
    if bogus:
        return (
            f"hallucinated chess module attribute(s) {bogus} — these names "
            f"do not exist in python-chess. Common confusions: chess.NAVY → "
            f"chess.KNIGHT; chess.between(...) → not a function; "
            f"chess.distance → not a function. Use only attributes that "
            f"exist on the real `chess` module."
        )
    return None


# Names of LLM-call helpers from darwin.llm. Catching either spelling
# (await complete_text(...) or await complete(...)) lets us detect any
# engine that consults the model from inside select_move.
_LLM_CALL_NAMES = frozenset({"complete", "complete_text"})


def _check_llm_call_in_loop(source: str) -> str | None:
    """Reject engines that call the LLM inside a `for` loop in select_move.

    The pattern we're catching is "evaluate every legal move with the LLM",
    e.g. ::

        async def select_move(self, board, time_remaining_ms):
            for move in board.legal_moves:           # ← O(legal_moves)
                board.push(move)
                score = await complete_text(...)     # ← LLM call per move
                board.pop()

    With ~20–30 legal moves per turn at ~0.5s per Gemini call, this
    pattern produces 10–15 seconds *per move* and 20+ minutes per smoke
    game. The 60s smoke wall-clock cap will eventually reject it, but
    only after burning a minute of validator time and a couple hundred
    LLM calls. Detecting it statically rejects in <100ms with no API
    spend.

    We deliberately allow LLM calls *outside* loops in select_move
    (the typical "ask the LLM for the best move once" pattern) and
    LLM calls *inside* loops in helper methods that are not select_move.
    The smoke validator is the second line of defense for those.

    Returns ``None`` if clean, or a human-readable reason on rejection.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"syntax error: {e.msg} at line {e.lineno}"

    # ast.walk doesn't track parent links, so attach them manually.
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent  # type: ignore[attr-defined]

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Await):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if isinstance(func, ast.Name):
            called = func.id
        elif isinstance(func, ast.Attribute):
            called = func.attr
        else:
            continue
        if called not in _LLM_CALL_NAMES:
            continue

        # Walk up the parent chain. The Await counts as a violation only
        # if there is a `for` loop ancestor that itself sits inside an
        # `async def select_move`. Helper methods are out of scope.
        cur = node
        in_for = False
        in_select_move = False
        while True:
            cur = getattr(cur, "_parent", None)
            if cur is None:
                break
            if isinstance(cur, ast.For):
                in_for = True
            if isinstance(cur, ast.AsyncFunctionDef) and cur.name == "select_move":
                in_select_move = True
                break
        if in_for and in_select_move:
            violations.append(called)

    if violations:
        return (
            f"select_move calls the LLM ({sorted(set(violations))}) inside a "
            "`for` loop. That's an O(legal_moves) ~20–30 LLM calls per turn, "
            "which makes the engine pathologically slow and forfeits every "
            "tournament game on time. Restructure to call the LLM at most "
            "once per turn (e.g. ask for the best move directly, not for an "
            "evaluation of each candidate)."
        )
    return None


def _static_check_source(source: str) -> str | None:
    """Run all static gates against ``source``. Return reason on failure."""
    if FORBIDDEN.search(source):
        return "forbidden import / call in source"
    for name, pattern, reason in REQUIRED_PATTERNS:
        if not pattern.search(source):
            return f"{name}: {reason}"
    bogus = _check_hallucinated_chess_attrs(source)
    if bogus is not None:
        return f"chess_attrs: {bogus}"
    # llm-call-in-loop check disabled on this experimental branch — engines
    # are allowed (encouraged) to skip the LLM at play time entirely.
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
    runner_up_code: str | None = None,
    runner_up_name: str | None = None,
) -> Path:
    """Generate a candidate engine module and return its path.

    Args:
        champion_code: source of the current champion module.
        champion_name: ``engine.name`` of the current champion (used to
            derive a unique filename and to populate ``lineage``).
        generation: the new candidate's generation number.
        question: the strategist question this candidate is answering.
        runner_up_code: source of the previous gen's runner-up engine,
            shown to the builder as context. The new candidate is still
            modelled on ``champion_code`` (which seeds the file shape
            and naming convention), but the builder may borrow specific
            ideas from the runner-up — e.g. its prompt style, its
            evaluation function — without being forced into a hybrid.
        runner_up_name: ``engine.name`` of the runner-up. Used only for
            the prompt's labelling; not added to ``lineage``.

    Returns:
        Path to the written ``.py`` module under ``engines/generated/``.

    Raises:
        ValueError: if any static gate rejects the returned source.
        RuntimeError: if the model never produced a ``tool_use`` block.
    """
    short = hashlib.sha1(question.text.encode()).hexdigest()[:6]
    engine_name = f"gen{generation}-{question.category}-{short}"
    safe_filename = engine_name.replace("-", "_") + ".py"

    runner_up_block = (
        runner_up_code
        if runner_up_code is not None
        else "(no runner-up — first generation, only baseline-v0 available)"
    )
    runner_up_label = runner_up_name or "-"

    user = PROMPT.format(
        category=question.category,
        question_text=question.text,
        champion_code=champion_code,
        engine_name=engine_name,
        generation=generation,
        champion_name=champion_name,
        runner_up_code=runner_up_block,
        runner_up_name=runner_up_label,
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
        provider=settings.provider_for("builder"),
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

      2. **Module load** via ``darwin.engines.registry.load_engine``.
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
        from darwin.engines.random_engine import RandomEngine
        from darwin.engines.registry import load_engine
        from darwin.tournament.referee import play_game
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
        #
        # Total wall-clock cap of 60s catches engines that pass the
        # per-move budget but design themselves into death-by-thousand-
        # LLM-calls patterns (e.g. an evaluator that calls the LLM once
        # per legal move, every move). Such an engine is technically
        # "valid" — it returns moves before the per-move timeout —
        # but the smoke game would still take ~20 minutes to complete
        # and the resulting candidate would lose every tournament game
        # on time anyway. Rejecting fast keeps the dashboard live.
        result = await asyncio.wait_for(
            play_game(eng, opp, time_per_move_ms=10_000),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        logger.error(
            "validate_engine smoke-game exceeded 60s engine=%s — rejecting "
            "(likely an N-LLM-calls-per-move pattern)",
            name_hint,
        )
        return False, "smoke too slow (>60s — likely calls LLM per legal move)"
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
