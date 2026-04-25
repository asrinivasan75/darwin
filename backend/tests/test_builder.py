"""Tests for cubist.agents.builder.

We mock ``cubist.llm.complete`` to avoid live API calls. The forbidden-
import regex is exercised both as a positive test (legal code passes)
and as a negative test (subprocess-laden code is rejected).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cubist.agents.builder import (
    FORBIDDEN,
    REJECT_TERMINATIONS,
    REQUIRED_PATTERNS,
    build_engine,
    validate_engine,
)
from cubist.agents.strategist import Question

# A minimal "good" engine source: subclasses BaseLLMEngine, has the
# required `engine = …` symbol, has the async select_move signature, and
# attempts a real LLM call (with the standard try/except → first-legal
# fallback so a CI run without an API key still produces a legal move).
LEGAL_ENGINE_SOURCE = """\
import chess

from cubist.engines.base import BaseLLMEngine
from cubist.llm import complete_text
from cubist.config import settings


class CandidateEngine(BaseLLMEngine):
    def __init__(self):
        super().__init__(name=\"PLACEHOLDER\", generation=1, lineage=[\"baseline-v0\"])

    async def select_move(self, board, time_remaining_ms):
        try:
            text = await complete_text(
                settings.player_model,
                "You are a chess engine.",
                f"FEN: {board.fen()}\\nYour move:",
                max_tokens=10,
            )
            return board.parse_san(text.strip().split()[0])
        except Exception:
            return next(iter(board.legal_moves))


engine = CandidateEngine()
"""


def _fake_tool_use_block(code: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name="submit_engine", input={"code": code})


@pytest.fixture
def question() -> Question:
    return Question(
        index=0,
        category="prompt",
        text="Add a system-prompt sentence about prophylactic king safety.",
    )


@pytest.mark.asyncio
async def test_build_engine_writes_module(tmp_path, monkeypatch, question):
    """Happy path — file is written under generated/, named after category + sha."""
    monkeypatch.setattr(
        "cubist.agents.builder.GENERATED_DIR", tmp_path / "generated"
    )

    async def fake_complete(**kwargs):
        return [_fake_tool_use_block(LEGAL_ENGINE_SOURCE)]

    monkeypatch.setattr("cubist.agents.builder.complete", fake_complete)

    path = await build_engine(
        champion_code="x = 1",
        champion_name="baseline-v0",
        generation=1,
        question=question,
    )

    assert path.exists()
    assert path.name.startswith("gen1_prompt_")
    assert "select_move" in path.read_text()


@pytest.mark.asyncio
async def test_build_engine_rejects_forbidden_imports(tmp_path, monkeypatch, question):
    """Source containing a banned token raises ValueError."""
    monkeypatch.setattr(
        "cubist.agents.builder.GENERATED_DIR", tmp_path / "generated"
    )

    bad_source = LEGAL_ENGINE_SOURCE + "\nimport subprocess\n"

    async def fake_complete(**kwargs):
        return [_fake_tool_use_block(bad_source)]

    monkeypatch.setattr("cubist.agents.builder.complete", fake_complete)

    with pytest.raises(ValueError, match="forbidden"):
        await build_engine(
            champion_code="x = 1",
            champion_name="baseline-v0",
            generation=1,
            question=question,
        )


@pytest.mark.asyncio
async def test_build_engine_no_tool_use_raises(tmp_path, monkeypatch, question):
    """Reply without a submit_engine tool_use block is a hard failure."""
    monkeypatch.setattr(
        "cubist.agents.builder.GENERATED_DIR", tmp_path / "generated"
    )

    async def fake_complete(**kwargs):
        return [SimpleNamespace(type="text", text="here is some prose, sorry")]

    monkeypatch.setattr("cubist.agents.builder.complete", fake_complete)

    with pytest.raises(RuntimeError, match="tool_use"):
        await build_engine(
            champion_code="x = 1",
            champion_name="baseline-v0",
            generation=1,
            question=question,
        )


def test_forbidden_regex_matches_known_bad_patterns():
    bad = [
        "import subprocess",
        "import urllib.request",
        "result = eval('1+1')",
        "exec(some_string)",
        "import socket",
        "os.system('whoami')",
        "import importlib",
        "import requests",
    ]
    for b in bad:
        assert FORBIDDEN.search(b), f"FORBIDDEN failed to match: {b!r}"


def test_forbidden_regex_allows_legitimate_imports():
    good = [
        "import chess",
        "from cubist.engines.base import BaseLLMEngine",
        "from cubist.llm import complete_text",
        "import random",
        "import math",
        "result = 1 + 2",
    ]
    for g in good:
        assert not FORBIDDEN.search(g), f"FORBIDDEN incorrectly matched: {g!r}"


@pytest.mark.asyncio
async def test_validate_engine_handles_unimportable_module(tmp_path):
    """Garbage input is rejected at static-source phase OR at module load.

    The new validator runs static gates BEFORE attempting to load the file,
    so syntactically-broken garbage usually fails at "static:" first
    (no engine symbol, no async select_move, no LLM call). Either reason
    is acceptable — both prevent a bad candidate from reaching the
    tournament.
    """
    bad = tmp_path / "nope.py"
    bad.write_text("this is not valid python !!!")

    ok, reason = await validate_engine(bad)
    assert ok is False
    assert reason is not None
    assert any(tag in reason for tag in ("static:", "load:", "import:"))


# ---------------------------------------------------------------------------
# New required-pattern gates added in the games-not-running fix.
# Each test asserts build_engine raises ValueError with a useful reason.
# ---------------------------------------------------------------------------


def _drop_pattern(source: str, pattern_name: str) -> str:
    """Strip the line(s) needed to satisfy ``pattern_name`` from ``source``."""
    if pattern_name == "engine_symbol":
        return source.replace("engine = CandidateEngine()", "# engine line removed")
    if pattern_name == "async_select_move":
        return source.replace("async def select_move", "def select_move")
    if pattern_name == "llm_call":
        # Remove the LLM call line + the import; leave the fallback.
        out = source.replace("from cubist.llm import complete_text\n", "")
        out = out.replace(
            "        try:\n"
            '            text = await complete_text(\n'
            "                settings.player_model,\n"
            '                "You are a chess engine.",\n'
            '                f"FEN: {board.fen()}\\nYour move:",\n'
            "                max_tokens=10,\n"
            "            )\n"
            "            return board.parse_san(text.strip().split()[0])\n"
            "        except Exception:\n"
            "            return next(iter(board.legal_moves))\n",
            "        return next(iter(board.legal_moves))\n",
        )
        return out
    raise ValueError(f"unknown pattern_name {pattern_name!r}")


@pytest.mark.parametrize("pattern_name", [p[0] for p in REQUIRED_PATTERNS])
@pytest.mark.asyncio
async def test_build_engine_rejects_missing_required_pattern(
    tmp_path, monkeypatch, question, pattern_name
):
    """When the model omits engine= / async select_move / LLM call, build raises."""
    monkeypatch.setattr("cubist.agents.builder.GENERATED_DIR", tmp_path / "generated")
    monkeypatch.setattr("cubist.agents.builder.FAILED_DIR", tmp_path / "failures")
    bad_source = _drop_pattern(LEGAL_ENGINE_SOURCE, pattern_name)

    async def fake_complete(**kwargs):
        return [_fake_tool_use_block(bad_source)]

    monkeypatch.setattr("cubist.agents.builder.complete", fake_complete)

    with pytest.raises(ValueError, match=pattern_name):
        await build_engine(
            champion_code="x = 1",
            champion_name="baseline-v0",
            generation=1,
            question=question,
        )

    # The rejected response should be persisted for post-mortem.
    failures = list((tmp_path / "failures").glob("*.txt"))
    assert failures, "rejected response was not saved to FAILED_DIR"


@pytest.mark.asyncio
async def test_validate_engine_runs_static_gates_on_existing_file(tmp_path):
    """A hand-edited file that bypasses build_engine still hits the static gate."""
    # Looks like Python, but no engine symbol, no async, no LLM call.
    bad = tmp_path / "bad_engine.py"
    bad.write_text(
        "import chess\n"
        "from cubist.engines.base import BaseLLMEngine\n"
        "class C(BaseLLMEngine):\n"
        "    def select_move(self, board, time_remaining_ms):\n"
        "        return next(iter(board.legal_moves))\n"
        "# no engine = ...\n"
    )
    ok, reason = await validate_engine(bad)
    assert ok is False
    assert reason is not None
    # The static phase fires before module load and reports the missing pattern.
    assert reason.startswith("static:")


def test_reject_terminations_constant_includes_new_modes():
    """The new validator catches illegal_move and time, not just error."""
    assert "error" in REJECT_TERMINATIONS
    assert "illegal_move" in REJECT_TERMINATIONS
    assert "time" in REJECT_TERMINATIONS


def _async_pattern():
    for name, pattern, _ in REQUIRED_PATTERNS:
        if name == "async_select_move":
            return pattern
    raise AssertionError("async_select_move pattern not found")


def test_async_pattern_accepts_multiline_signature_with_type_annotations():
    """Regression: real LLMs (Gemini, GPT-4) format the signature multi-line
    with `: chess.Board`/`: int` annotations. The original regex required
    everything on one line and false-rejected every such candidate, leaving
    candidates=[] and the tournament with zero games. Reproducer:
    """
    real_gemini_source = (
        "class CandidateEngine(BaseLLMEngine):\n"
        "    async def select_move(\n"
        "        self,\n"
        "        board: chess.Board,\n"
        "        time_remaining_ms: int,\n"
        "    ) -> chess.Move:\n"
        "        pass\n"
    )
    assert _async_pattern().search(real_gemini_source)


def test_async_pattern_accepts_one_line_signature_too():
    """The compact form a hand-written engine might use must still match."""
    src = "async def select_move(self, board, time_remaining_ms):\n    pass\n"
    assert _async_pattern().search(src)


def test_async_pattern_rejects_non_async_def():
    """Sync `def select_move(...)` must still be rejected."""
    src = (
        "def select_move(\n"
        "    self,\n"
        "    board: chess.Board,\n"
        "    time_remaining_ms: int,\n"
        ") -> chess.Move:\n"
        "    pass\n"
    )
    assert not _async_pattern().search(src)


def test_async_pattern_rejects_missing_param():
    """Missing `time_remaining_ms` parameter must be rejected."""
    src = "async def select_move(self, board):\n    pass\n"
    assert not _async_pattern().search(src)


# ---------------------------------------------------------------------------
# chess module attribute hallucination check (gen1_sampling_*: chess.NAVY;
# gen1_evaluation_*: chess.between(...) — both real failures from a Gemini
# generation that the prior validator only caught at module-load or smoke-game
# time, polluting the dashboard with cryptic AttributeError noise).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_engine_rejects_chess_attribute_hallucination(
    tmp_path, monkeypatch, question
):
    """Gemini wrote `chess.NAVY: 300,` in a piece-value table. Caught now."""
    monkeypatch.setattr("cubist.agents.builder.GENERATED_DIR", tmp_path / "generated")
    monkeypatch.setattr("cubist.agents.builder.FAILED_DIR", tmp_path / "failures")
    bad_source = LEGAL_ENGINE_SOURCE.replace(
        "import chess", "import chess\nPIECE_VALUES = {chess.NAVY: 300}"
    )

    async def fake_complete(**kwargs):
        return [_fake_tool_use_block(bad_source)]

    monkeypatch.setattr("cubist.agents.builder.complete", fake_complete)

    with pytest.raises(ValueError, match="chess_attrs"):
        await build_engine(
            champion_code="x = 1",
            champion_name="baseline-v0",
            generation=1,
            question=question,
        )


@pytest.mark.asyncio
async def test_build_engine_rejects_nonexistent_chess_function(
    tmp_path, monkeypatch, question
):
    """Genuinely hallucinated function name (caught at static phase).

    Note: ``chess.between(a, b)`` IS real (2 args). When Gemini calls
    ``chess.between(a, b, c)`` with the wrong arg count, that's a TypeError
    at runtime — caught by the smoke-game phase, not static. This test uses
    ``chess.distance`` which doesn't exist at all, so the static check
    is what fires.
    """
    monkeypatch.setattr("cubist.agents.builder.GENERATED_DIR", tmp_path / "generated")
    monkeypatch.setattr("cubist.agents.builder.FAILED_DIR", tmp_path / "failures")
    bad_source = LEGAL_ENGINE_SOURCE.replace(
        "return next(iter(board.legal_moves))",
        "x = chess.distance(0, 5)\n            return next(iter(board.legal_moves))",
    )

    async def fake_complete(**kwargs):
        return [_fake_tool_use_block(bad_source)]

    monkeypatch.setattr("cubist.agents.builder.complete", fake_complete)

    with pytest.raises(ValueError, match="chess_attrs"):
        await build_engine(
            champion_code="x = 1",
            champion_name="baseline-v0",
            generation=1,
            question=question,
        )


def test_chess_attribute_check_accepts_real_attrs():
    """Sanity: every chess.X reference that's real must NOT be flagged."""
    from cubist.agents.builder import _check_hallucinated_chess_attrs

    real_source = (
        "import chess\n"
        "x = chess.PAWN\n"
        "y = chess.KNIGHT\n"
        "z = chess.WHITE\n"
        "a = chess.Board()\n"
        "b = chess.Move.from_uci('e2e4')\n"
        "c = chess.SQUARES\n"
        "d = chess.square(0, 0)\n"
        "e = chess.square_file(0)\n"
        "f = chess.A1\n"
    )
    assert _check_hallucinated_chess_attrs(real_source) is None


def test_chess_attribute_check_flags_hallucinations():
    """The real Gemini failure was chess.NAVY (a piece-type confusion).

    chess.between IS a real function (with 2 args) so it does not trip
    the static gate; that case shows up as a runtime TypeError caught
    by the smoke-game phase.
    """
    from cubist.agents.builder import _check_hallucinated_chess_attrs

    src = "import chess\nx = chess.NAVY\ny = chess.distance(0, 5)\n"
    reason = _check_hallucinated_chess_attrs(src)
    assert reason is not None
    assert "NAVY" in reason
    assert "distance" in reason
