"""Tests for darwin.agents.fixer.

The fixer is best-effort: any failure mode (empty critique, LLM error,
no tool_use, static-gate rejection) must leave the original engine
file untouched so the un-revised candidate still gets validated.
Those degraded paths are the load-bearing contract here, so they
each get a dedicated test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from darwin.agents import fixer
from darwin.agents.fixer import fix_engine
from darwin.agents.strategist import Question

ORIGINAL_SRC = """\
import chess
from darwin.engines.base import BaseLLMEngine


class CandidateEngine(BaseLLMEngine):
    def __init__(self):
        super().__init__(name="gen1-search-abc123", generation=1, lineage=["baseline-v0"])

    async def select_move(self, board, time_remaining_ms):
        return next(iter(board.legal_moves))


engine = CandidateEngine()
"""

REVISED_SRC = ORIGINAL_SRC.replace(
    "return next(iter(board.legal_moves))",
    "# revised by fixer\n        return next(iter(board.legal_moves))",
)


def _tool_use_block(code: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name="submit_engine", input={"code": code})


@pytest.fixture
def question() -> Question:
    return Question(
        index=0,
        category="search",
        text="Add iterative deepening up to depth 4.",
    )


@pytest.fixture
def engine_path(tmp_path):
    p = tmp_path / "gen1_search_abc123.py"
    p.write_text(ORIGINAL_SRC)
    return p


@pytest.mark.asyncio
async def test_empty_critique_is_noop(monkeypatch, question, engine_path):
    """A blank critique must skip the LLM call entirely and leave the file alone."""
    called = False

    async def fake_complete(**kwargs):
        nonlocal called
        called = True
        return [_tool_use_block(REVISED_SRC)]

    monkeypatch.setattr(fixer, "complete", fake_complete)

    out = await fix_engine(
        engine_path,
        question,
        critique="   ",
        champion_code="x = 1",
        champion_name="baseline-v0",
        generation=1,
    )

    assert out == engine_path
    assert engine_path.read_text() == ORIGINAL_SRC
    assert not called


@pytest.mark.asyncio
async def test_successful_fix_overwrites_file(monkeypatch, tmp_path, question, engine_path):
    monkeypatch.setattr("darwin.agents.builder.GENERATED_DIR", tmp_path / "generated")

    captured: dict = {}

    async def fake_complete(**kwargs):
        captured["user"] = kwargs["user"]
        captured["provider"] = kwargs.get("provider")
        return [_tool_use_block(REVISED_SRC)]

    monkeypatch.setattr(fixer, "complete", fake_complete)

    out = await fix_engine(
        engine_path,
        question,
        critique="The search depth is fixed — add iterative deepening as the question requested.",
        champion_code="x = 1",
        champion_name="baseline-v0",
        generation=1,
    )

    assert out == engine_path
    assert engine_path.read_text() == REVISED_SRC
    # Prompt must include both the original code and the critique.
    assert "iterative deepening as the question requested" in captured["user"]
    assert "CandidateEngine" in captured["user"]


@pytest.mark.asyncio
async def test_llm_failure_keeps_original(monkeypatch, question, engine_path):
    async def boom(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(fixer, "complete", boom)

    out = await fix_engine(
        engine_path,
        question,
        critique="anything non-empty here, long enough to be meaningful",
        champion_code="x = 1",
        champion_name="baseline-v0",
        generation=1,
    )

    assert out == engine_path
    assert engine_path.read_text() == ORIGINAL_SRC


@pytest.mark.asyncio
async def test_no_tool_use_keeps_original(monkeypatch, tmp_path, question, engine_path):
    monkeypatch.setattr("darwin.agents.builder.FAILED_DIR", tmp_path / "failures")

    async def text_only(**kwargs):
        return [SimpleNamespace(type="text", text="I refuse to revise this.")]

    monkeypatch.setattr(fixer, "complete", text_only)

    out = await fix_engine(
        engine_path,
        question,
        critique="needs revision, original critique is more than 20 characters",
        champion_code="x = 1",
        champion_name="baseline-v0",
        generation=1,
    )

    assert out == engine_path
    assert engine_path.read_text() == ORIGINAL_SRC


@pytest.mark.asyncio
async def test_invalid_revision_keeps_original(monkeypatch, tmp_path, question, engine_path):
    """If the fixer's revised code fails a static gate, the original wins."""
    monkeypatch.setattr("darwin.agents.builder.FAILED_DIR", tmp_path / "failures")

    bad = ORIGINAL_SRC + "\nimport subprocess  # banned\n"

    async def fake(**kwargs):
        return [_tool_use_block(bad)]

    monkeypatch.setattr(fixer, "complete", fake)

    out = await fix_engine(
        engine_path,
        question,
        critique="critique that is long enough to qualify as a non-empty review",
        champion_code="x = 1",
        champion_name="baseline-v0",
        generation=1,
    )

    assert out == engine_path
    assert engine_path.read_text() == ORIGINAL_SRC
