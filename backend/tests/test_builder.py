"""Tests for cubist.agents.builder.

We mock ``cubist.llm.complete`` to avoid live API calls. The forbidden-
import regex is exercised both as a positive test (legal code passes)
and as a negative test (subprocess-laden code is rejected).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cubist.agents.builder import FORBIDDEN, build_engine, validate_engine
from cubist.agents.strategist import Question

LEGAL_ENGINE_SOURCE = """\
import chess

from cubist.engines.base import BaseLLMEngine


class CandidateEngine(BaseLLMEngine):
    def __init__(self):
        super().__init__(name=\"PLACEHOLDER\", generation=1, lineage=[\"baseline-v0\"])

    async def select_move(self, board, time_remaining_ms):
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
    """When registry/referee can't load a module, validator returns (False, reason).
    This also doubles as a smoke test that the lazy imports don't break the
    function before the registry/referee stubs are filled in."""
    bad = tmp_path / "nope.py"
    bad.write_text("this is not valid python !!!")

    ok, reason = await validate_engine(bad)
    assert ok is False
    # We accept either an "import" error or a "load" error — depends on
    # whether registry.load_engine has been implemented yet.
    assert reason is not None
    assert any(tag in reason for tag in ("load:", "import:"))
