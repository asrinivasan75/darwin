"""Tests for cubist.orchestration.generation.run_generation_task."""
from __future__ import annotations

import json
from datetime import datetime

import pytest
from sqlmodel import Session, SQLModel, create_engine

from cubist.storage.models import EngineRow, GenerationRow


class FakeEngine:
    def __init__(self, name: str) -> None:
        self.name = name
        self.generation = 1
        self.lineage: list[str] = []

    async def select_move(self, board, time_remaining_ms):  # pragma: no cover
        raise NotImplementedError


@pytest.fixture()
def mem_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr("cubist.storage.db._engine", engine)
    return engine


@pytest.mark.asyncio
async def test_run_generation_task_resumes_from_last_champion(mem_db, monkeypatch):
    """After gen 1, the API path must start gen 2 from gen 1's champion."""
    gen1_winner = "gen1_prompt_abc123"
    gen1_path = "/absolute/path/gen1_prompt_abc123.py"

    with Session(mem_db) as s:
        s.add(GenerationRow(
            number=1,
            champion_before="baseline-v0",
            champion_after=gen1_winner,
            strategist_questions_json="[]",
            finished_at=datetime.utcnow(),
        ))
        s.add(EngineRow(
            name=gen1_winner,
            generation=1,
            parent_name="baseline-v0",
            code_path=gen1_path,
        ))
        s.commit()

    fake_champion = FakeEngine(gen1_winner)
    monkeypatch.setattr(
        "cubist.orchestration.generation.load_engine",
        lambda path: fake_champion,
    )

    called: dict = {}

    async def fake_run_generation(champion, generation_number):
        called["champion"] = champion
        called["generation_number"] = generation_number
        return champion

    monkeypatch.setattr(
        "cubist.orchestration.generation.run_generation",
        fake_run_generation,
    )

    from cubist.orchestration.generation import run_generation_task

    await run_generation_task()

    assert called["champion"].name == gen1_winner
    assert called["generation_number"] == 2


@pytest.mark.asyncio
async def test_run_generation_task_first_run_uses_baseline(mem_db, monkeypatch):
    """With an empty DB, the first generation must start from baseline-v0."""
    called: dict = {}

    async def fake_run_generation(champion, generation_number):
        called["champion"] = champion
        called["generation_number"] = generation_number
        return champion

    monkeypatch.setattr(
        "cubist.orchestration.generation.run_generation",
        fake_run_generation,
    )

    from cubist.orchestration.generation import run_generation_task

    await run_generation_task()

    assert called["champion"].name == "baseline-v0"
    assert called["generation_number"] == 1


def test_load_history_empty_for_first_generation(mem_db):
    from cubist.orchestration.generation import _load_history

    assert _load_history(1) == []


def test_load_history_summarises_prior_generations(mem_db):
    """Past generations come back with the questions tried, promoted flag,
    and winning_category derived from the new champion's name."""
    with Session(mem_db) as s:
        s.add(GenerationRow(
            number=1,
            champion_before="baseline-v0",
            champion_after="gen1-prompt-abc123",
            strategist_questions_json=json.dumps([
                {"category": "prompt", "text": "tighter system prompt"},
                {"category": "search", "text": "wrap in 1-ply minimax"},
            ]),
            finished_at=datetime.utcnow(),
        ))
        s.add(GenerationRow(
            number=2,
            champion_before="gen1-prompt-abc123",
            champion_after="gen1-prompt-abc123",
            strategist_questions_json=json.dumps([
                {"category": "book", "text": "opening book lookup"},
            ]),
            finished_at=datetime.utcnow(),
        ))
        s.commit()

    from cubist.orchestration.generation import _load_history

    history = _load_history(3)

    assert len(history) == 2
    assert history[0]["generation"] == 1
    assert history[0]["promoted"] is True
    assert history[0]["winning_category"] == "prompt"
    assert history[0]["questions"][0]["category"] == "prompt"
    assert history[1]["promoted"] is False
    assert history[1]["winning_category"] is None


def test_champion_question_picks_latest_promoted_generation():
    """When the latest promotion was generation 3, the champion question
    is the question from gen 3 whose category equals gen 3's winning
    category — not gen 4's (which didn't promote) and not gen 2's."""
    from cubist.orchestration.generation import _champion_question

    history = [
        {
            "generation": 1,
            "promoted": True,
            "winning_category": "prompt",
            "questions": [
                {"category": "prompt", "text": "old gen1 prompt question"},
                {"category": "search", "text": "old gen1 search question"},
            ],
        },
        {
            "generation": 2,
            "promoted": False,
            "winning_category": None,
            "questions": [{"category": "book", "text": "gen2 book question"}],
        },
        {
            "generation": 3,
            "promoted": True,
            "winning_category": "search",
            "questions": [
                {"category": "search", "text": "the actual current originator"},
                {"category": "book", "text": "gen3 book question"},
            ],
        },
        {
            "generation": 4,
            "promoted": False,
            "winning_category": None,
            "questions": [{"category": "evaluation", "text": "gen4 eval question"}],
        },
    ]

    cq = _champion_question(history)
    assert cq == {"category": "search", "text": "the actual current originator"}


def test_champion_question_none_when_no_promotion():
    """If every prior generation kept the baseline as champion, the
    champion has no originating strategist question."""
    from cubist.orchestration.generation import _champion_question

    history = [
        {
            "generation": 1,
            "promoted": False,
            "winning_category": None,
            "questions": [{"category": "prompt", "text": "..."}],
        },
    ]
    assert _champion_question(history) is None


def test_champion_question_none_for_empty_history():
    from cubist.orchestration.generation import _champion_question

    assert _champion_question([]) is None


def test_load_history_promotion_with_unparsable_champion_name(mem_db):
    """If champion_after doesn't follow the gen{N}-{cat}-{hash} format
    (e.g. baseline somehow re-promoted), winning_category stays None
    rather than raising — promoted is still True."""
    with Session(mem_db) as s:
        s.add(GenerationRow(
            number=1,
            champion_before="baseline-v0",
            champion_after="some-handcrafted-name",
            strategist_questions_json="[]",
            finished_at=datetime.utcnow(),
        ))
        s.commit()

    from cubist.orchestration.generation import _load_history

    history = _load_history(2)
    assert history[0]["promoted"] is True
    assert history[0]["winning_category"] is None
