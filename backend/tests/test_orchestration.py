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

    async def fake_run_generation(incumbents, generation_number):
        called["incumbents"] = incumbents
        called["generation_number"] = generation_number
        return incumbents

    monkeypatch.setattr(
        "cubist.orchestration.generation.run_generation",
        fake_run_generation,
    )

    from cubist.orchestration.generation import run_generation_task

    await run_generation_task()

    assert called["incumbents"][0].name == gen1_winner
    assert called["generation_number"] == 2


@pytest.mark.asyncio
async def test_run_generation_task_first_run_uses_baseline(mem_db, monkeypatch):
    """With an empty DB, the first generation must start from baseline-v0."""
    called: dict = {}

    async def fake_run_generation(incumbents, generation_number):
        called["incumbents"] = incumbents
        called["generation_number"] = generation_number
        return incumbents

    monkeypatch.setattr(
        "cubist.orchestration.generation.run_generation",
        fake_run_generation,
    )

    from cubist.orchestration.generation import run_generation_task

    await run_generation_task()

    assert called["incumbents"][0].name == "baseline-v0"
    assert called["generation_number"] == 1


def test_champion_question_none_for_first_generation(mem_db):
    from cubist.orchestration.generation import _champion_question

    assert _champion_question(1) is None


def test_champion_question_picks_latest_promoted_generation(mem_db):
    """When the latest promotion was generation 3, the champion question
    is the question from gen 3 whose category equals gen 3's winning
    category — not gen 4's (which didn't promote) and not gen 1's."""
    with Session(mem_db) as s:
        s.add(GenerationRow(
            number=1,
            champion_before="baseline-v0",
            champion_after="gen1-prompt-aaaaaa",
            strategist_questions_json=json.dumps([
                {"category": "prompt", "text": "old gen1 prompt question"},
                {"category": "search", "text": "old gen1 search question"},
            ]),
            finished_at=datetime.utcnow(),
        ))
        s.add(GenerationRow(
            number=2,
            champion_before="gen1-prompt-aaaaaa",
            champion_after="gen1-prompt-aaaaaa",
            strategist_questions_json=json.dumps([
                {"category": "book", "text": "gen2 book question"},
            ]),
            finished_at=datetime.utcnow(),
        ))
        s.add(GenerationRow(
            number=3,
            champion_before="gen1-prompt-aaaaaa",
            champion_after="gen3-search-bbbbbb",
            strategist_questions_json=json.dumps([
                {"category": "search", "text": "the actual current originator"},
                {"category": "book", "text": "gen3 book question"},
            ]),
            finished_at=datetime.utcnow(),
        ))
        s.add(GenerationRow(
            number=4,
            champion_before="gen3-search-bbbbbb",
            champion_after="gen3-search-bbbbbb",
            strategist_questions_json=json.dumps([
                {"category": "evaluation", "text": "gen4 eval question"},
            ]),
            finished_at=datetime.utcnow(),
        ))
        s.commit()

    from cubist.orchestration.generation import _champion_question

    cq = _champion_question(5)
    assert cq == {"category": "search", "text": "the actual current originator"}


def test_champion_question_none_when_no_promotion(mem_db):
    """If every prior generation kept the baseline as champion, the
    champion has no originating strategist question."""
    with Session(mem_db) as s:
        s.add(GenerationRow(
            number=1,
            champion_before="baseline-v0",
            champion_after="baseline-v0",
            strategist_questions_json=json.dumps([
                {"category": "prompt", "text": "..."}
            ]),
            finished_at=datetime.utcnow(),
        ))
        s.commit()

    from cubist.orchestration.generation import _champion_question

    assert _champion_question(2) is None


def test_champion_question_none_for_unparsable_champion_name(mem_db):
    """If champion_after doesn't follow the gen{N}-{cat}-{hash} format
    (e.g. a hand-crafted promotion to baseline), the lookup returns None
    rather than guessing a category."""
    with Session(mem_db) as s:
        s.add(GenerationRow(
            number=1,
            champion_before="gen0-prompt-aaaaaa",
            champion_after="baseline-v0",
            strategist_questions_json="[]",
            finished_at=datetime.utcnow(),
        ))
        s.commit()

    from cubist.orchestration.generation import _champion_question

    assert _champion_question(2) is None
