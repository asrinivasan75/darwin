"""Tests for cubist.agents.strategist on the experimental pure-code branch.

The strategist is deterministic now — no LLM calls. Tests cover:
  - 4 questions returned, one per category in CATEGORIES_USED
  - All four categories distinct, all from the allowed set
  - Different generations get different question text (rotation works)
  - Optional kwargs (champion_code, runner_up_code, champion_question)
    are accepted without error and don't affect the output
"""

from __future__ import annotations

import pytest

from cubist.agents.strategist import (
    CATEGORIES_USED,
    QUESTION_POOLS,
    Question,
    propose_questions,
)


@pytest.mark.asyncio
async def test_propose_questions_returns_4_distinct_categories():
    qs = await propose_questions(champion_code="x = 1", history=[])

    assert len(qs) == 4
    categories = [q.category for q in qs]
    assert len(set(categories)) == 4
    assert set(categories) == set(CATEGORIES_USED)
    assert all(isinstance(q, Question) for q in qs)
    assert all(len(q.text) >= 20 for q in qs)


@pytest.mark.asyncio
async def test_propose_questions_rotates_with_history_length():
    """Different gen numbers (encoded as len(history)) must hit different
    pool entries. With pools of size >=4, gen 1 and gen 2 should differ."""
    qs1 = await propose_questions(champion_code="x = 1", history=[])
    qs2 = await propose_questions(
        champion_code="x = 1", history=[{"generation": 1}]
    )

    movable = [c for c in CATEGORIES_USED if len(QUESTION_POOLS[c]) > 1]
    assert movable, "expected at least one rotating pool"

    cat_to_text_1 = {q.category: q.text for q in qs1}
    cat_to_text_2 = {q.category: q.text for q in qs2}
    for cat in movable:
        assert cat_to_text_1[cat] != cat_to_text_2[cat], (
            f"category {cat} did not rotate between gen 1 and gen 2"
        )


@pytest.mark.asyncio
async def test_propose_questions_accepts_optional_kwargs():
    """The signature retains the LLM-era kwargs for orchestrator
    compatibility, but they're ignored in the deterministic path."""
    qs_no_extras = await propose_questions(champion_code="x = 1", history=[])
    qs_with_extras = await propose_questions(
        champion_code="x = 1",
        history=[],
        runner_up_code="y = 2",
        champion_question={"category": "search", "text": "old question"},
    )

    assert [q.text for q in qs_no_extras] == [q.text for q in qs_with_extras]


@pytest.mark.asyncio
async def test_propose_questions_index_field_is_unique():
    qs = await propose_questions(champion_code="x = 1", history=[])
    indexes = [q.index for q in qs]
    assert sorted(indexes) == list(range(len(qs)))
