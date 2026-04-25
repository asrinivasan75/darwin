"""Tests for cubist.agents.strategist.

We mock ``cubist.llm.complete`` so the test suite never hits the
Anthropic API. The mock returns a single ``tool_use`` block whose
``input`` matches the real submit_questions schema.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cubist.agents.strategist import CATEGORIES, Question, propose_questions


def _fake_tool_use_block(payload: dict) -> SimpleNamespace:
    """Build a stand-in for an Anthropic ContentBlock of type=tool_use."""
    return SimpleNamespace(type="tool_use", name="submit_questions", input=payload)


def _fake_text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


@pytest.mark.asyncio
async def test_propose_questions_happy_path(monkeypatch):
    """Strategist returns four distinct-category Question records."""
    chosen = CATEGORIES[:4]
    payload = {
        "questions": [
            {"category": cat, "text": f"Make the {cat} better please" * 2}
            for cat in chosen
        ]
    }

    async def fake_complete(**kwargs):
        return [_fake_text_block("Sure thing"), _fake_tool_use_block(payload)]

    monkeypatch.setattr("cubist.agents.strategist.complete", fake_complete)

    qs = await propose_questions(champion_code="x = 1", history=[])

    assert len(qs) == 4
    assert {q.category for q in qs} == set(chosen)
    assert all(isinstance(q, Question) for q in qs)
    assert all(len(q.text) >= 20 for q in qs)


@pytest.mark.asyncio
async def test_propose_questions_dedupes_repeated_category(monkeypatch):
    """Both questions share a category: dedupe collapses to 1, raising."""
    payload = {
        "questions": [
            {"category": "prompt", "text": "first prompt question is fine"},
            {"category": "prompt", "text": "second prompt question is fine"},
        ]
    }

    async def fake_complete(**kwargs):
        return [_fake_tool_use_block(payload)]

    monkeypatch.setattr("cubist.agents.strategist.complete", fake_complete)

    with pytest.raises(ValueError, match="distinct categories"):
        await propose_questions(champion_code="x = 1", history=[])


@pytest.mark.asyncio
async def test_propose_questions_rejects_no_tool_use(monkeypatch):
    """Plain text reply without a tool_use block must surface a clear error."""

    async def fake_complete(**kwargs):
        return [_fake_text_block("I refuse to use the tool.")]

    monkeypatch.setattr("cubist.agents.strategist.complete", fake_complete)

    with pytest.raises(RuntimeError, match="tool_use"):
        await propose_questions(champion_code="x = 1", history=[])


@pytest.mark.asyncio
async def test_propose_questions_passes_history_into_prompt(monkeypatch):
    """The user prompt should include the JSON-encoded history."""
    captured: dict = {}
    payload = {
        "questions": [
            {"category": cat, "text": f"valid question for {cat} category"}
            for cat in CATEGORIES[:4]
        ]
    }

    async def fake_complete(**kwargs):
        captured.update(kwargs)
        return [_fake_tool_use_block(payload)]

    monkeypatch.setattr("cubist.agents.strategist.complete", fake_complete)

    history = [{"generation": 1, "champion": "baseline-v0", "delta": 0.0}]
    await propose_questions(champion_code="x = 1", history=history)

    assert "baseline-v0" in captured["user"]
    assert "delta" in captured["user"]


@pytest.mark.asyncio
async def test_propose_questions_includes_champion_question(monkeypatch):
    """When a champion_question is supplied, both its category and text
    must appear in the user prompt verbatim so the strategist can build on
    top of the improvement that produced the champion."""
    captured: dict = {}
    payload = {
        "questions": [
            {"category": cat, "text": f"valid question for {cat} category"}
            for cat in CATEGORIES[:4]
        ]
    }

    async def fake_complete(**kwargs):
        captured.update(kwargs)
        return [_fake_tool_use_block(payload)]

    monkeypatch.setattr("cubist.agents.strategist.complete", fake_complete)

    cq = {"category": "book", "text": "opening-book lookup hugely-distinctive-marker"}
    await propose_questions(champion_code="x = 1", history=[], champion_question=cq)

    assert "category: book" in captured["user"]
    assert "hugely-distinctive-marker" in captured["user"]


@pytest.mark.asyncio
async def test_propose_questions_baseline_has_no_champion_question(monkeypatch):
    """When champion_question is None (gen 1 / baseline), the prompt must
    still render — the slot is filled with a clear placeholder rather than
    leaving a literal ``{champion_question}`` token in the prompt."""
    captured: dict = {}
    payload = {
        "questions": [
            {"category": cat, "text": f"valid question for {cat} category"}
            for cat in CATEGORIES[:4]
        ]
    }

    async def fake_complete(**kwargs):
        captured.update(kwargs)
        return [_fake_tool_use_block(payload)]

    monkeypatch.setattr("cubist.agents.strategist.complete", fake_complete)

    await propose_questions(champion_code="x = 1", history=[])

    assert "{champion_question}" not in captured["user"]
    assert "baseline" in captured["user"]
