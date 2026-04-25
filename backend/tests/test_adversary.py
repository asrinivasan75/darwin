"""Tests for darwin.agents.adversary.

The LLM call is mocked. Empty-string return on any failure is a load-
bearing contract (the orchestrator skips the fixer when critique is
empty), so it gets dedicated coverage.
"""

from __future__ import annotations

import pytest

from darwin.agents import adversary
from darwin.agents.adversary import critique_engine
from darwin.agents.strategist import Question


@pytest.fixture
def question() -> Question:
    return Question(
        index=0,
        category="quiescence",
        text="Add capture-only quiescence search to depth 4.",
    )


@pytest.mark.asyncio
async def test_returns_critique_text_on_success(monkeypatch, question):
    captured: dict = {}

    async def fake_complete_text(model, system, user, max_tokens=256, provider=None):
        captured["model"] = model
        captured["provider"] = provider
        captured["user"] = user
        return (
            "Quiescence has no depth bound — at high capture density this "
            "will exceed the 5s budget and forfeit on time. Cap the recursion "
            "at depth 4. The eval also has a sign error in the negamax return."
        )

    monkeypatch.setattr(adversary, "complete_text", fake_complete_text)

    text = await critique_engine(question, "code goes here", "gen1-quiescence-abc123")

    assert text.startswith("Quiescence has no depth bound")
    # Prompt must include the question text and code so the model can ground its review.
    assert "capture-only quiescence" in captured["user"]
    assert "code goes here" in captured["user"]


@pytest.mark.asyncio
async def test_returns_empty_string_on_llm_failure(monkeypatch, question):
    async def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(adversary, "complete_text", boom)

    text = await critique_engine(question, "code", "gen1-quiescence-abc123")
    assert text == ""


@pytest.mark.asyncio
async def test_returns_empty_string_on_short_response(monkeypatch, question):
    async def fake(*args, **kwargs):
        return "ok"

    monkeypatch.setattr(adversary, "complete_text", fake)

    text = await critique_engine(question, "code", "gen1-quiescence-abc123")
    assert text == ""


@pytest.mark.asyncio
async def test_uses_adversary_provider_override(monkeypatch, question):
    """Provider routing for adversary must come from settings.adversary_provider."""
    from darwin.config import settings

    monkeypatch.setattr(settings, "adversary_provider", "gemini")

    captured: dict = {}

    async def fake(model, system, user, max_tokens=256, provider=None):
        captured["provider"] = provider
        return "A long enough critique to clear the 20-char minimum threshold easily."

    monkeypatch.setattr(adversary, "complete_text", fake)

    await critique_engine(question, "code", "gen1-quiescence-abc123")
    assert captured["provider"] == "gemini"
