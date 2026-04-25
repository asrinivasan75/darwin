"""Person C — strategist agent.

Calls the strategist model (default ``claude-opus-4-6``) once per
generation with a ``submit_questions`` tool. The model must return
exactly one improvement question per category in
``CATEGORIES``; we deduplicate on category and raise if fewer than 5
distinct categories come back.

The free-form ``text`` of each question is shown verbatim in the
dashboard — keep it readable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cubist.config import settings
from cubist.llm import complete

CATEGORIES = ["prompt", "search", "book", "evaluation", "sampling"]

PROMPT = (Path(__file__).parent / "prompts" / "strategist_v1.md").read_text()

TOOL = {
    "name": "submit_questions",
    "description": (
        "Submit exactly 5 improvement questions, one per category from "
        "[prompt, search, book, evaluation, sampling]. Each question's text "
        "is shown verbatim to a human, so write plain English."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": CATEGORIES},
                        "text": {"type": "string", "minLength": 20},
                    },
                    "required": ["category", "text"],
                },
            }
        },
        "required": ["questions"],
    },
}


@dataclass
class Question:
    index: int
    category: str
    text: str


async def propose_questions(
    champion_code: str,
    history: list[dict],
) -> list[Question]:
    """Return 5 distinct improvement questions across the locked categories.

    Args:
        champion_code: source of the current champion module.
        history: prior-generation summaries (each generation: champion name,
            wins/losses, accepted question category, etc.). Empty on
            generation 1; otherwise the strategist may use it to ground its
            rationale.

    Returns:
        Length-5 list of ``Question`` records, deduplicated by category.

    Raises:
        ValueError: if the model returns fewer than 5 distinct categories.
        RuntimeError: if the model never produced a ``tool_use`` block.
    """
    user = PROMPT.format(
        champion_code=champion_code,
        history_json=json.dumps(history, indent=2),
    )
    content = await complete(
        model=settings.strategist_model,
        system="You are an expert chess engine designer.",
        user=user,
        max_tokens=2048,
        tools=[TOOL],
    )

    for block in content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_questions":
            qs = block.input["questions"]
            seen: set[str] = set()
            out: list[Question] = []
            for i, q in enumerate(qs):
                cat = q.get("category", "")
                txt = q.get("text", "").strip()
                if cat in seen or cat not in CATEGORIES or not txt:
                    continue
                seen.add(cat)
                out.append(Question(index=i, category=cat, text=txt))
            if len(out) != 5:
                raise ValueError(
                    f"strategist returned {len(out)} distinct categories, expected 5; "
                    f"got categories={[q['category'] for q in qs]!r}"
                )
            return out

    raise RuntimeError("strategist did not return tool_use")
