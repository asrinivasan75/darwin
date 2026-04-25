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
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from cubist.config import settings
from cubist.llm import complete

log = logging.getLogger("cubist.strategist")

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


def _validated_questions(qs: list[dict]) -> list[Question]:
    seen: set[str] = set()
    out: list[Question] = []
    for i, q in enumerate(qs):
        cat = str(q.get("category", "")).strip().lower()
        txt = str(q.get("text", "")).strip()
        if cat in seen or cat not in CATEGORIES or not txt:
            continue
        seen.add(cat)
        out.append(Question(index=i, category=cat, text=txt))
    if len(out) != 5:
        raise ValueError(
            f"strategist returned {len(out)} distinct categories, expected 5; "
            f"got categories={[q.get('category') for q in qs]!r}"
        )
    return out


def _json_candidates(text: str) -> list[str]:
    candidates = [text.strip()]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.I)
    )

    first_obj, last_obj = text.find("{"), text.rfind("}")
    if first_obj != -1 and last_obj != -1 and first_obj < last_obj:
        candidates.append(text[first_obj : last_obj + 1])

    first_arr, last_arr = text.find("["), text.rfind("]")
    if first_arr != -1 and last_arr != -1 and first_arr < last_arr:
        candidates.append(text[first_arr : last_arr + 1])

    return candidates


def _questions_from_json_text(text: str) -> list[Question] | None:
    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(data, dict):
            qs = data.get("questions")
        else:
            qs = data

        if isinstance(qs, list):
            return _validated_questions(qs)

    return None


def _questions_from_labeled_text(text: str) -> list[Question] | None:
    heading = re.compile(
        r"(?:^|\n)\s*(?:[-*]|\d+[.)])?\s*(?:[`*_]+)?"
        r"(prompt|search|book|evaluation|sampling)"
        r"(?:[`*_]+)?\s*[:\-–]\s*"
        r"(.*?)(?=(?:\n\s*(?:[-*]|\d+[.)])?\s*(?:[`*_]+)?"
        r"(?:prompt|search|book|evaluation|sampling)(?:[`*_]+)?\s*[:\-–])|\Z)",
        flags=re.DOTALL | re.I,
    )
    qs = [
        {"category": match.group(1).lower(), "text": " ".join(match.group(2).split())}
        for match in heading.finditer(text)
    ]
    if qs:
        return _validated_questions(qs)

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) == 5:
        return _validated_questions(
            [
                {"category": category, "text": re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", paragraph)}
                for category, paragraph in zip(CATEGORIES, paragraphs)
            ]
        )

    return None


def _questions_from_text(text: str) -> list[Question] | None:
    return _questions_from_json_text(text) or _questions_from_labeled_text(text)


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
    user += (
        "\n\nReturn the result by calling the `submit_questions` tool. "
        "If tool calling is unavailable, return a JSON object with a single "
        "`questions` array using this shape: "
        '{"questions":[{"category":"prompt","text":"..."}, ...]}.'
    )
    log.info(
        "strategist start model=%s history=%d champion_chars=%d",
        settings.strategist_model,
        len(history),
        len(champion_code),
    )
    content = await complete(
        model=settings.strategist_model,
        system="You are an expert chess engine designer.",
        user=user,
        max_tokens=2048,
        tools=[TOOL],
    )

    block_kinds = [getattr(b, "type", "?") for b in content]
    log.info("strategist response blocks=%s", block_kinds)

    for block in content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_questions":
            qs = _validated_questions(block.input["questions"])
            log.info("strategist tool_use ok categories=%s", [q.category for q in qs])
            return qs

    text = "\n".join(
        block.text for block in content if getattr(block, "type", None) == "text" and block.text
    )
    log.info("strategist falling back to text parse, text_chars=%d", len(text))
    questions = _questions_from_text(text)
    if questions is not None:
        log.info("strategist text parse ok categories=%s", [q.category for q in questions])
        return questions

    excerpt = text[:200].replace("\n", " ")
    log.error("strategist parse failed excerpt=%r", excerpt)
    raise RuntimeError(f"strategist did not return tool_use or parseable questions: {excerpt!r}")
