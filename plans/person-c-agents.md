# Person C — Agents (Strategist + Builder)

**Branch:** `feat/agents`

## Goal

Build the brain of the self-improvement loop: the agent that decides *what to try* and the agent that *writes the code*. The demo's wow-factor comes from the strategist's reasoning being legible — make it good prose, not just JSON.

## Scope

Files owned:

```
backend/darwin/agents/
├── strategist.py              # IMPLEMENT
├── builder.py                 # IMPLEMENT
└── prompts/
    ├── strategist_v1.md       # CREATE
    └── builder_v1.md          # CREATE
backend/tests/test_strategist.py   # CREATE
backend/tests/test_builder.py      # CREATE
```

Read first:

1. `docs/proposal.pdf` — focus on §3.1 (Three Roles) and §9 (Risks).
2. `plans/README.md` — merge order and frozen contracts.
3. `backend/darwin/engines/base.py` — the Protocol that builder output must satisfy.
4. `backend/darwin/llm.py` — the shared Anthropic helper. Use `complete()` (with `tools=`) for both agents.
5. `backend/darwin/engines/random_engine.py` — your validator's sparring partner.
6. `backend/darwin/engines/baseline.py` — read this once Person A merges it; this is what your builder will be modifying.

Already done for you:

- `strategist.py` and `builder.py` have stubs with the right function signatures and `Question` dataclass.
- `darwin.llm.complete()` handles auth, semaphore, retry, and tool-use. Pass it `tools=[...]` and read the `tool_use` block from the returned `content`.
- `RandomEngine` exists for your validator's smoke games.

## Frozen contracts touched

- **Engine Protocol** — `backend/darwin/engines/base.py`. Builder output must satisfy it; do not modify.

## Deliverables

### Step 1 — Branch and verify

```bash
git checkout -b feat/agents
cd backend && uv sync
uv run python -c "from darwin.llm import complete; print('ok')"
```

### Step 2 — Stub responses to unblock Person E

E starts integrating with you immediately. Make `propose_questions` and `build_engine` return hardcoded values that satisfy the contract, so E can wire orchestration end-to-end against fakes.

In `strategist.py`:
```python
async def propose_questions(champion_code: str, history: list[dict]) -> list[Question]:
    cats = ["prompt", "search", "book", "evaluation", "sampling"]
    return [
        Question(i, c, f"[STUB] Try a {c}-based improvement.")
        for i, c in enumerate(cats)
    ]
```

In `builder.py` — for now just write a copy of the baseline with a new name:
```python
async def build_engine(champion_code, champion_name, generation, question):
    name = f"gen{generation}-{question.category}-stub{question.index}"
    path = Path(__file__).parent.parent / "engines" / "generated" / f"{name}.py"
    path.write_text(champion_code.replace('"baseline-v0"', f'"{name}"'))
    return path

async def validate_engine(module_path):
    return True, None
```

Commit and push: "feat(agents): stub implementations for parallel dev." Tell Person E you've shipped stubs.

### Step 3 — Real strategist with tool-use

`backend/darwin/agents/prompts/strategist_v1.md`:
```
You are the strategist for a self-improving chess engine. Below is the current
champion's source code and a history of prior generations.

Your job: propose exactly 2 distinct improvement questions. Each question
must target a DIFFERENT category from this fixed list:
  - prompt:     change how the LLM is asked for moves
  - search:     wrap the LLM in a lookahead / minimax / MCTS layer
  - book:       opening-book or endgame-tablebase lookup
  - evaluation: have the LLM score positions before choosing
  - sampling:   draw multiple candidate moves, pick by vote / best-eval

For each question, give:
  - a concrete hypothesis a builder can implement in ~50 lines of Python
  - a one-sentence rationale grounded in the history (or first principles)

CURRENT CHAMPION SOURCE:
{champion_code}

HISTORY (prior generations, JSON):
{history_json}
```

`backend/darwin/agents/strategist.py`:
```python
import json
from dataclasses import dataclass
from pathlib import Path

from darwin.config import settings
from darwin.llm import complete

PROMPT = (Path(__file__).parent / "prompts" / "strategist_v1.md").read_text()
CATEGORIES = ["prompt", "search", "book", "evaluation", "sampling"]

TOOL = {
    "name": "submit_questions",
    "description": "Submit exactly 2 improvement questions, each from a different category.",
    "input_schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 2, "maxItems": 2,
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


async def propose_questions(champion_code: str, history: list[dict]) -> list[Question]:
    user = PROMPT.format(champion_code=champion_code, history_json=json.dumps(history, indent=2))
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
            seen = set()
            out: list[Question] = []
            for i, q in enumerate(qs):
                if q["category"] in seen:
                    continue
                seen.add(q["category"])
                out.append(Question(index=i, category=q["category"], text=q["text"]))
            if len(out) != 2:
                raise ValueError(f"expected 2 distinct categories, got {len(out)}")
            return out
    raise RuntimeError("strategist did not return tool_use")
```

### Step 4 — Real builder with tool-use

`backend/darwin/agents/prompts/builder_v1.md`:
```
You are a chess engine builder. Modify the champion below to answer ONE
specific improvement question.

QUESTION (category={category}):
{question_text}

CHAMPION SOURCE:
{champion_code}

REQUIREMENTS:
- Subclass `BaseLLMEngine` from `darwin.engines.base`.
- The class __init__ must call super().__init__(name="{engine_name}",
  generation={generation}, lineage=["{champion_name}"]).
- Implement `async def select_move(self, board, time_remaining_ms)`.
- Module must end with: `engine = YourEngineClass()`.
- Stay under 100 lines.
- Allowed imports ONLY: stdlib, `chess`, `darwin.config`, `darwin.engines.base`,
  `darwin.llm`. NO subprocess, os.system, socket, or eval.
- Always have a fallback to a legal move so the engine never crashes a game.
```

`backend/darwin/agents/builder.py`:
```python
import hashlib
import re
from pathlib import Path

from darwin.config import settings
from darwin.engines.registry import GENERATED_DIR
from darwin.llm import complete
from darwin.agents.strategist import Question

PROMPT = (Path(__file__).parent / "prompts" / "builder_v1.md").read_text()

TOOL = {
    "name": "submit_engine",
    "description": "Submit the new engine module.",
    "input_schema": {
        "type": "object",
        "properties": {"code": {"type": "string", "minLength": 100}},
        "required": ["code"],
    },
}

FORBIDDEN = re.compile(r"\b(subprocess|os\.system|socket|eval|exec)\b")


async def build_engine(champion_code, champion_name, generation, question) -> Path:
    short = hashlib.sha1(question.text.encode()).hexdigest()[:6]
    engine_name = f"gen{generation}-{question.category}-{short}"
    user = PROMPT.format(
        category=question.category, question_text=question.text,
        champion_code=champion_code, engine_name=engine_name,
        generation=generation, champion_name=champion_name,
    )
    content = await complete(
        model=settings.builder_model,
        system="You write Python chess engines.",
        user=user, max_tokens=4096, tools=[TOOL],
    )
    for block in content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_engine":
            code = block.input["code"]
            if FORBIDDEN.search(code):
                raise ValueError("builder code uses forbidden imports")
            path = GENERATED_DIR / f"{engine_name.replace('-', '_')}.py"
            path.write_text(code)
            return path
    raise RuntimeError("builder did not return tool_use")


async def validate_engine(module_path: Path) -> tuple[bool, str | None]:
    """Smoke-test: load + play one short game vs random."""
    from darwin.engines.registry import load_engine
    from darwin.engines.random_engine import RandomEngine
    from darwin.tournament.referee import play_game

    try:
        eng = load_engine(str(module_path))
    except Exception as e:
        return False, f"load: {e!r}"

    try:
        opp = RandomEngine(seed=0); opp.name = "validator-opp"
        # short game cap: rely on max_moves_per_game in settings, but force quick eval
        result = await play_game(eng, opp, time_per_move_ms=10000, game_id=-1)
        if result.termination == "error":
            return False, "engine crashed during smoke game"
    except Exception as e:
        return False, f"play: {e!r}"
    return True, None
```

### Step 5 — Tests

`tests/test_strategist.py` — mock `darwin.llm.complete` with `monkeypatch`, return a fake tool-use response, assert 2 distinct categories.

`tests/test_builder.py` — same pattern; verify file is written, FORBIDDEN regex rejects bad code, validator rejects a syntax-error module.

### Step 6 — Open the PR

```bash
git add -A && git commit -m "feat: strategist + builder agents with tool-use"
git push -u origin feat/agents
gh pr create --title "Agents" --body "Closes plan C."
```

### Integration points

- **Person A**'s `BaseLLMEngine` is what your builder's output subclasses. Read it before writing the prompt.
- **Person A**'s `registry.load_engine` is what your validator calls.
- **Person B**'s `play_game` is what your validator's smoke game uses.
- **Person E** calls `propose_questions` once per generation, then `gather(*build_engine(...) for q in questions)`.

## Acceptance criteria

- [ ] Stubs (Step 2) pushed to a branch quickly so Person E unblocks.
- [ ] Real strategist returns 2 distinct categories from a real Opus call.
- [ ] Real builder produces an engine that validates and beats `RandomEngine` >50% in a 4-game match.
- [ ] PR opened, then merged after review.

## Risks & mitigations

- **JSON schema enforcement.** Free-form text output will bite you. The `tool_use` pattern with `tools=[TOOL]` is mandatory.
- **Builder hallucination.** The prompt MUST explicitly list allowed imports; the FORBIDDEN regex is a backstop.
- **Sandbox concerns.** Builder code runs in our process. The regex check is the minimum bar for a hackathon.
- **Cost control.** Strategist is Opus and gets a long champion source. If you have time, switch to Anthropic's prompt caching (mark the SYSTEM prompt as cacheable).
- **Make questions readable.** Person D shows them verbatim. No JSON-ese in the user-facing text.

## Status

Merged. Note: post-hackathon the builder prompt + `FORBIDDEN` regex were tightened to reject the broken `from darwin import config as settings` pattern (see [followup-2-builder-quality.md](followup-2-builder-quality.md)).
