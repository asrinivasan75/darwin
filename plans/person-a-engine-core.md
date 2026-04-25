# Person A — Engine Core & Baseline

**Branch:** `feat/engine-core`

## Goal

Own the abstraction every other workstream depends on: get the Engine contract right and ship a working baseline early. Everyone else is blocked on you to test their integration paths.

## Scope

Files owned:

```
backend/darwin/engines/
├── base.py              # FROZEN — Engine Protocol + BaseLLMEngine (done)
├── baseline.py          # generation-0 LLM engine — IMPLEMENT
├── random_engine.py     # already complete; you maintain it
├── registry.py          # dynamic loader — IMPLEMENT
└── generated/           # builder output lands here (gitignored)
backend/tests/test_baseline.py   # CREATE
backend/tests/test_registry.py   # CREATE
```

Read first:

1. `docs/proposal.pdf` — the whole project, 9 pages.
2. `plans/README.md` — the merge order and frozen contracts.
3. `backend/darwin/engines/base.py` — the Engine Protocol you'll build against (already complete).
4. `backend/darwin/llm.py` — the shared Anthropic helper you'll call (already complete).

Already done for you:

- `base.py`: Engine Protocol + `BaseLLMEngine` helper class.
- `random_engine.py`: `RandomEngine` — picks a random legal move.
- `llm.py`: `complete()` and `complete_text()` async helpers around Anthropic with semaphore + retry.
- `baseline.py` and `registry.py` have stub signatures with `NotImplementedError`.

## Frozen contracts touched

- **Engine Protocol** — `backend/darwin/engines/base.py`. You consume it; do not modify after it lands.

## Deliverables

### Step 1 — Branch and verify the env

```bash
git checkout -b feat/engine-core
cd backend && uv sync
cp ../.env.example ../.env  # add your ANTHROPIC_API_KEY
uv run python -c "from darwin.engines.base import Engine; print('ok')"
uv run python -c "from darwin.llm import complete_text; print('ok')"
```

### Step 2 — Implement `BaselineEngine.select_move`

Edit `backend/darwin/engines/baseline.py`. Use this exact structure:

```python
import chess
from darwin.config import settings
from darwin.engines.base import BaseLLMEngine
from darwin.llm import complete_text

SYSTEM = (
    "You are a chess engine. Reply with EXACTLY ONE legal move in standard "
    "algebraic notation (SAN). No prose, no explanation, just the move."
)

class BaselineEngine(BaseLLMEngine):
    def __init__(self) -> None:
        super().__init__(name="baseline-v0", generation=0, lineage=[])

    async def select_move(self, board: chess.Board, time_remaining_ms: int) -> chess.Move:
        legal = [board.san(m) for m in board.legal_moves]
        user = (
            f"FEN: {board.fen()}\n"
            f"Move number: {board.fullmove_number}\n"
            f"Side to move: {'White' if board.turn else 'Black'}\n"
            f"Legal moves: {', '.join(legal)}\n"
            f"Your move:"
        )
        text = await complete_text(settings.player_model, SYSTEM, user, max_tokens=10)
        san = text.strip().split()[0] if text.strip() else ""
        try:
            return board.parse_san(san)
        except (ValueError, IndexError):
            return next(iter(board.legal_moves))  # fallback: first legal

engine = BaselineEngine()
```

Verify by running:
```bash
uv run python -c "
import asyncio, chess
from darwin.engines.baseline import engine
b = chess.Board()
m = asyncio.run(engine.select_move(b, 10000))
print('move:', b.san(m))
"
```

### Step 3 — Implement `registry.load_engine`

Edit `backend/darwin/engines/registry.py`:

```python
import importlib
import importlib.util
from pathlib import Path

from darwin.engines.base import Engine

GENERATED_DIR = Path(__file__).parent / "generated"


def load_engine(module_path: str) -> Engine:
    if module_path.endswith(".py") or "/" in module_path:
        path = Path(module_path)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {module_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(module_path)

    eng = getattr(mod, "engine", None)
    if eng is None:
        raise AttributeError(f"{module_path} has no top-level `engine` symbol")
    if not isinstance(eng, Engine):  # runtime_checkable Protocol check
        raise TypeError(f"{module_path}.engine does not satisfy Engine Protocol")
    return eng


def list_generated() -> list[Path]:
    return sorted(GENERATED_DIR.glob("*.py"))
```

### Step 4 — Write tests

`backend/tests/test_registry.py`:
```python
import tempfile
from pathlib import Path
from darwin.engines.registry import load_engine

def test_loads_baseline_by_dotted_path():
    eng = load_engine("darwin.engines.baseline")
    assert eng.name == "baseline-v0"

def test_loads_random_by_dotted_path():
    eng = load_engine("darwin.engines.random_engine")
    assert eng.name == "random"

def test_loads_from_file():
    src = '''
from darwin.engines.random_engine import RandomEngine
engine = RandomEngine()
'''
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(src); f.flush()
        eng = load_engine(f.name)
    assert eng.name == "random"

def test_rejects_module_without_engine_symbol(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("x = 1")
    import pytest
    with pytest.raises(AttributeError):
        load_engine(str(bad))
```

`backend/tests/test_baseline.py`:
```python
import chess
import pytest
from darwin.engines.baseline import engine

@pytest.mark.asyncio
async def test_baseline_returns_legal_move():
    board = chess.Board()
    move = await engine.select_move(board, 10000)
    assert move in board.legal_moves
```

Run them: `uv run pytest tests/ -v`.

### Step 5 — End-to-end self-play smoke test

Add `scripts/smoke_self_play.py`:
```python
import asyncio, chess
from darwin.engines.baseline import engine

async def main():
    board = chess.Board()
    while not board.is_game_over() and board.fullmove_number < 10:
        m = await engine.select_move(board, 10000)
        print(board.san(m))
        board.push(m)
    print("done:", board.result())

asyncio.run(main())
```

Run it: `uv run python scripts/smoke_self_play.py`. If 10 moves complete, you're done.

### Step 6 — Open the PR

```bash
git add -A && git commit -m "feat: baseline LLM engine + registry"
git push -u origin feat/engine-core
gh pr create --title "Engine core + baseline" --body "Closes plan A. Unblocks B and C."
```

Page Person B and C in chat: "engine-core merged."

### Integration points

- **Person B** imports `Engine` and calls `select_move` from `referee.py`. Also uses `RandomEngine` in tests.
- **Person C**'s builder writes modules to `generated/`; **you** load them via `registry.load_engine`. Also uses `RandomEngine` in the validator.
- **Person E** seeds the DB with the baseline by calling `BaselineEngine()` and storing its module dotted path.

## Acceptance criteria

- [ ] Step 2 verification command prints a legal move.
- [ ] All four tests in Step 4 pass.
- [ ] `smoke_self_play.py` plays 10 moves without crashing.
- [ ] PR opened, then merged after review.

## Risks & mitigations

- **Don't change `base.py` once it's landed.** Other people are mocking against it. If you absolutely need to, page the team.
- **Illegal-move parsing.** Always have a fallback (the `next(iter(board.legal_moves))` line) so a malformed LLM response doesn't crash a game.
- **Don't import `anthropic` directly.** Always go through `darwin.llm` so we have one rate-limit choke point.

## Status

Merged.
