# Engine Core

This package is the shared engine contract used by the tournament runner,
builder validator, and orchestration layer.

## Contract

Every runnable engine exposes a top-level `engine` object that satisfies
`cubist.engines.base.Engine`:

```python
name: str
generation: int
lineage: list[str]

async def select_move(board: chess.Board, time_remaining_ms: int) -> chess.Move
```

`select_move` must return a legal `chess.Move` for `board.turn`. The referee
may adjudicate a loss if an engine raises, times out, or returns an illegal
move, so engine implementations should always have a legal fallback.

## Built-in Engines

- `cubist.engines.baseline`: generation-0 local chess engine. It does not call
  an LLM or any external API. It uses a deterministic two-ply alpha-beta search
  with terminal detection, material, and mobility heuristics.
- `cubist.engines.random_engine`: no-API random legal move engine for tests,
  tournament smoke checks, and builder validation.

## Loading Engines

Use `cubist.engines.registry.load_engine` for both built-ins and generated
candidate files:

```python
from cubist.engines.registry import load_engine

baseline = load_engine("cubist.engines.baseline")
candidate = load_engine("backend/cubist/engines/generated/gen1_prompt.py")
```

Generated candidate modules are expected under `cubist/engines/generated/`.
`list_generated()` returns the sorted `*.py` files in that directory.

## Verification

From `backend/`:

```bash
python -m pytest tests/test_baseline.py tests/test_registry.py -v
```

From the repo root:

```bash
python scripts/smoke_self_play.py
```
