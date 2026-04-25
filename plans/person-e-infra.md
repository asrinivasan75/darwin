# Person E — Infra, API, Orchestration & Demo

**Branch:** `feat/infra`

## Goal

Be the integrator. Own the database, the API, the WebSocket bus, the top-level loop that calls everyone else's code, the shared LLM helper, and the demo. The other four people produce parts; you make the machine run.

## Scope

Files owned:

```
backend/darwin/
├── config.py                  # done
├── llm.py                     # basic version done; you can extend (caching, batching)
├── storage/
│   ├── models.py              # FROZEN — do not edit
│   └── db.py                  # session helpers — done; extend as needed
├── api/
│   ├── server.py              # IMPLEMENT — wire routes + WS
│   ├── websocket.py           # FROZEN — do not edit; you'll add an EventBus class
│   └── routes.py              # IMPLEMENT
└── orchestration/
    ├── generation.py          # IMPLEMENT — the big loop
    └── run.py                 # IMPLEMENT — CLI
scripts/
├── seed_baseline.py           # IMPLEMENT
└── run_generation.py          # IMPLEMENT
Procfile                       # done
```

Read first:

1. `docs/proposal.pdf` — the whole document, but especially §11 (Demo Plan).
2. `plans/README.md` — merge order.
3. All four other plans — you integrate against them, so know their shapes.
4. `backend/darwin/storage/models.py` — the **frozen** schema you persist to.
5. `backend/darwin/api/websocket.py` — the **frozen** event payloads you broadcast.
6. `backend/darwin/llm.py` — the shared Anthropic helper (you own this; basic version already works).

Already done for you:

- `config.py` reads `.env` into `Settings` (key, model IDs, time controls).
- `llm.py` provides `complete()` and `complete_text()` with semaphore + retry.
- `storage/models.py` defines `EngineRow`, `GameRow`, `GenerationRow`.
- `storage/db.py` has `init_db()` and `get_session()`.
- `api/websocket.py` defines event payloads (frozen).
- `api/server.py` is a hello-world FastAPI app with `/api/health`.
- `Procfile` runs backend + frontend together via `honcho start`.

## Frozen contracts touched

- **DB schema** — `backend/darwin/storage/models.py`. You write to it; do not modify.
- **WebSocket events** — `backend/darwin/api/websocket.py`. You emit through `bus`; do not modify the event types.

## Deliverables

### Step 1 — Branch and verify

```bash
git checkout -b feat/infra
cd backend && uv sync
uv run python -c "from darwin.api.server import app; print('ok')"
uv run uvicorn darwin.api.server:app --reload  # check /api/health in browser
```

### Step 2 — EventBus + WebSocket route

Add to `backend/darwin/api/websocket.py` (don't edit the frozen types — append):

```python
import asyncio


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def emit(self, event_payload: dict) -> None:
        """Accepts a dict matching one of the Event types. Wraps + broadcasts."""
        envelope = {"event": event_payload}
        for q in list(self._subscribers):
            try:
                q.put_nowait(envelope)
            except asyncio.QueueFull:
                pass


bus = EventBus()
```

Add to `backend/darwin/api/server.py`:
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from darwin.api.websocket import bus
from darwin.api.routes import router

app = FastAPI(title="Darwin")
app.include_router(router, prefix="/api")


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    q = bus.subscribe()
    try:
        while True:
            envelope = await q.get()
            await websocket.send_json(envelope)
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(q)
```

### Step 3 — REST routes

`backend/darwin/api/routes.py`:
```python
import asyncio
from fastapi import APIRouter
from sqlmodel import select

from darwin.storage.db import get_session
from darwin.storage.models import EngineRow, GameRow, GenerationRow

router = APIRouter()


@router.get("/engines")
def list_engines():
    with get_session() as s:
        return s.exec(select(EngineRow)).all()


@router.get("/generations")
def list_generations():
    with get_session() as s:
        return s.exec(select(GenerationRow).order_by(GenerationRow.number)).all()


@router.get("/games")
def list_games(gen: int | None = None):
    with get_session() as s:
        q = select(GameRow)
        if gen is not None:
            q = q.where(GameRow.generation == gen)
        return s.exec(q).all()


@router.post("/generations/run")
async def run():
    from darwin.orchestration.generation import run_generation_task
    asyncio.create_task(run_generation_task())
    return {"started": True}
```

### Step 4 — Seed script

`scripts/seed_baseline.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from darwin.engines.baseline import BaselineEngine  # noqa
from darwin.storage.db import get_session, init_db
from darwin.storage.models import EngineRow
from sqlmodel import select


def main() -> None:
    init_db()
    with get_session() as s:
        existing = s.exec(select(EngineRow).where(EngineRow.name == "baseline-v0")).first()
        if existing:
            print("baseline already seeded:", existing.name); return
        row = EngineRow(name="baseline-v0", generation=0,
                        parent_name=None, code_path="darwin.engines.baseline")
        s.add(row); s.commit()
        print("seeded baseline-v0")


if __name__ == "__main__":
    main()
```

### Step 5 — Orchestration with fakes

The first version of `generation.py` runs against fakes so you can ship the whole loop before A/B/C land. Use Person C's stubs (they push them early).

`backend/darwin/orchestration/generation.py`:
```python
import json

from darwin.agents.builder import build_engine, validate_engine
from darwin.agents.strategist import propose_questions
from darwin.api.websocket import bus
from darwin.config import settings
from darwin.engines.base import Engine
from darwin.engines.registry import load_engine
from darwin.storage.db import get_session
from darwin.storage.models import EngineRow, GameRow, GenerationRow
from darwin.tournament.runner import round_robin
from darwin.tournament.selection import select_champion


def _read_source(engine: Engine) -> str:
    import inspect
    return inspect.getsource(type(engine))


async def run_generation(champion: Engine, generation_number: int) -> Engine:
    await bus.emit({"type": "generation.started",
                    "number": generation_number, "champion": champion.name})

    questions = await propose_questions(_read_source(champion), [])
    for q in questions:
        await bus.emit({"type": "strategist.question",
                        "index": q.index, "category": q.category, "text": q.text})

    import asyncio
    paths = await asyncio.gather(
        *[build_engine(_read_source(champion), champion.name, generation_number, q) for q in questions],
        return_exceptions=True,
    )

    candidates: list[Engine] = []
    for q, p in zip(questions, paths):
        if isinstance(p, Exception):
            await bus.emit({"type": "builder.completed", "question_index": q.index,
                            "engine_name": "-", "ok": False, "error": str(p)})
            continue
        ok, err = await validate_engine(p)
        name = p.stem
        await bus.emit({"type": "builder.completed", "question_index": q.index,
                        "engine_name": name, "ok": ok, "error": err})
        if ok:
            candidates.append(load_engine(str(p)))

    standings = await round_robin(
        [champion, *candidates],
        games_per_pairing=settings.games_per_pairing,
        time_per_move_ms=settings.time_per_move_ms,
        on_event=bus.emit,
    )
    new_champion, promoted = select_champion(standings, champion, candidates)

    with get_session() as s:
        gen_row = GenerationRow(
            number=generation_number, champion_before=champion.name,
            champion_after=new_champion.name,
            strategist_questions_json=json.dumps([{"category": q.category, "text": q.text} for q in questions]),
        )
        s.add(gen_row)
        for g in standings.games:
            s.add(GameRow(generation=generation_number, white_name=g.white,
                          black_name=g.black, pgn=g.pgn, result=g.result, termination=g.termination))
        s.commit()

    await bus.emit({"type": "generation.finished", "number": generation_number,
                    "new_champion": new_champion.name, "elo_delta": 0.0, "promoted": promoted})
    return new_champion


async def run_generation_task() -> None:
    """Triggered by the API. Loads current champion from DB, runs one generation."""
    # TODO: load champion from DB; for now, baseline
    from darwin.engines.baseline import engine as baseline
    await run_generation(baseline, 1)
```

### Step 6 — CLI runner

`backend/darwin/orchestration/run.py`:
```python
import argparse
import asyncio

from darwin.engines.baseline import engine as baseline
from darwin.orchestration.generation import run_generation
from darwin.storage.db import init_db


async def main(generations: int) -> None:
    init_db()
    champion = baseline
    for n in range(1, generations + 1):
        champion = await run_generation(champion, n)
    print(f"final champion: {champion.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(main(args.generations))
```

### Step 7 — End-to-end demo run (after A/B/C merge)

```bash
uv run python scripts/seed_baseline.py
honcho start  # runs backend + frontend together
# in another terminal:
uv run python -m darwin.orchestration.run --generations 1
```

Verify: dashboard shows the full loop streaming live.

### Step 8 — Replay mode for demo safety

After a clean generation completes, snapshot the DB. Add `scripts/replay.py` that reads `GameRow`/`GenerationRow` and re-emits the WS events with realistic delays. This is your insurance against demo-day API flakes — **build this before demo prep starts**.

### Step 9 — Open the PR

```bash
git add -A && git commit -m "feat: API + WS bus + orchestration + seed/replay scripts"
git push -u origin feat/infra
gh pr create --title "Infra + orchestration" --body "Closes plan E."
```

### Integration points

- **Person A** provides `BaselineEngine`, `Engine` Protocol, `load_engine`. You instantiate baseline at seed time.
- **Person B** provides `round_robin`, `select_champion`, `play_game`. Pass `bus.emit` as `on_event`.
- **Person C** provides `propose_questions`, `build_engine`, `validate_engine`. They ship stubs early; you can wire end-to-end immediately.
- **Person D** consumes `/ws`. Validate event payloads against their `events.ts` mirror as soon as you start.

## Acceptance criteria

- [ ] `/api/health`, `/api/engines`, `/api/generations`, `/api/games`, `/ws` all respond.
- [ ] `seed_baseline.py` is idempotent and inserts the baseline row.
- [ ] One full generation completes via `run.py` and persists to DB.
- [ ] `honcho start` brings up backend + frontend with WS events flowing.
- [ ] Replay mode works without API access (demo insurance).
- [ ] PR opened, then merged after review.

## Risks & mitigations

- **You are the rate-limit choke point.** The semaphore in `llm.py` (currently 30) controls everyone. Tune it down if you see 429s, up if the tournament drags.
- **WS backpressure.** Slow frontend → queue fills → events drop. The `put_nowait` + drop policy is intentional.
- **Crash recovery.** If a generation crashes mid-tournament, a `GenerationRow` with `finished_at=NULL` is left behind. For the hackathon, just resume from the next generation number.
- **Demo day.** Step 8 (replay mode) is the **single most important risk mitigation**. Do not skip it.

## Status

Merged. Note: post-hackathon `run_generation_task` was extended to load the reigning champion + runner-up from the DB instead of always restarting from baseline (see [followup-3-champion-resume.md](followup-3-champion-resume.md)), and the orchestrator gained cancel/replace semantics for the Run/Stop buttons.
