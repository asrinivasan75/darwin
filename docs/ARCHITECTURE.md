# Architecture

One generation of Darwin: an LLM proposes improvements, parallel LLMs build them, a validator gates them, a tournament ranks them, the top two advance. The dashboard streams every event live.

## Components

| Layer | Module | Responsibility |
|---|---|---|
| LLM client | [backend/darwin/llm.py](../backend/darwin/llm.py) | Single choke-point for Anthropic + Gemini; semaphore + retry. |
| Engines | [backend/darwin/engines/](../backend/darwin/engines/) | `Engine` Protocol, baseline, random, dynamic loader for `generated/`. |
| Agents | [backend/darwin/agents/](../backend/darwin/agents/) | Strategist (proposes 2 questions), builder (writes a candidate engine). |
| Tournament | [backend/darwin/tournament/](../backend/darwin/tournament/) | Referee (one game), runner (round-robin), Elo, top-N selection. |
| Orchestration | [backend/darwin/orchestration/generation.py](../backend/darwin/orchestration/generation.py) | The big loop wiring the pieces together. |
| API | [backend/darwin/api/](../backend/darwin/api/) | FastAPI REST + `/ws` WebSocket fanout. |
| Storage | [backend/darwin/storage/](../backend/darwin/storage/) | SQLite via SQLModel: engines, games, generations. |
| Frontend | [frontend/src/](../frontend/src/) | React + Vite dashboard subscribing to `/ws`. |

## Data flow per generation

```
                 ┌─────────────┐
                 │  champion   │  (loaded from DB or baseline)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐  emits strategist.question ×2
                 │  strategist │
                 └──────┬──────┘
                        ▼
              ┌─────────┴─────────┐  parallel asyncio.gather
              ▼                   ▼
         ┌─────────┐         ┌─────────┐
         │ builder │   …     │ builder │   emits builder.completed
         └────┬────┘         └────┬────┘
              ▼                   ▼
            validator (smoke game vs RandomEngine)
                        │
                        ▼
              round-robin tournament   emits game.move / game.finished
                        │
                        ▼
              top-N selection (win rate, random tiebreak)
                        │
                        ▼
              persist EngineRow / GameRow / GenerationRow
                        │
                        ▼
              emits generation.finished (ratings, promoted)
```

The cohort each tournament is `[champion, runner_up, *accepted_candidates]`. Top-2 by win rate seed the next generation, so a runner-up persists across generations and the line of descent doesn't collapse.

## Frozen contracts

These four files are the parallelization seams between workstreams. Changes require team sign-off.

- **Engine Protocol** — [backend/darwin/engines/base.py](../backend/darwin/engines/base.py)
- **DB schema** — [backend/darwin/storage/models.py](../backend/darwin/storage/models.py)
- **WebSocket events (backend)** — [backend/darwin/api/websocket.py](../backend/darwin/api/websocket.py)
- **WebSocket events (frontend mirror)** — [frontend/src/api/events.ts](../frontend/src/api/events.ts)

## Storage

Single-machine SQLite (`backend/darwin.db`), three tables: `engines`, `games`, `generations`. Schema in [models.py](../backend/darwin/storage/models.py). `EngineRow.code_path` is either a dotted module name (baseline) or an absolute filesystem path (generated). `make replay` re-emits the persisted event stream — designed as a demo safety net.

## Event bus

`/ws` clients subscribe to a per-connection bounded `asyncio.Queue` (1000). Backpressure policy: a slow client drops events for itself only — never blocks the orchestrator. Events are typed in [websocket.py](../backend/darwin/api/websocket.py); the frontend re-declares them in [events.ts](../frontend/src/api/events.ts) and discriminates on `event.type`.

## Optional Modal backend

`TOURNAMENT_BACKEND=modal` dispatches each game to its own Modal container — real OS-level parallel, no GIL. The runner pre-warms 40 containers while the strategist + builders are working, drains a shared `modal.Queue` of move events back onto our bus, and synthesizes a draw + `termination=error` when a container times out so the rest of the cohort still scores fairly. Falls back to `local` automatically on any Modal-side error. See [tournament/runner.py](../backend/darwin/tournament/runner.py) and [tournament/modal_runner.py](../backend/darwin/tournament/modal_runner.py).
