# Darwin

<img src="docs/darwin-logo.png" alt="Darwin logo" width="180" align="right" />

A self-improving chess engine that evolves its own scaffolding via agentic tournament selection.

> Built at the Point72 Hackathon as a 24-hour, five-engineer build.

---

## Overview

Darwin asks an LLM to write a chess engine, then asks another LLM how to make it better, then asks four more LLMs (one per improvement category) to act on those suggestions, then plays the resulting candidates in a round-robin to decide who survives. Each pass through that pipeline is a **generation**. The next generation starts from the previous generation's winner.

Each generation runs four roles:

1. **Strategist** — looks at the reigning champion's source code and proposes concrete, distinct improvement directions (e.g. "add quiescence search", "add a transposition table", "weight the king-safety term more heavily").
2. **Builders (in parallel)** — each builder agent receives one strategist question, the champion source, and a chess library reference, and emits a complete Python module that satisfies the `Engine` Protocol.
3. **Validator** — static-source gates (forbidden imports, hallucinated `chess.X` attributes, missing `select_move`, etc.) plus a smoke game vs a `RandomEngine`. Anything that crashes, times out, or returns illegal moves never reaches the tournament.
4. **Tournament** — a round-robin of the surviving candidates plus the previous champion and its runner-up. Pairings play two games (one per color). Tournament scores decide who advances; Elo is tracked for the dashboard but does not gate selection.

The promotion rule is tournament-wide **win rate** (score / games_played) with random tiebreak. An earlier version required the top scorer to also beat the prior champion head-to-head, but with only two games per pair that gate's variance was high enough to lock the demo on baseline; cohort-wide win rate is a more stable signal.

The dashboard streams every move, every strategist question, every bracket result, and every Elo update over a WebSocket so you can watch a generation unfold in real time.

A second branch — written up in [`docs/experiment-pure-code.md`](./docs/experiment-pure-code.md) — flips the design from "LLM picks moves at play time" to "LLM writes a classical alpha-beta engine that plays without any LLM at runtime." That branch is local-only and is not on `main`.

---

## How to run

The repo ships a `Makefile` with all the common targets — run `make help` to list them.

### First-time setup

```bash
make install              # uv sync backend + npm install frontend
cp .env.example .env      # then fill in ANTHROPIC_API_KEY (or GOOGLE_API_KEY)
uv tool install honcho    # only if you plan to use `make dev`
make seed                 # initialize the DB and insert baseline-v0 (idempotent)
```

### Picking an LLM provider

Set `LLM_PROVIDER=claude` (default) or `LLM_PROVIDER=gemini` in `.env`. Claude uses `ANTHROPIC_API_KEY`; Gemini uses `GOOGLE_API_KEY`. When switching, also update `STRATEGIST_MODEL`, `PLAYER_MODEL`, and `BUILDER_MODEL` to provider-appropriate values (e.g. `gemini-2.5-pro`). The strategist and builder agents both use function-calling, which is supported on both providers — call sites are unchanged.

### Running the dashboard

```bash
make dev                  # backend (:8000) + frontend (:5173) together via honcho
# or separately, in two terminals:
make backend
make frontend
```

Then open [http://localhost:5173](http://localhost:5173) and click **Run Generation**.

### Triggering a generation from the CLI

```bash
curl -X POST http://localhost:8000/api/generations/run   # via the API
make run                                                 # one generation
make run N=3                                             # three back-to-back
```

### Other useful targets

```bash
make test                                       # pytest
make check                                      # lint + tests (pre-PR gate)
make smoke                                      # 10-move baseline self-play
make eval WHITE=baseline-v0 BLACK=random N=10   # head-to-head match
make replay                                     # re-emit persisted gens over WS
make reset                                      # drop the DB and re-seed
```

---

## Repository layout

```
backend/      Python: engines, agents, tournament, API, orchestration
frontend/     React + Vite dashboard
scripts/      One-off CLIs (run a generation, eval matches, smoke, replay)
plans/        Per-person build plans for the 5-engineer team
docs/         Reference docs (proposal, architecture)
```

### Frozen contracts

These define the interfaces between the workstreams — change them only with team sign-off:

- **Engine Protocol** — [backend/darwin/engines/base.py](backend/darwin/engines/base.py)
- **DB schema** — [backend/darwin/storage/models.py](backend/darwin/storage/models.py)
- **WebSocket events** — [backend/darwin/api/websocket.py](backend/darwin/api/websocket.py) and [frontend/src/api/events.ts](frontend/src/api/events.ts)

---

## Dependencies

### Runtime

| Component | Purpose |
|---|---|
| Python ≥ 3.11 | Backend runtime |
| [`python-chess`](https://python-chess.readthedocs.io/) | Board state, move generation, SAN parsing |
| [`anthropic`](https://docs.anthropic.com/) | Claude SDK (default LLM provider) |
| [`google-genai`](https://ai.google.dev/) | Gemini SDK (alternate provider) |
| `fastapi` + `uvicorn` | HTTP API + WebSocket transport |
| `sqlmodel` | Persistence over SQLite |
| `pydantic` / `pydantic-settings` | Config + payload validation |
| `modal` | Optional remote tournament backend (one container per game) |
| Node ≥ 18 | Frontend toolchain |
| React 18 + Vite + TypeScript | Dashboard |
| `react-chessboard` + `chess.js` | Live board rendering |
| `recharts` | Elo charts |
| `tailwindcss` | Styling |

### Tooling

- [`uv`](https://docs.astral.sh/uv/) — Python dependency + virtualenv manager (`make install` shells out to it)
- `honcho` — runs backend + frontend together for `make dev`
- `ruff` — lint + format
- `pytest` + `pytest-asyncio` — backend tests

### Required configuration

At minimum you need an `ANTHROPIC_API_KEY` (or `GOOGLE_API_KEY` if you set `LLM_PROVIDER=gemini`). Everything else has sensible defaults — see [`.env.example`](./.env.example).

---

## Improvements

What we built beyond the original 24-hour scope:

- **Two LLM providers behind one interface** — strategist/builder/player agents all go through `darwin.llm.complete*`, which dispatches to Anthropic or Google based on `LLM_PROVIDER`. Function-calling shape is identical on both sides.
- **Top-2 lineage** — the runner-up from each generation is carried into the next round-robin alongside the new champion. Stops the population from collapsing onto a single line of descent.
- **Win-rate selection** — promotion is by tournament-wide win rate (score / games_played) with random tiebreak. We started with a head-to-head gate against the prior champion, but with only 2 games per pair its variance could lock the demo on baseline indefinitely.
- **Static + dynamic candidate gating** — forbidden imports, hallucinated `chess.X` attributes, missing required structure, and a smoke game vs `RandomEngine` all run before a candidate enters the tournament. Most builder failures are caught in <1 s instead of corrupting a 5-minute round-robin.
- **Optional Modal tournament backend** — flip `TOURNAMENT_BACKEND=modal` in `.env` and each game runs in its own Modal container. Real OS-level parallelism and frees the local machine; warmup runs in parallel with strategist + builders so it's a net win on wall-clock.
- **Live dashboard** — every move, strategist question, and Elo update streams over WebSocket. `state.cleared` events let one client wipe everyone's view in lockstep.
- **Replay command** — `make replay` re-emits the persisted event stream over WS. Designed as a demo safety net for when the live run misbehaves.
- **Pure-code engine experiment** — the [`experiment-pure-code-engines`](./docs/experiment-pure-code.md) branch flips the design so the LLM *writes* a classical alpha-beta engine that plays in pure Python (no LLM at move time). ~50 ms per move instead of seconds, ~5 LLM calls per gen instead of ~1000.

---

## Shortcomings

We know about these. They are deliberate punts given the timeline, not surprises.

- **Selection is by tournament score, not Elo.** Highest cohort score wins; ties are broken randomly. Elo is persisted and shown but does not gate promotion. Over many generations this can let a slightly-noisier engine displace a slightly-stronger one.
- **No time-decay on Elo.** An engine that only ever played in generation 1 holds its gen-1 Elo forever (forward-fill in the chart). Useful for legibility, misleading for absolute strength comparisons.
- **Strategist question pool is small.** On the deterministic-strategist branch the rotation cycles after ~5 generations per category. On the LLM-strategist branch novelty depends on whatever the model is willing to suggest given recent history — empirically that also plateaus.
- **Builder failures are common at first.** ~30–50% of generated engines hit a static gate on the first turn. Logging tells you which gate killed which candidate, but the failure rate eats into the per-gen cohort size.
- **Mid-tournament dashboard state is stale.** The bracket's blue-highlighted "incumbent" row tracks the champion *coming into* the gen and only flips when `generation.finished` fires. Screenshotting mid-tournament shows stale state.
- **Single-machine SQLite.** Persistence is a SQLite file (`backend/darwin.db`). Fine for a hackathon and for `make replay`; not suitable for a long-running, multi-tenant deployment.
- **No resume.** If a generation crashes or the backend restarts mid-tournament, the partial state is dropped. The orchestrator restarts from the last persisted champion.
- **Modal backend is opt-in.** It works and is faster, but it's not the default — there are still some sharp edges around stale-event drainage when a generation is cancelled.
- **Pure-code branch is local-only.** It does not ship to `main`. Merging would require also bringing across the Modal deployment, the model env vars, and an API contract change for `ratings` on `generation.finished`.

---

## Development process

Five engineers, one weekend hackathon, five branches in parallel. Plans live in [`plans/`](./plans/):

| Branch | Owner | Workstream |
|---|---|---|
| `feat/engine-core` | Person A | Engine Protocol, baseline, registry |
| `feat/tournament` | Person B | Referee, runner, Elo, selection gate |
| `feat/agents` | Person C | Strategist, builder, validator |
| `feat/frontend` | Person D | Dashboard, board, charts, strategist feed |
| `feat/infra` | Person E | API, WebSocket, DB, orchestration, demo |