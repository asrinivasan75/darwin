# Darwin

<p align="center">
  <img src="docs/darwin-logo.png" alt="Darwin logo" width="220" />
</p>

A self-improving chess engine that evolves its own scaffolding via agentic tournament selection.

**Demo:** [youtube.com/watch?v=g2TgF6kXoFA](https://www.youtube.com/watch?v=g2TgF6kXoFA)

---

## Overview

Darwin is survival-of-the-fittest for chess engines. Each **generation** spawns a handful of variations of the current champion, plays them all against each other in a round-robin tournament, and crowns the best as the new champion that the next generation will mutate. Run it long enough the engine improves over time, with each champion descended from the last.

What happens inside one generation:

1. **Propose.** The reigning champion's source is read, and a few concrete improvement directions are generated for it (different angles each time — opening books, search, prompt tweaks, position evaluation, sampling).
2. **Mutate, in parallel.** Each direction is turned into a full candidate engine — a real Python module subclassing the `Engine` Protocol. Critique-and-revise pass before validation.
3. **Cull.** Static gates + a smoke game vs `RandomEngine` reject anything that crashes, times out, or plays illegal moves. Nothing broken reaches the arena.
4. **Tournament.** Round-robin of the survivors plus the previous champion and runner-up. Top-2 by win rate (random tiebreak) advance — the runner-up keeps the lineage from collapsing onto a single line of descent.

The dashboard streams every move, every proposed direction, every bracket result, and every Elo update over a WebSocket, so you can watch a generation unfold in real time. A second branch — [`docs/experiment-pure-code.md`](./docs/experiment-pure-code.md) — flips the design so the candidate is a *classical* alpha-beta engine that runs with no model in the loop at game time.

For deeper reading: [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md), [`docs/PROCESS.md`](./docs/PROCESS.md), [`plans/`](./plans/), [`docs/SHORTCOMINGS.md`](./docs/SHORTCOMINGS.md).

---

## Quickstart

```bash
make install                        # uv sync backend + npm install frontend
cp .env.example .env                # then fill in ANTHROPIC_API_KEY (or GOOGLE_API_KEY)
uv tool install honcho              # only if you plan to use `make dev`
make seed                           # initialize the DB and insert baseline-v0 (idempotent)
```

Pick a provider in `.env`: `LLM_PROVIDER=claude` (default) or `LLM_PROVIDER=gemini`. Update `STRATEGIST_MODEL`, `PLAYER_MODEL`, `BUILDER_MODEL` to provider-appropriate values when switching.

### Running the dashboard

```bash
make dev                            # backend (:8000) + frontend (:5173) via honcho
```

Open [http://localhost:5173](http://localhost:5173) and click **Run Generation**.

### Triggering a generation

```bash
make run                            # one generation via the CLI
make run N=3                        # three back-to-back
curl -X POST http://localhost:8000/api/generations/run   # via the API
```

### Other targets

```bash
make test                                       # pytest
make check                                      # lint + tests (pre-PR gate)
make smoke                                      # 10-move baseline self-play
make eval WHITE=baseline-v0 BLACK=random N=10   # head-to-head match
make replay                                     # re-emit persisted gens over WS
make reset                                      # drop the DB and re-seed
```

`make help` lists everything.

---

## Repository layout

```
backend/      Python: engines, agents, tournament, API, orchestration
frontend/     React + Vite dashboard
scripts/      One-off CLIs (run a generation, eval matches, smoke, replay)
plans/        Per-person build plans for the 5-engineer team
docs/         Architecture, process, proposal, shortcomings
```

Frozen contracts (interfaces between workstreams — change only with team sign-off): see [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md#frozen-contracts).

---

## Configuration

At minimum: `ANTHROPIC_API_KEY` (or `GOOGLE_API_KEY` if `LLM_PROVIDER=gemini`). Everything else has sensible defaults — see [`.env.example`](./.env.example). Full dependency list in [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md).

---

## Development process

We parallelized the workload across each other — see [`docs/PROCESS.md`](./docs/PROCESS.md) and [`plans/`](./plans/) for details.
