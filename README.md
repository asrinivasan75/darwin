# Darwin

<p align="center">
  <img src="docs/darwin-logo.png" alt="Darwin logo" width="220" />
</p>

A self-improving chess engine that evolves its own scaffolding via agentic tournament selection.

**Demo:** [youtube.com/watch?v=g2TgF6kXoFA](https://www.youtube.com/watch?v=g2TgF6kXoFA)

---

## Overview

Darwin asks an LLM to write a chess engine, then asks another LLM how to make it better, then asks more LLMs to act on those suggestions, then plays the resulting candidates in a round-robin to decide who survives. Each pass through that pipeline is a **generation**. The next generation starts from the previous generation's winner (and runner-up).

Roles each generation:

1. **Strategist** — proposes concrete improvement directions for the reigning champion.
2. **Builders (in parallel)** — each builder agent receives one question and emits a complete Python module satisfying the `Engine` Protocol.
3. **Validator** — static-source gates plus a smoke game vs `RandomEngine`. Anything that crashes, times out, or returns illegal moves never reaches the tournament.
4. **Tournament** — round-robin of survivors plus the previous champion + runner-up. Promotion is by tournament-wide **win rate** with random tiebreak.

The dashboard streams every move, every strategist question, every bracket result, and every Elo update over a WebSocket so you can watch a generation unfold in real time. A second branch — [`docs/experiment-pure-code.md`](./docs/experiment-pure-code.md) — flips the design so the LLM *writes* a classical alpha-beta engine that plays without any LLM at runtime.

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

Five engineers, one weekend hackathon, five branches in parallel. Plans live in [`plans/`](./plans/); the canonical workflow doc is [`docs/PROCESS.md`](./docs/PROCESS.md).
