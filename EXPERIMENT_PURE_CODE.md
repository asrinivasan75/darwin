# Experiment: pure-code engines + Modal tournaments

**Branch:** `experiment-pure-code-engines`
**Status:** local-only — does **not** ship to `main`
**Owner:** Person C (Aadithya)

This branch flips Cubist from "LLM-prompt evolution" to "LLM-as-classical-
engine-author." Candidate engines are pure Python — they don't call the
LLM at play time. Only the *builder* (Gemini, generating engine source)
touches an LLM; the strategist became deterministic on this branch.

The TL;DR is: faster tournaments, cheaper API spend, totally different
demo storyline. Production `main` is unchanged and continues running
the LLM-driven design.

---

## Why this branch exists

The original Cubist design has every candidate engine subclass
`BaseLLMEngine` and call `complete_text(...)` from inside `select_move`.
Every move = one Gemini API call. ~30 candidate moves per game × ~24
games per generation = ~720 LLM calls per gen tournament *just for
moves*, before counting strategist + builder + smoke. That's slow
(seconds per move), expensive (real quota burn), and entirely gated on
Gemini's rate limit.

The pure-code design asks Gemini to *write a complete classical chess
engine* — alpha-beta search, evaluation function, opening book, etc. —
and that engine plays without consulting any LLM. ~50 ms per move
instead of ~1–3 s. Tournaments finish in seconds instead of minutes.
API spend is ~5 calls/gen total (4 builder + 0 strategist), not 1000+.

Tradeoff: the demo storyline is "LLM writes alpha-beta variants" rather
than "LLM evolves chess prompts" — arguably less novel, but visibly
working in seconds instead of half-broken in minutes.

---

## What was changed (file-by-file)

### Backend

#### `backend/cubist/agents/builder.py`
- Dropped the `llm_call` entry from `REQUIRED_PATTERNS` — engines no
  longer have to call `complete_text` / `complete`.
- Removed the `_check_llm_call_in_loop` static gate (was AST-walking
  for LLM calls inside loops; pointless when there are no LLM calls).
- All other gates still active: forbidden-imports, `BANNED_IMPORTS`
  (the `from cubist import config as settings` trap), hallucinated
  `chess.X` attributes via `_check_hallucinated_chess_attrs`.

#### `backend/cubist/agents/prompts/builder_v1.md`
- Header rewritten from "LLM-prompt strategy" to "complete classical
  chess engine in pure Python."
- `cubist.llm` removed from the allowed-imports list.
- New explicit rules:
  - `select_move` is pure Python — must NOT call `complete*`.
  - Per-move budget is 5 s; engines must respect this.
  - If you implement quiescence, cap recursion at depth ≤ 4.
  - In any inner loop with > ~200 iterations, insert
    `await asyncio.sleep(0)` once per outer-loop iteration so
    `asyncio.wait_for` cancellation can actually kill a slow move.
- Worked example replaced: previously showed an LLM-wrapper engine,
  now shows a 1-ply material-eval engine with proper fallback.

#### `backend/cubist/agents/strategist.py`
**Rewritten — no longer calls an LLM.**

- `propose_questions` is now deterministic. Picks 4 questions per gen,
  one each from `CATEGORIES_USED = ["search", "evaluation", "book",
  "sampling"]`. (`prompt` dropped — meaningless for pure-code.)
- Question texts come from `QUESTION_POOLS`: 4–5 concrete, actionable
  improvement directions per category (e.g., for `search`: iterative
  deepening, PVS, transposition table, MVV-LVA, late-move reductions).
- Rotation: `(generation_number - 1 + champion_wins_in_this_category) %
  pool_size`. Winning categories advance their pointer faster — closest
  deterministic analogue to "build on what's working."
- Signature preserves `champion_code`, `runner_up_code`,
  `champion_question`, `history` for orchestrator API compatibility,
  but the only inputs that affect output are `generation_number` and
  the champion-category counts derivable from `history`.

#### `backend/cubist/orchestration/generation.py`
- Calls `propose_questions` with real `history` and `generation_number`
  (previously passed `[]` every gen, which broke rotation).
- Builds history list from `GenerationRow`, parsing each gen's
  `champion_after` name (`gen{N}-{cat}-{hash}`) to recover the winning
  category — feeds that into the strategist's bias logic.
- Calls `warm_modal_pool(20)` at the start of each `run_generation`
  so the warm pool spins up while strategist + builder + smoke run
  (~30 s of compute), and `cool_modal_pool()` in a `finally` so it
  always drains back to 0 idle even on cancel/crash.
- Logs each incumbent load with `loaded incumbent X from <path>` so we
  can see whether top-2 actually carries over, and prints
  `DROPPED incumbent X — ... (this is why next-gen cohort is smaller
  than expected)` if a runner-up's `EngineRow` is missing or
  `load_engine` raises.

#### `backend/cubist/tournament/runner.py`
- Branches on `settings.tournament_backend` between `_round_robin_local`
  (existing asyncio path) and `_round_robin_modal` (new — dispatches
  each game to a Modal container).
- `warm_modal_pool(n)` and `cool_modal_pool()` helpers — best-effort
  calls to `modal.Function.update_autoscaler(min_containers=N)`.
- `_round_robin_modal`:
  - Looks up the deployed `play_game_remote` and
    shared `events_queue` via `modal.Function.from_name` /
    `modal.Queue.from_name`.
  - Drains stale events from the queue at the start.
  - Runs a concurrent drainer task that pulls events in batches of 10
    via `events_queue.get_many.aio(10)` and forwards them to the
    local bus, so the dashboard sees moves in near-real-time.
  - Spawns games via `play_game_remote.starmap.aio`.
  - Tail-drains the queue after the last game, then cancels the drainer.

#### `backend/cubist/tournament/modal_runner.py` (new file)
- Defines `cubist-tournament` Modal app.
- Image: `debian_slim` + `python-chess`/`sqlmodel`/`pydantic`/
  `pydantic-settings`. **No** `google-genai` or `anthropic` —
  pure-code engines don't need them. Saves ~100 MB image weight and
  ~2 s cold-start.
- Local `cubist` source baked into the image via
  `add_local_python_source("cubist", copy=True)`.
- `play_game_remote` function:
  - `cpu=1`, `timeout=60` (down from initial 180; pathologically slow
    engines die at the container level instead of holding up the
    tournament).
  - `max_containers=40` so the worst-case 30-game round-robin runs
    without queueing.
  - `min_containers=0` — no idle baseline cost. The orchestrator's
    auto-warm bumps this to 20 just for the duration of a generation.
  - `enable_memory_snapshot=True` — Modal checkpoints the container
    after `from cubist...` imports complete, dropping cold-start
    from ~5–10 s to ~1–2 s for non-warm containers.
  - Takes engine source as strings (full module text), `exec`s it
    into a fresh module namespace inside the container, plays one
    game via `cubist.tournament.referee.play_game`.
  - Buffers events into a list, flushes via
    `events_queue.put_many.aio(batch)` every 10 events to amortize
    the per-RPC ~50–100 ms cost.

#### `backend/cubist/config.py`
- Added `tournament_backend: str = "local"`. Toggle to `"modal"` via
  `TOURNAMENT_BACKEND=modal` in `.env` to dispatch tournaments to
  Modal containers.

#### `backend/cubist/agents/builder.py`
- Existing chess-attrs gate (catches hallucinated `chess.X` like
  `chess.NAVY`) and import-allowlist regex still active.

#### `backend/cubist/engines/registry.py`
- Already-merged fix from earlier today: registers file-loaded
  modules in `sys.modules` so `inspect.getsource(type(engine))` works
  on file-imported candidates — required for top-2 lineage to read
  the new champion's source on the next gen.

### Frontend

#### `frontend/src/components/Bracket.tsx`
- Bracket cells changed from per-color W/L/D to **pair-aggregate
  scores** (e.g. `1.5/2`). Eliminates the "W in one cell, D in the
  symmetric cell" confusion when white-advantage produces asymmetric
  results across two color games.
- Color: green if matchup won (>50%), red if lost (<50%), yellow if
  exactly 50/50, gray if not played yet.

#### `frontend/src/components/EnginesEloChart.tsx`
- **Forward-fill Elo across non-played gens.** Engines that didn't
  play in a gen now hold their last-known Elo as a flat segment
  rather than dropping out. Lines are continuous.
- **Top-8 by current Elo only** — prevents legend-melt with 30+
  candidate engines after a few gens.
- Legend sorted by current Elo descending (with `baseline-v0` always
  first so the blue color slot is consistent).

#### `frontend/src/components/LiveBoards.tsx` (Kevin's `cc19875`)
- Cherry-picked from `origin/main`. Adds:
  - Per-board move list with PGN-style `1. e4 e5` pair rendering
    (latest pair on top).
  - Termination labels with color (red for hallucination,
    yellow for checkmate, sky-blue for draw).
  - Color-coded result badges.
- Plus: my own tweak so move text is raw SAN with proper move
  numbering (`N.` for white, `N...` for black if shown alone).

### Configuration

#### `.env`
- `TOURNAMENT_BACKEND=modal` — dispatches all tournament games to
  Modal containers.
- `TIME_PER_MOVE_MS=5000` — 5 s per-move budget (was 20 s). Pure-code
  engines are ms-fast; the tighter budget kills synchronous slow
  engines faster.
- `MAX_PARALLEL_GAMES=12` — local-fallback concurrency cap.
- All three Gemini model env vars (`STRATEGIST_MODEL`,
  `PLAYER_MODEL`, `BUILDER_MODEL`) set to `gemini-3-flash-preview`.
  The strategist no longer calls an LLM, so its model is unused; the
  builder uses it to write engine code; the player_model is unused
  on this branch (pure-code engines don't call LLMs).

#### `backend/pyproject.toml` + `uv.lock`
- Added `modal` dependency.

### Tests

- `backend/tests/test_strategist.py` — rewritten for the deterministic
  strategist:
  - Returns 4 distinct categories (count + uniqueness)
  - Rotates between gens (different gens hit different pool entries)
  - Accepts but ignores legacy `champion_code`/`runner_up_code`
    /`champion_question` kwargs
- `backend/tests/test_runner.py` — added `_force_local_backend` autouse
  fixture so the tests don't try to dispatch to Modal when the user's
  `.env` is set to `TOURNAMENT_BACKEND=modal`.
- All 46 tests pass.

---

## Modal app

Deployed at https://modal.com/apps/asrinivasan75/main/deployed/cubist-tournament

To redeploy after changing local `cubist` code:
```
cd backend
.venv/bin/modal deploy cubist/tournament/modal_runner.py
```

Manual warm pool control (if not using auto-warm):
```
modal app keep-warm cubist-tournament play_game_remote 20  # warm up
modal app keep-warm cubist-tournament play_game_remote 0   # cool down
```

---

## How to run locally

```bash
# Backend
cd backend
.venv/bin/python ../scripts/seed_baseline.py     # seed baseline-v0 if DB is fresh
.venv/bin/uvicorn cubist.api.server:app --host 127.0.0.1 --port 8000

# Frontend (separate terminal)
cd frontend
npm run dev    # serves on localhost:5173
```

Then click **Run Generation** in the dashboard. With `TOURNAMENT_BACKEND=
modal`, expect:
- Strategist + 4 builders + smoke validation: ~20–25 s on local backend
- Modal warm-up of 20 containers: in parallel with the above (no extra
  wall-clock)
- Tournament dispatch: 20–30 games (depending on accepted candidates +
  top-2 incumbents), all running concurrently on Modal containers
- Tournament wall-clock: ~10–20 s typical
- Total per-gen wall-clock: ~30–40 s

If `TOURNAMENT_BACKEND=local`, games run on this machine via
`asyncio.gather` capped at `MAX_PARALLEL_GAMES`. Slower but no Modal
dependency.

---

## Known limitations

- **`Bracket` may show stale incumbent during in-flight gens.** The
  blue-highlight row tracks the incumbent coming *into* the gen and
  flips to the new champion only when `generation.finished` fires.
  If you screenshot mid-tournament you'll see stale state.
- **Elo persistence is per-gen.** An engine that only played gen 1
  shows a flat horizontal line at its gen-1 Elo across all later gens
  (forward-fill). That's by design — there's no time-decay.
- **Selection is by tournament score, not Elo.** Highest cohort score
  wins (random tiebreak). Elo is a separate stat that's persisted but
  doesn't gate promotion.
- **Strategist pool size is small.** ~5 entries per category means
  the rotation cycles after 5 gens. If you want more variety, extend
  `QUESTION_POOLS` in `strategist.py` — each entry is just a string.
- **Pure-code branch is local-only.** `main` continues running the
  LLM-driven design. To merge this into `main`, you'd need to also
  bring across the Modal deployment + the `.env` model + the API
  contract changes (`ratings` field on `generation.finished`).

---

## Commit log on this branch

(top of `git log` — see history for the full list)

```
4ac2f67  fix(ui): persist engine Elos across non-played gens; cap chart to top-8
ca25bb0  fix(experiment): strategist rotates per gen + biases toward winning categories
ddccb12  experiment: aggregate bracket scores + bigger warm pool + lineage logging
a151a02  experiment: deterministic strategist (no LLM calls)
68f6c81  experiment: Modal tournament backend + speed/UI polish
8878ddd  boards changes  (Kevin's cherry-pick)
d24ff92  experiment(local-only, NOT for main): pure-code engine builder
```

Below `d24ff92`, the branch contains the same history as `person-c-ux-cancel`
(all the chess-attrs/Clear-button/Modal-prep/etc work that *did* land on
`main`).
