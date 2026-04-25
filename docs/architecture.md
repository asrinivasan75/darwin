---
title: "Darwin — Architecture & Technical Report"
subtitle: "A self-improving chess engine that evolves its own scaffolding via agentic tournament selection"
author: "Darwin team"
date: "2026"
---

\newpage

# 1. Executive summary

Darwin is a chess-engine factory. It is not a chess engine.

The system asks one large language model (the **strategist**) to read the
source code of the current champion engine and propose concrete,
categorised improvement directions. It then asks four other large
language models in parallel (the **builders**) to each implement one of
those directions as a complete Python module. Each candidate module is
critiqued by a fifth model (the **adversary**), revised by a sixth (the
**fixer**), passed through six static-source gates, and smoke-tested
against a `RandomEngine` opponent. Survivors enter a round-robin
tournament alongside the prior champion and the prior runner-up. The
top engine by win rate becomes the new champion; the runner-up is
carried forward as a second incumbent so the population does not
collapse onto a single lineage.

One pass through that pipeline is a **generation**. The next generation
starts from the previous generation's winner, and the loop runs until
the operator stops it. Every move played, every strategist question
asked, and every Elo update produced is streamed live to a React
dashboard over a WebSocket.

The core insight is that the LLM is not the chess engine. The LLM is
the *author* of chess engines — a generator over a small, sharply
constrained search space (Python modules conforming to one Protocol).
Tournament play is the fitness function, win rate is the selection
gradient, and `git`-style lineage tracking is the genome. Every
component except the LLM is deterministic, observable, and testable in
isolation.

This document is the system reference. It covers the components,
contracts, control flow, and design trade-offs of Darwin in enough
detail that a reader who has never opened the codebase can predict how
each subsystem behaves under failure and why the boundaries are drawn
where they are. Nothing in this document is forward-looking: every
behaviour described corresponds to code on the `adversary-changes`
branch as of writing.

\newpage

# 2. System architecture

## 2.1 Top-level block diagram

```
+----------------------------------------------------------------------------+
|                                Browser                                     |
|                                                                            |
|   +-----------------+   +-----------------+   +------------------------+   |
|   |  Live boards    |   |  Strategist     |   |  Tournament bracket    |   |
|   |  (react-chess-  |   |  question feed  |   |  + Elo charts          |   |
|   |   board)        |   |                 |   |  (recharts)            |   |
|   +--------+--------+   +--------+--------+   +-----------+------------+   |
|            |                     |                        |                |
|            +---------- WebSocket /ws (envelope events) ---+                |
|                                  |                                         |
+----------------------------------|-----------------------------------------+
                                   |
+----------------------------------|-----------------------------------------+
|                            FastAPI process                                 |
|                                  |                                         |
|        +---------- EventBus (in-process pub/sub fanout) -----------+      |
|        |                                                            |      |
|   +----+--+   +-----------+   +-----------+   +----------+   +----+---+  |
|   | /ws   |   | /api/...  |   | Orchestr- |   | Storage  |   |Tourna- |  |
|   |       |   | routes    |   | ation     |   | (SQLite) |   |ment    |  |
|   +-------+   +-----------+   +-----+-----+   +----------+   +---+----+  |
|                                     |                            |       |
+-------------------------------------|----------------------------|-------+
                                      |                            |
              strategist / builder / adversary / fixer            |
                                      |                            |
                                      v                            v
                       +-------------------------+    +-----------------------+
                       | LLM provider dispatch   |    | Tournament backend    |
                       | (darwin/llm.py)         |    |  - local asyncio      |
                       |  - Anthropic / Claude   |    |  - Modal (1 game per  |
                       |  - Google / Gemini      |    |    container)         |
                       +-----------+-------------+    +-----------+-----------+
                                   |                              |
                                   v                              v
                         Anthropic / Google APIs        Modal cluster (warm pool
                                                         scaled with the gen)
```

The dashboard, the FastAPI process, and the tournament backend are
three independent failure domains. The dashboard can disconnect and
reconnect without affecting the orchestrator; the orchestrator can
crash without affecting the dashboard's already-rendered state; the
Modal backend can fail and the orchestrator falls back to a local
`asyncio.gather` over the same pairings. Every cross-domain boundary
is a documented contract — Engine Protocol at the engine boundary, DB
schema at the persistence boundary, WebSocket envelope at the UI
boundary.

## 2.2 Workstreams

Darwin was built by five engineers working in parallel branches off of
`main`. Each branch owned a vertical slice of the system, with the
boundaries chosen so any two slices share at most one frozen contract:

| Branch                | Owner    | Workstream                                                           |
|-----------------------|----------|----------------------------------------------------------------------|
| `feat/engine-core`    | Person A | Engine Protocol, baseline engine, dynamic registry                   |
| `feat/tournament`     | Person B | Referee, round-robin runner, Elo, selection gate                     |
| `feat/agents`         | Person C | Strategist, builder, validator, adversary, fixer                     |
| `feat/frontend`       | Person D | Dashboard, board, charts, live strategist feed                       |
| `feat/infra`          | Person E | API, WebSocket, DB, orchestration, demo plumbing                     |

The frozen contracts (§4) are the communication primitives between
these workstreams. They are deliberately small and hard to change.

\newpage

# 3. The generation loop

## 3.1 Sequence

A single generation is one execution of
`darwin.orchestration.generation.run_generation`. From the orchestrator's
viewpoint it is a strict, mostly-sequential pipeline; internally many
phases fan out into bounded parallelism.

```
  run_generation(incumbents, generation_number)
          |
          | 1. Warm Modal pool (no-op on local backend)
          |    -> sets min_containers=40 in the background
          |
          | 2. Emit  generation.started
          |
          | 3. Strategist
          |    deterministic rotation over (gen_number + win_count) % pool
          |    -> 4 Question objects (one per category)
          |    Emit  strategist.question x 4
          |
          | 4. asyncio.gather over 4 build_engine calls (parallel)
          |    each call: champion source + question -> Python module text
          |    -> 4 Path objects (or Exception)
          |
          | 5. asyncio.gather over 4 _validate_one calls (parallel)
          |    each call:
          |       a. (optional) adversary critique  -> Critique
          |          Emit  adversary.completed
          |       b. (optional) fixer rewrite       -> revised module on disk
          |          Emit  fixer.completed
          |       c. validate_engine (static + smoke vs RandomEngine)
          |          Emit  builder.completed
          |    -> list of (Engine, abs_path) or None
          |
          | 6. round_robin(incumbents + survivors)
          |    For every distinct (white, black) pair, play games_per_pairing games
          |    Each game emits  game.move...  game.finished
          |
          | 7. select_top_n(standings, primary, others, n=2)
          |    win rate, random tiebreak  -> [new_champion, runner_up]
          |
          | 8. update_ratings_for_games  (Elo, K=32, batched)
          |
          | 9. Persist:  GenerationRow, GameRow x N, EngineRow x candidates
          |
          | 10. Emit  generation.finished  (with full cohort Elo dict)
          |
          | 11. Cool Modal pool back to 0
          |
          v
          return [new_champion, runner_up]
```

`run_generation_task` wraps `run_generation` so that any unhandled
exception still emits a terminal `generation.finished` (with
`promoted=false`), and an `asyncio.CancelledError` emits
`generation.cancelled` before the cancel propagates. Without those
two, the dashboard would hang on "Waiting for strategist…" until the
operator notices the orphaned task.

## 3.2 Parallelism inside the loop

There are three places where the orchestrator goes wide:

1. **Builders.** Four independent `build_engine` calls run under a
   single `asyncio.gather`. Each can take 10–30 seconds; running them
   serially would dominate generation wall-clock. They are
   independent because each receives the same champion source and a
   different `Question` — there is no cross-talk.

2. **Validators.** The four candidates that returned successfully are
   each handed to `_validate_one` under another `asyncio.gather`. This
   means the adversary critique, fixer rewrite, and 60-second smoke
   game cap run in parallel across candidates. Each emits its own
   `builder.completed` immediately on finish so the dashboard sees the
   results stream rather than batched at the end.

3. **Tournament games.** `round_robin` enumerates every (white, black,
   game_id) triple where `i != j` and dispatches each as one
   `play_game` call. On the local backend an `asyncio.Semaphore`
   bounded by `max_parallel_games` (default 16) caps concurrency. On
   the Modal backend each game is one container — true OS-level
   parallel, no GIL — and the bound is Modal's `max_containers`.

The Modal pool warm-up (`warm_modal_pool(40)`) is deliberately the
*first* call inside `run_generation`. Modal needs ~25–30 seconds to
spin up cold containers; that is roughly the wall clock of the
strategist + builder + smoke phases. By the time the round-robin
actually wants to dispatch, the pool is warm and tournament wall-clock
skips cold-start entirely.

## 3.3 Anti-collapse provisions

A naive implementation of "next generation seeds from this
generation's winner" rapidly collapses onto a single line of descent —
every gen mutates the same ancestor along whatever direction the
strategist picks first. Darwin makes three choices to fight this:

1. **Top-2 lineage.** After tournament selection
   (`select_top_n(..., n=2)`), both the champion and the runner-up are
   stored in the lineage table and re-injected as `incumbents` in the
   next call to `run_generation`. The runner-up is shown to the
   strategist as context, so the next gen's questions can reason about
   "two strong but distinct designs from the same population".

2. **Strategist categories.** Questions are partitioned across four
   *disjoint* categories — `search`, `evaluation`, `book`, `sampling`
   — and one builder is dispatched per category. A generation cannot
   fail to explore any of these axes because there is always exactly
   one candidate trying each.

3. **Win-rate selection with random tiebreak.** When two engines tie
   on win rate (very common in 2-game-per-pair round-robins), tie
   resolution is uniform-random. This preserves population variance
   that a deterministic tiebreak (e.g. by name) would actively erode.

\newpage

# 4. Frozen contracts

These are the interfaces between the workstreams. Each is marked as a
*frozen contract* in its source file: changes require team sign-off
because every other workstream's code depends on the shape staying
fixed.

## 4.1 Engine Protocol

Defined in `backend/darwin/engines/base.py`.

```python
@runtime_checkable
class Engine(Protocol):
    name: str
    generation: int
    lineage: list[str]

    async def select_move(
        self,
        board: chess.Board,
        time_remaining_ms: int,
    ) -> chess.Move: ...
```

This is the *only* shape that the tournament runner cares about. The
baseline engine, every builder-emitted module, the seeded baseline
loaded from `darwin.engines.baseline` — all satisfy this Protocol and
nothing more. The Protocol is `runtime_checkable`, so
`isinstance(engine, Engine)` is the gate that the dynamic registry
uses to reject malformed builder output.

The Protocol is small for two reasons. First, it minimises the surface
area an LLM has to get right: the builder needs to subclass
`BaseLLMEngine` (which fills in `name` / `generation` / `lineage` for
free) and implement one async method. Second, it gives the tournament
runner a hard isolation boundary: a misbehaving engine that infinite-
loops or returns illegal moves cannot corrupt anything outside its
`select_move` call.

`BaseLLMEngine` is a convenience base class — not part of the Protocol
— that stores the three required fields and raises
`NotImplementedError` on `select_move`. Every builder-generated engine
in practice subclasses it.

**Why frozen.** Three workstreams depend on this exact shape. The
tournament runner (`backend/darwin/tournament/referee.py`) calls
`engine.select_move(board, time_remaining_ms)` with positional args.
The builder agent emits text that subclasses `BaseLLMEngine` and is
typed against this signature. The dynamic registry
(`backend/darwin/engines/registry.py`) does an `isinstance` check
against the Protocol. Any change to the field set or method signature
breaks all three at once.

## 4.2 Database schema

Defined in `backend/darwin/storage/models.py` as SQLModel tables over
SQLite.

| Table         | Key columns                                                                                                |
|---------------|------------------------------------------------------------------------------------------------------------|
| `engines`     | `id` PK, `name` UNIQUE, `generation`, `parent_name`, `code_path`, `elo`, `created_at`                      |
| `games`       | `id` PK, `generation`, `white_name`, `black_name`, `pgn`, `result`, `termination`, `created_at`            |
| `generations` | `id` PK, `number` UNIQUE, `champion_before`, `champion_after`, `strategist_questions_json`, `started_at`, `finished_at` |

There are deliberately no foreign keys between tables. `games.white_name`
joins to `engines.name` only at read time, and the orchestrator
explicitly tolerates orphan `GameRow`s that reference an `EngineRow`
the operator wiped via `POST /api/state/clear`. This makes
`/api/state/clear` a one-table-at-a-time delete with no ordering
constraints — which is what we want when an operator clicks the wipe
button mid-generation.

`code_path` is an absolute filesystem path for builder-emitted engines
and a dotted module path (`darwin.engines.baseline`) for the seeded
baseline. The `/api/engines/{name}/code` route distinguishes the two
by checking `Path.is_absolute()` and falling back to
`importlib.find_spec` for the dotted form.

`strategist_questions_json` is a JSON-encoded array of
`{category, text}` objects. It is stored as a string so the schema
does not have to grow a fourth table for a per-question child row.
Every consumer parses it with `json.loads` and crashes loudly on
malformed JSON — there is no migration story for the strategist
schema, by design.

**Why frozen.** Person A writes to `engines`, Person B writes to
`games`, the orchestration loop writes to `generations`. The frontend
reads all three via the REST routes; the replay command
(`make replay`) reads `games` to re-emit historical move events.
Changes need to be coordinated across all five workstreams.

## 4.3 WebSocket events

Defined in `backend/darwin/api/websocket.py` and mirrored in
`frontend/src/api/events.ts`. Every event is wrapped in an `Envelope`
discriminated by `event.type`:

| Event                   | Producer       | Consumer                | Payload notes                                              |
|-------------------------|----------------|-------------------------|------------------------------------------------------------|
| `generation.started`    | orchestration  | dashboard header        | `number`, `champion`                                       |
| `strategist.question`   | orchestration  | strategist feed panel   | `index 0..3`, `category`, `text`                           |
| `builder.completed`     | orchestration  | candidate panel         | `question_index`, `engine_name`, `ok`, `error`             |
| `adversary.completed`   | orchestration  | candidate panel         | `summary` (≤ 140 chars), `critique_chars`, `ok`            |
| `fixer.completed`       | orchestration  | candidate panel         | `ok`, `error`                                              |
| `game.move`             | tournament     | live board              | `game_id`, `fen`, `san`, `white`, `black`, `ply`           |
| `game.finished`         | tournament     | bracket                 | `result`, `termination`, `pgn`                             |
| `generation.finished`   | orchestration  | header + Elo chart      | `new_champion`, `elo_delta`, `promoted`, `ratings` dict    |
| `generation.cancelled`  | orchestration  | dashboard               | clears in-progress panels                                  |
| `state.cleared`         | API route      | dashboard               | wipes accumulated event log                                |

`EventBus` is in-process pub/sub: each `/ws` connection calls
`subscribe()` to get its own `asyncio.Queue(maxsize=1000)`, and every
`emit()` does a `put_nowait` fan-out across the live subscriber set.
Backpressure policy is intentional: if a subscriber's queue is full
(slow or stalled browser tab), the event is dropped *for that
subscriber only*, never for the producer. This means a stuck dashboard
client can never block the orchestrator; the cost is a partial event
stream for that one client, which it can recover from by reloading
the page (the REST routes serve the persisted state).

**Why frozen.** The frontend's `events.ts` file is structurally
identical to the backend's `websocket.py` — they are kept in sync by
hand. The discriminated-union shape lets TypeScript exhaustively check
the dashboard's event handlers, so adding a new event without updating
both sides produces a type error, not a silent runtime drop.

\newpage

# 5. Agent design

Darwin runs four LLM-driven roles per generation. Each is a thin
adapter around `darwin.llm.complete`: it formats a Markdown prompt
template, optionally attaches a JSON-Schema tool, calls the model,
and parses the response. None of them remembers state across
generations — the only signal carried forward is the persisted
`GenerationRow` history that the next generation's strategist reads.

## 5.1 Strategist

Location: `backend/darwin/agents/strategist.py`.

On the experimental pure-code branch the strategist is **deterministic
and does not call an LLM**. The reason is symmetry: pure-code engines
have no LLM at runtime, so there is no value in spending API quota on
strategist questions that we can author ahead of time. The async
`propose_questions` signature is preserved so the orchestrator does
not need to branch.

The four active categories are `search`, `evaluation`, `book`,
`sampling`. The `prompt` category exists in `CATEGORIES` (used by the
`_WINNING_CATEGORY_RE` lookup in the orchestrator) but is dropped from
`CATEGORIES_USED` because pure-code engines have no LLM-prompt
component. Each category has a hand-authored pool of 4–5 concrete
question templates — for example, the `search` pool covers iterative
deepening, principal-variation search, transposition tables, MVV-LVA
move ordering, and late-move reductions.

Question selection has two layers:

1. **Base rotation by generation number.** Within each category, the
   pointer is `(generation_number - 1) % len(pool)`. Gen 1 picks
   index 0, gen 2 picks index 1, and so on. This gives every category
   *some* variation across generations even if no candidate from that
   category ever wins.

2. **Performance-aware bias.** Across all prior generations, count
   how often each category produced the new champion (parsed from
   `champion_after`'s `gen{N}-{cat}-{hash}` prefix). For each win in
   category X, advance X's rotation pointer by one extra step. The
   effect is that winning categories explore their pool faster — the
   deterministic analogue of "double down on what's working" without
   an LLM in the loop.

Both layers are pure functions of the persisted generation history,
so `propose_questions` is reproducible: replaying the same DB state
returns the same four questions every time.

A non-deterministic, LLM-driven strategist does exist on the main
line; the prompt template is `prompts/strategist_v1.md`, kept loaded
into `PROMPT` for back-compat. On this branch it is inert.

## 5.2 Builder

Location: `backend/darwin/agents/builder.py`.

The builder receives one `Question`, the source code of the current
champion, and (if available) the source of the prior runner-up. It
formats the `builder_v1.md` prompt template, calls the configured
builder model with a `submit_engine` JSON-Schema tool attached, and
expects the model to invoke that tool with one argument — `code` (a
single Python source string).

The tool definition is forwarded to both providers via
`darwin.llm.complete`. On Anthropic this is the native tool-use
shape; on Google the shape is translated to `Tool(function_declarations=...)`
and the tool config is set to `mode="ANY"` so the model is forced to
emit a `function_call` rather than free text.

Builder validation is a six-gate cascade. Each gate runs cheaper than
the next, so most failures are caught in milliseconds:

| #  | Gate                        | Where                            | What it catches                                                 |
|----|-----------------------------|----------------------------------|-----------------------------------------------------------------|
| 1  | Forbidden import / call     | `FORBIDDEN` regex                | `subprocess`, `os.system`, `eval(`, network sockets, `importlib`, `urllib`, `pty`, `fcntl` |
| 2  | No `tool_use` block         | response inspection              | Model replied with prose instead of invoking `submit_engine`    |
| 3  | Required structure          | `REQUIRED_PATTERNS`              | Missing `engine = ...` symbol; missing `async def select_move`  |
| 4  | Hallucinated `chess.X`      | `_check_hallucinated_chess_attrs`| `chess.NAVY`, `chess.between(...)`, `chess.distance` — names that do not exist on the python-chess module |
| 5  | Static check at validate    | `_static_check_source`           | Re-runs gates 1, 3, 4 against on-disk source — catches hand-edits |
| 6  | Module load                 | `darwin.engines.registry`        | Import error, no `engine` symbol, fails `isinstance(engine, Engine)` |

Gate 6 is followed by a 60-second-cap **smoke game** vs `RandomEngine`
with `time_per_move_ms=10_000`. Any termination in the rejection
set — `error`, `illegal_move`, or `time` — fails the candidate. The
60-second wall-clock cap is the safety net for engines that legally
return moves but design themselves into death-by-N-LLM-calls
patterns: a per-move budget alone would let those run to 20 minutes
of smoke time before the cap fires.

Every rejection is logged at ERROR with the engine name, the gate
that fired, and the failure reason. The raw model response (or, for
gate-1/3/4 rejections, the proposed source) is persisted to
`engines/generated/_failures/<name>.txt` so the operator can do
post-mortems.

`_check_llm_call_in_loop` is an AST-level gate that detects
`await complete(...)` or `await complete_text(...)` calls inside a
`for` loop within `select_move` — the pathological "evaluate every
legal move with the LLM" pattern. It is **disabled** on this branch
because pure-code engines do not call the LLM at runtime at all, so
the stricter check is unnecessary and would false-reject legitimate
helper code.

## 5.3 Adversary

Location: `backend/darwin/agents/adversary.py`.

The adversary sits between the builder and the validator. It reads
the builder's source and the originating question and returns one
short summary line (≤ 140 chars, parsed from a `SUMMARY:` prefix
contract) plus a multi-sentence critique paragraph.

The `summary` is what the dashboard panel shows; the full paragraph
is fed to the fixer. The summary cap exists because an over-long
summary line would dominate the dashboard layout — the prompt asks
for ≤ 90 characters and the parser hard-trims to 140 as a safety
margin.

By design the adversary uses a *different model role* from the
builder: `settings.adversary_provider` and `settings.adversary_model`
are independent settings, so the operator can pin
`builder=gemini`, `adversary=claude` without restarting other roles.
Pairing the same family on both sides tends to rubber-stamp its own
output — a weakness that homogeneous critique cannot catch.

Failure mode: any LLM error or `<20`-character response degrades to
`Critique(summary="", full="")`, which the orchestrator treats as
"skip the fixer". This means an adversary outage falls back cleanly
to the pre-adversary pipeline rather than blocking the candidate.

`enable_adversary` in `Settings` toggles the adversary and fixer
together. When false, builder output goes straight to the validator
— matching the original two-agent pipeline.

## 5.4 Fixer

Location: `backend/darwin/agents/fixer.py`.

The fixer is structurally a *second builder call*. It uses the same
`submit_engine` tool, the same on-disk layout, the same static gates,
and the same builder model — but its prompt includes the original
code and the adversary's critique paragraph. The intent is "revise"
not "rewrite": the same model family that wrote the candidate is the
right family to apply targeted fixes.

If the fixer's response fails any static gate, or if the LLM call
errors, or if no `submit_engine` block is returned, the original
builder file is left in place untouched. This is the correct
fallback because the original passed gate 1–4 already; rejecting the
candidate just because the fix failed would punish the builder for
the adversary's noise.

Both `adversary.completed` and `fixer.completed` are emitted on the
WebSocket so the dashboard can show "critiqued / revised" badges in
real time.

\newpage

# 6. Tournament system

## 6.1 Round-robin schedule

`darwin.tournament.runner.round_robin` enumerates pairings as

```
for i, white in enumerate(engines):
    for j, black in enumerate(engines):
        if i == j: continue
        repeat games_per_pairing times: append (white, black, game_id)
```

For a cohort of `N` engines and `games_per_pairing = K`, the schedule
is `N × (N - 1) × K` games. With the defaults — 6 engines (2
incumbents + up to 4 candidates) and `games_per_pairing = 2` — that's
60 games per generation. The `i == j` skip means an engine never
plays itself; one round-robin column is one fixed colour assignment,
so a pair (A, B) gets `K` games with A as white and `K` games with A
as black, for `2K` games per pair total.

`max_parallel_games` (default 16) caps the asyncio.gather concurrency
on the local backend, which keeps a slow LLM provider from making
every game time out at once. On the Modal backend the bound moves to
Modal's autoscaler.

## 6.2 Referee

`darwin.tournament.referee.play_game` is the per-game state machine.
Its loop is:

```
while not board.is_game_over(claim_draw=True):
    if board.fullmove_number > settings.max_moves_per_game:
        finish as draw, termination = "max_moves"
    engine = white if board.turn == WHITE else black
    try:
        move = await asyncio.wait_for(
            engine.select_move(board.copy(), time_per_move_ms),
            timeout=time_per_move_ms / 1000 + 5,
        )
    except TimeoutError:
        finish as loss, termination = "time"
    except Exception:
        finish as loss, termination = "error"
    if move not in board.legal_moves:
        finish as loss, termination = "illegal_move"
    push move; emit game.move
finish; emit game.finished
```

Five points are deliberate:

1. **Engine sees a copy of the board.** `board.copy()` is passed to
   `select_move`, so a misbehaving engine cannot mutate the
   reference board mid-think.

2. **Timeout has +5 seconds of slack** beyond the requested per-move
   budget. This absorbs provider/network jitter without letting one
   stalled move block the entire tournament indefinitely.

3. **Three terminal-loss terminations: `time`, `error`, `illegal_move`.**
   All three score the same 0–1 for the opponent but emit distinct
   termination strings so the dashboard can colour the bracket cell
   appropriately.

4. **`max_moves_per_game` defaults to 120 fullmoves.** This caps any
   pair of engines that fall into a perpetual shuffling loop — both
   `RandomEngine` self-play and a borderline-broken builder are common
   causes.

5. **PGN is built from `board.move_stack`,** so a partial PGN is
   always emitable on early-termination paths (timeout, error,
   illegal-move).

## 6.3 Local vs Modal backend

`tournament_backend` in `Settings` switches between two
implementations of `round_robin`:

- **`local`** — `asyncio.gather` over every pairing, bounded by an
  `asyncio.Semaphore(max_parallel_games)`. CPU-bound games share the
  GIL; LLM-bound games (legacy path) are limited mostly by provider
  rate limits.

- **`modal`** — each pairing is dispatched as a Modal function call
  (`darwin-tournament/play_game_remote`). One container per game,
  real OS-level parallelism, no GIL contention. A shared
  `modal.Queue` named `darwin-events` is drained by an asyncio task
  on the orchestrator side, so the dashboard sees `game.move` and
  `game.finished` in real time exactly as on the local path.

The Modal path has three pieces of stale-state hygiene:

- **Stale-event drain at start.** Before dispatching new games, the
  drainer pulls from the shared `modal.Queue` until it times out or
  errors, throwing those events away. This stops events from a
  cancelled previous run from bleeding into the current dashboard
  view.

- **Per-game synthesised draw on container failure.** If a Modal
  container raises (typically `FunctionTimeoutError` from the
  container's own 30s OS-level kill), the orchestrator synthesises a
  draw `GameResult(termination="error", pgn="")` and emits a
  `game.finished` for it. The cohort still has all 90 results; one
  game is lost rather than the whole tournament.

- **Tail drain at end.** After the last container finishes, the
  drainer waits 0.5s for in-flight events to land before stopping.

If Modal dispatch raises for *any* reason — auth expired, function
not deployed, network outage, quota — `round_robin` logs a warning
and falls through to the local asyncio path with the same pairings.
The demo still completes; the operator can flip
`TOURNAMENT_BACKEND=local` in `.env` to make the fallback permanent.

## 6.4 Selection: win rate, not Elo

`darwin.tournament.selection.select_top_n` ranks every engine by

```
win_rate(name) = score(name) / games_played(name)
```

with random tiebreak, and returns the top *n* (default 2). The first
element is the new champion; the rest are runners-up that the
orchestrator carries into the next gen.

There are two non-obvious decisions here.

**Win rate, not Elo.** Elo is computed and persisted but never feeds
back into selection. The reason is that Elo is a noisy *per-game*
update whose value depends on the opponent's prior rating; across
five or ten games per cohort, Elo drifts in ways that don't reliably
reflect this generation's actual head-to-head performance. Win rate
is the simple, transparent signal: "how often did you beat the
field?". Elo lives on the dashboard chart as a long-term legibility
signal; it does not gate promotion.

**Win rate, not raw score.** In a clean round-robin every engine
plays the same number of games, and the two metrics give identical
orderings. But if one game errors out, that engine is short a game
— raw score penalises them, win rate does not. Using rate keeps
selection robust to partial cohorts.

A previous gating rule required the top scorer to additionally beat
the prior champion in their direct head-to-head subset. With only
2 games per pair, the head-to-head gate's variance was high enough
to lock the demo on baseline indefinitely (a 1–1 split is the modal
outcome and resolves to *not promoted*). It was removed in favour of
cohort-wide win rate.

## 6.5 Elo bookkeeping

`update_ratings_for_games` applies a single rating-period update at
the end of the tournament:

1. Snapshot `start = dict(ratings)` at the top of the function.
2. For each game, compute `expected_white = 1 / (1 + 10^((R_b - R_w) / 400))`
   from the *snapshot* ratings, accumulate `K * (actual - expected)`
   into a per-engine `delta` dict.
3. Return `start[name] + delta[name]` for every engine.

This is order-independent — exactly the property we need given that
games arrive concurrently from `asyncio.gather` (or from Modal
containers). Applying `update_elo` to each game as results arrive
would produce a different final rating depending on completion order.

`K = 32` is the standard USCF-style hackathon value. New engines and
the seeded baseline both start at 1500.

\newpage

# 7. LLM provider abstraction

## 7.1 Goal

Strategist, builder, adversary, and fixer all go through one entry
point — `darwin.llm.complete` — which dispatches to either Anthropic
or Google based on the resolved provider for the call. Caller code is
provider-agnostic. The shape returned to the caller is the same on
both backends:

```
content = await complete(model, system, user, max_tokens, tools, provider)
# content is a list of blocks
# block.type in {"text", "tool_use"}
# block.text                   when type == "text"
# block.name, block.input      when type == "tool_use"
```

For text-only calls, `complete_text` is a thin wrapper that returns
the first text block's content or `""`.

## 7.2 Per-role provider override

The `Settings` model carries one default provider plus four optional
overrides:

```python
llm_provider: Provider = "claude"
strategist_provider: Provider | None = None
player_provider:     Provider | None = None
builder_provider:    Provider | None = None
adversary_provider:  Provider | None = None
```

`settings.provider_for("builder")` returns `builder_provider or
llm_provider`. Each role passes its resolved provider explicitly to
`complete`, so a single generation can fan out to multiple providers
in parallel — for example, `builder=gemini, adversary=claude` to
break the rubber-stamp failure mode of homogeneous critique. The
model ID for a role must be appropriate for its resolved provider
because each SDK only knows its own model namespace.

## 7.3 Gemini -> Anthropic adapter

Gemini's tool surface is `Tool(function_declarations=[...])` and its
response carries `function_call` parts. To keep caller code identical
across providers, `darwin.llm` does two adapters:

- **Outbound:** `_anthropic_tools_to_gemini` translates
  `{name, description, input_schema}` to
  `FunctionDeclaration(name, description, parameters=input_schema)`.
  No structural translation is needed because Darwin's tool schemas
  are already JSON Schema, which Gemini's `parameters` field accepts.

- **Inbound:** `_gemini_response_to_blocks` walks
  `response.candidates[0].content.parts` and produces a list of
  `SimpleNamespace`s that quack like Anthropic `ContentBlock`s.
  Function calls become `SimpleNamespace(type="tool_use", name=...,
  input={...})`; text becomes `SimpleNamespace(type="text",
  text=...)`. The agent code that iterates `content` for a
  `tool_use` block does not branch on the provider.

Two Gemini-specific knobs are set:

- **`thinking_budget=0`.** Gemini 2.5 Flash/Pro enable "thinking" by
  default, which consumes the output-token budget *before* any
  function_call is emitted. For a builder that needs ~1–2k tokens of
  Python code, thinking can eat the entire budget and the response
  comes back empty. Disabled.

- **`tool_config.function_calling_config.mode="ANY"`.** Forces the
  model to emit a `function_call` rather than free text, which is
  what the builder pipeline expects.

## 7.4 Concurrency and retries

A module-level `asyncio.Semaphore(30)` caps in-flight LLM calls
across all roles. This is enough for the 4 builder + 4 adversary +
4 fixer = 12 concurrent calls a generation produces, with headroom.

Both providers use a 5-attempt exponential backoff (1, 2, 4, 8 s).
The Anthropic path retries on `RateLimitError` and `APIError`. The
Gemini path retries on `genai_errors.APIError` and logs the HTTP
status of each retry — a string of 429s vs a string of 503s used to
look identical from outside the function, which made operator
debugging painful.

When all 5 attempts fail, the Gemini path raises
`RuntimeError("gemini call failed after 5 retries (model=..., last_status=..., ...)")`
with the actual last-status code. This replaces a previous opaque
`RuntimeError("unreachable")` that made it impossible to tell rate-
limit from upstream-overload.

\newpage

# 8. Persistence and observability

## 8.1 Storage

SQLite via SQLModel, file path `backend/darwin.db`. Every write goes
through a fresh `with get_session() as s:` context — there is no
shared session, no connection pool, no async wrapper. SQLite's
write lock is fine for the single-process orchestrator.

The orchestrator commits exactly once per generation, near the end of
`run_generation`, after the tournament has finished and the new
champion has been chosen. Three writes happen in that one commit:

1. One `GenerationRow` with `champion_before`, `champion_after`, the
   serialised strategist questions, and `finished_at`.
2. One `GameRow` per round-robin game. The PGN is stored as text;
   `result` is `1-0` / `0-1` / `1/2-1/2`; `termination` is one of the
   five terminal strings.
3. One `EngineRow` per accepted candidate (de-duplicated by name).
   The Elo column is updated for every engine in the cohort,
   including the prior champion and the prior runner-up.

There is no resume. If the orchestrator crashes mid-tournament, the
partial games are dropped — no `GameRow`s are written, no
`GenerationRow` is written. The next `run_generation_task` call
loads the previous generation's champion from disk and starts again.
This is a deliberate trade: a resume mechanism would require a
"generation in progress" sentinel row plus per-game commits, which
turns the schema into a state machine and substantially raises the
write rate. For a hackathon-scale system the loss of one bad
generation is not worth that complexity.

## 8.2 Live event stream

Every WebSocket subscriber gets its own `asyncio.Queue(maxsize=1000)`.
The orchestrator's `bus.emit(payload)` does an O(subscribers)
fan-out via `put_nowait`. Three properties follow:

- **No blocking.** A stalled subscriber's `put_nowait` raises
  `QueueFull`; the producer catches and drops the event for that
  subscriber only. Other subscribers and the producer itself are
  unaffected. We trade per-subscriber completeness for global
  liveness.
- **No persistence.** The queue is in-memory. A subscriber that
  reconnects gets the *future* event stream; to recover history it
  reads the REST routes (`/api/games?gen=N`, `/api/generations`).
- **Pure pub/sub.** No message has an ID, no message is acked. If
  the orchestrator emits between an old socket closing and a new
  one opening, those events are gone for that client. This is
  acceptable because every event has a corresponding DB row that the
  REST layer can serve on demand.

## 8.3 Replay

`make replay` re-emits the persisted event stream over the WebSocket
in real time. It is a demo safety net: when the live run misbehaves
during a presentation, the operator can replay the last successful
generation instead. The script reads `GameRow.pgn`, walks the moves,
and synthesises `game.move` and `game.finished` events at a
configured cadence. From the dashboard's perspective the replay is
indistinguishable from a live run.

## 8.4 Lineage tracking

Every `EngineRow` carries `parent_name`, set by the builder pipeline
to the *primary* incumbent's name (the seed for the build). Walking
`parent_name` upward gives a parent-pointer tree of every engine
that was ever produced — including engines that lost their
generation and never advanced. The two-incumbent regime means the
runner-up's `parent_name` points at the *previous* champion, not at
the runner-up's own predecessor; this is fine because the lineage
table is not a strict "this engine descends from" assertion, just a
"this engine was built by mutating that engine".

Naming convention enforces the same lineage at the symbol level:
every builder-emitted file is `gen{N}_{cat}_{6-char-sha1}.py` and
the corresponding `engine.name` is the same string with hyphens
(`gen{N}-{cat}-{6-char-sha1}`). The category prefix is what
`_WINNING_CATEGORY_RE` parses to recover which strategist category
produced the current champion, which the next generation's
strategist uses for its performance-aware bias.

\newpage

# 9. Configuration reference

All settings are loaded by `pydantic-settings` from `.env` (path
`../.env` relative to the backend module). Type-checked at import
time; unknown keys are ignored.

| Setting                  | Default                  | Purpose                                                                     |
|--------------------------|--------------------------|-----------------------------------------------------------------------------|
| `llm_provider`           | `claude`                 | Default LLM provider; `claude` or `gemini`                                  |
| `strategist_provider`    | `None`                   | Per-role override; falls back to `llm_provider`                             |
| `player_provider`        | `None`                   | Per-role override                                                           |
| `builder_provider`       | `None`                   | Per-role override                                                           |
| `adversary_provider`     | `None`                   | Per-role override                                                           |
| `anthropic_api_key`      | `""`                     | Required when any role resolves to `claude`                                 |
| `google_api_key`         | `""`                     | Required when any role resolves to `gemini`                                 |
| `strategist_model`       | `claude-opus-4-6`        | Model ID — must match the strategist's resolved provider                    |
| `player_model`           | `claude-sonnet-4-6`      | Player model ID                                                             |
| `builder_model`          | `claude-sonnet-4-6`      | Builder + fixer model ID                                                    |
| `adversary_model`        | `claude-opus-4-6`        | Adversary model ID                                                          |
| `enable_adversary`       | `True`                   | Toggle for the adversary -> fixer chain                                     |
| `database_url`           | `sqlite:///./darwin.db`  | SQLModel database URL                                                       |
| `time_per_move_ms`       | `20_000`                 | Per-move budget in tournament games                                         |
| `games_per_pairing`      | `2`                      | Number of games per (white, black) pair per round-robin                     |
| `max_parallel_games`     | `16`                     | Local-backend concurrency cap (asyncio.Semaphore)                           |
| `max_moves_per_game`     | `120`                    | Fullmove cap before declaring a draw with `termination=max_moves`           |
| `tournament_backend`     | `local`                  | `local` or `modal`                                                          |
| `api_host`               | `127.0.0.1`              | FastAPI bind address                                                        |
| `api_port`               | `8000`                   | FastAPI port                                                                |

\newpage

# 10. Known limitations and design trade-offs

These are deliberate punts. They are documented here so a reviewer
can distinguish "we shipped it like this on purpose" from "we missed
this".

## 10.1 Selection by tournament score, not Elo

Highest cohort score wins; ties resolve randomly. Elo is persisted
and shown but does not gate promotion (§6.4). Over many generations
this can let a slightly-noisier engine displace a slightly-stronger
one — Elo is the better long-run strength signal but its variance at
small sample sizes makes it the worse short-run gate. We chose
short-run robustness.

## 10.2 No time-decay on Elo

An engine that played only in generation 1 holds its gen-1 Elo
forever (forward-fill in the dashboard chart). Useful for legibility
("this engine ended at 1623 and is no longer actively tracked"),
misleading for absolute strength comparisons across distant
generations. The fix is per-generation Elo decay, which we did not
implement.

## 10.3 Strategist question pool is small

On the deterministic-strategist branch (this branch) the rotation
cycles after 4–5 generations per category, depending on pool size.
On the LLM-strategist branch novelty depends on whatever the model
is willing to suggest given recent history — empirically that also
plateaus, just less obviously. Mitigation is hand-authoring more
pool entries; structural fix would be a generative strategist that
synthesises new questions from the win/loss record.

## 10.4 Builder failure rate is non-trivial

Roughly 30–50% of generated engines hit a static gate on the first
generation. Logging tells the operator which gate killed which
candidate, but the failure rate eats into the per-generation cohort
size. The adversary + fixer pass is a partial mitigation: it reduces
the late-stage rejection rate but does not eliminate gate-1 / gate-3
syntactic failures, which are caught before the adversary even sees
the code.

## 10.5 Mid-tournament dashboard state is stale

The bracket's "incumbent" highlight tracks the champion *coming into*
the gen and only flips when `generation.finished` fires. Screenshotting
mid-tournament shows stale state. A streaming-update model would fix
this but adds complexity to the dashboard's reducer; we chose the
stale snapshot.

## 10.6 Single-machine SQLite

Persistence is a single SQLite file. Fine for a hackathon and for
`make replay`; not suitable for a long-running, multi-tenant
deployment. The schema (§4.2) has no foreign keys precisely so a
future migration to Postgres or a managed backend can drop in
without ordering constraints.

## 10.7 No resume

If a generation crashes or the backend restarts mid-tournament, the
partial state is dropped. The orchestrator restarts from the last
persisted champion. §8.1 explains the trade.

## 10.8 Modal backend is opt-in

It works and is faster, but it is not the default. There are still
sharp edges around stale-event drainage when a generation is
cancelled (the cancellation path emits `generation.cancelled` but
in-flight container events may land on the next dashboard tick).
Setting `TOURNAMENT_BACKEND=modal` in `.env` enables it; fall-back
to local on dispatch failure (§6.3) means the demo never blocks on
Modal availability.

## 10.9 Pure-code branch is local-only

The experimental pure-code branch (this document's branch) does not
ship to `main`. Merging would require bringing across the Modal
deployment, the model env vars, and an API contract change for
`ratings` on `generation.finished`. The trade is real: pure-code
engines play in ~50ms per move instead of seconds and use ~5 LLM
calls per generation instead of ~1000 — a 200× cost reduction at
play time — but the merge work was deferred for the hackathon.

\newpage

# 11. References

- Engine Protocol: `backend/darwin/engines/base.py`
- DB schema: `backend/darwin/storage/models.py`
- WebSocket events: `backend/darwin/api/websocket.py`,
  `frontend/src/api/events.ts`
- Generation loop: `backend/darwin/orchestration/generation.py`
- Strategist: `backend/darwin/agents/strategist.py`
- Builder + validator: `backend/darwin/agents/builder.py`
- Adversary: `backend/darwin/agents/adversary.py`
- Fixer: `backend/darwin/agents/fixer.py`
- Tournament referee: `backend/darwin/tournament/referee.py`
- Tournament runner: `backend/darwin/tournament/runner.py`
- Selection: `backend/darwin/tournament/selection.py`
- Elo: `backend/darwin/tournament/elo.py`
- LLM provider dispatch: `backend/darwin/llm.py`
- Configuration: `backend/darwin/config.py`
- REST routes: `backend/darwin/api/routes.py`
- Engine registry: `backend/darwin/engines/registry.py`
- Baseline engine: `backend/darwin/engines/baseline.py`
- Per-person plans: `plans/`
- Pure-code experiment write-up: `docs/experiment-pure-code.md`
