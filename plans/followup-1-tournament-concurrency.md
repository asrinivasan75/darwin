# Follow-up 1 — Tournament concurrency + referee observability

**Owner:** TBD  •  **Branch:** `followup/tournament-concurrency`

## Goal

Stop the round-robin from self-immolating under slow LLM providers. A run of gen 2 on Gemini 2.5 Flash produced 12 games, **all** with `termination: "time"` after 0–1 moves: every pairing fired in parallel, individual move calls ballooned past the referee's 25s per-move timeout under Gemini's rate limits, and the "tournament" became pure noise. The winner was whichever engine's bugs fired *after* the opposing engine's rate-limit-induced timeout.

## Scope

- `backend/darwin/config.py` — new `max_parallel_games` setting.
- `backend/darwin/tournament/runner.py` — semaphore around game dispatch.
- `backend/darwin/tournament/referee.py` — error logging + PGN annotation.
- `backend/tests/test_runner.py` — semaphore-cap test.
- `.env.example` — document the new knob.

## Frozen contracts touched

None. `engines/base.py` and `api/websocket.py` event shapes stay untouched.

## Deliverables

### 1. Gate parallelism at the tournament level

`backend/darwin/tournament/runner.py` currently does one big `asyncio.gather(*pairing_coros)`. Add a semaphore so at most `settings.max_parallel_games` run at a time (default **2**, tunable in `.env`). The tournament still takes the same wall-clock time on a fast provider, but stops self-immolating on a slow one.

```python
# config.py
max_parallel_games: int = 2

# runner.py
sem = asyncio.Semaphore(settings.max_parallel_games)
async def _guarded(coro):
    async with sem:
        return await coro
results = await asyncio.gather(*[_guarded(c) for c in pairings])
```

### 2. Surface the real termination reason

`referee.py` already distinguishes `time`, `error`, `illegal_move`, `checkmate`, `stalemate`, `max_moves`. Verify the `error` path carries the exception repr into the log and the PGN comment so post-mortems are possible.

- Log `log.warning("game error: %s vs %s: %r", white.name, black.name, exc)` in the `except Exception:` branch.
- Add the exception class name to the PGN as a comment header (`[ErrorClass "TypeError"]`) so the frontend can show it.

### 3. Optional — per-move time budget

Today `timeout_s = time_per_move_ms/1000 + 5`. That 5-second padding is a magic number. Either justify it in a comment or make it configurable.

## Acceptance criteria

- [ ] `settings.max_parallel_games` exists and defaults to 2.
- [ ] Running gen 2 on the Gemini provider produces at least one **real** game (≥ 10 moves, terminates on `checkmate`/`stalemate`/`max_moves`, not `time`).
- [ ] The `error` termination path logs the exception type + message.
- [ ] A new test in `backend/tests/test_runner.py` asserts the semaphore caps in-flight games.

## Risks & mitigations

- **Default too low.** A default of 2 is conservative for fast providers and may make tournaments needlessly serial. Mitigation: surface the knob in `.env.example` so operators can tune up; the live default ships at 16 (see `.env.example`).
- **Per-move padding magic number.** The 5s timeout buffer can mask slow engines; either justify with a comment or expose as config.

## Status

Merged. `max_parallel_games` is in [backend/darwin/config.py](../backend/darwin/config.py) and [.env.example](../.env.example); the semaphore lives in [backend/darwin/tournament/runner.py](../backend/darwin/tournament/runner.py).
