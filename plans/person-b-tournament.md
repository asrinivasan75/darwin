# Person B — Tournament & Referee

**Branch:** `feat/tournament`

## Goal

Make engines fight each other. This is the load-bearing piece for the whole pipeline — if a single tournament can't finish in under 90 minutes the demo can't show 3 generations. Parallelism is non-negotiable.

## Scope

Files owned:

```
backend/darwin/tournament/
├── referee.py       # plays one game between two engines — IMPLEMENT
├── runner.py        # parallel round-robin — IMPLEMENT
├── elo.py           # rating updates — IMPLEMENT
└── selection.py     # anti-regression gate — IMPLEMENT
scripts/eval_match.py            # head-to-head N-game match — IMPLEMENT
backend/tests/test_referee.py    # CREATE
backend/tests/test_runner.py     # CREATE
backend/tests/test_elo.py        # CREATE
backend/tests/test_selection.py  # CREATE
```

Read first:

1. `docs/proposal.pdf` — focus on §3 (Approach) and §9 (Risks).
2. `plans/README.md` — merge order.
3. `backend/darwin/engines/base.py` — the Engine Protocol you call into.
4. `backend/darwin/engines/random_engine.py` — your day-one test partner before A merges baseline.
5. `backend/darwin/api/websocket.py` — the event payloads you'll emit through E's bus.

Already done for you:

- All four module files have stubs with the right function signatures and dataclass returns.
- `RandomEngine` exists in `darwin.engines.random_engine` — use it for every test so you don't burn API budget.
- The `GameResult` and `Standings` dataclasses are already defined.

## Frozen contracts touched

- **Engine Protocol** — `backend/darwin/engines/base.py`. Consumed via `Engine.select_move`.
- **WebSocket events** — `backend/darwin/api/websocket.py`. You emit `game.move` / `game.finished` shapes via the `on_event` callback, but never modify the schema.

## Deliverables

### Step 1 — Branch and verify

```bash
git checkout -b feat/tournament
cd backend && uv sync
uv run python -c "from darwin.engines.random_engine import RandomEngine; print('ok')"
```

### Step 2 — Implement Elo first (it's tiny)

`backend/darwin/tournament/elo.py`:
```python
def update_elo(rating_a: float, rating_b: float, score_a: float, k: float = 32.0) -> tuple[float, float]:
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * ((1 - score_a) - expected_b)
    return new_a, new_b
```

Test (`tests/test_elo.py`):
```python
from darwin.tournament.elo import update_elo

def test_draw_at_equal_rating_unchanged():
    a, b = update_elo(1500, 1500, 0.5)
    assert abs(a - 1500) < 1e-9 and abs(b - 1500) < 1e-9

def test_win_increases_winner():
    a, b = update_elo(1500, 1500, 1.0)
    assert a > 1500 and b < 1500
```

### Step 3 — Implement `play_game`

`backend/darwin/tournament/referee.py`:
```python
import asyncio
import io
from dataclasses import dataclass
from typing import Awaitable, Callable

import chess
import chess.pgn

from darwin.config import settings
from darwin.engines.base import Engine

EventCb = Callable[[dict], Awaitable[None]] | None


@dataclass
class GameResult:
    white: str
    black: str
    result: str
    termination: str
    pgn: str


def _to_pgn(board: chess.Board, white: str, black: str, result: str) -> str:
    game = chess.pgn.Game()
    game.headers["White"] = white
    game.headers["Black"] = black
    game.headers["Result"] = result
    node = game
    for move in board.move_stack:
        node = node.add_variation(move)
    out = io.StringIO()
    print(game, file=out)
    return out.getvalue()


def _loss(loser_is_white: bool, white: str, black: str, board: chess.Board, term: str) -> GameResult:
    result = "0-1" if loser_is_white else "1-0"
    return GameResult(white, black, result, term, _to_pgn(board, white, black, result))


async def play_game(
    white: Engine,
    black: Engine,
    time_per_move_ms: int,
    on_event: EventCb = None,
    game_id: int = 0,
) -> GameResult:
    board = chess.Board()
    timeout_s = time_per_move_ms / 1000 + 5

    while not board.is_game_over(claim_draw=True):
        if board.fullmove_number > settings.max_moves_per_game:
            result = "1/2-1/2"
            return GameResult(white.name, black.name, result, "max_moves",
                              _to_pgn(board, white.name, black.name, result))

        engine = white if board.turn == chess.WHITE else black
        is_white = board.turn == chess.WHITE
        try:
            move = await asyncio.wait_for(
                engine.select_move(board, time_per_move_ms), timeout=timeout_s
            )
            if move not in board.legal_moves:
                return _loss(is_white, white.name, black.name, board, "illegal_move")
        except asyncio.TimeoutError:
            return _loss(is_white, white.name, black.name, board, "time")
        except Exception:
            return _loss(is_white, white.name, black.name, board, "error")

        san = board.san(move)
        board.push(move)
        if on_event:
            await on_event({
                "type": "game.move", "game_id": game_id, "fen": board.fen(),
                "san": san, "white": white.name, "black": black.name, "ply": board.ply(),
            })

    result = board.result(claim_draw=True)
    term = "checkmate" if board.is_checkmate() else "stalemate" if board.is_stalemate() else "draw"
    pgn = _to_pgn(board, white.name, black.name, result)
    if on_event:
        await on_event({
            "type": "game.finished", "game_id": game_id, "result": result,
            "termination": term, "pgn": pgn, "white": white.name, "black": black.name,
        })
    return GameResult(white.name, black.name, result, term, pgn)
```

Test (`tests/test_referee.py`):
```python
import pytest
from darwin.engines.random_engine import RandomEngine
from darwin.tournament.referee import play_game

@pytest.mark.asyncio
async def test_two_random_engines_finish():
    a = RandomEngine(seed=1); a.name = "a"
    b = RandomEngine(seed=2); b.name = "b"
    r = await play_game(a, b, time_per_move_ms=1000)
    assert r.result in ("1-0", "0-1", "1/2-1/2")
    assert r.pgn.startswith("[Event")
```

### Step 4 — Implement `round_robin`

`backend/darwin/tournament/runner.py`:
```python
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Awaitable, Callable

from darwin.engines.base import Engine
from darwin.tournament.referee import GameResult, play_game

EventCb = Callable[[dict], Awaitable[None]] | None


@dataclass
class Standings:
    scores: dict[str, float]
    games: list[GameResult]


async def round_robin(
    engines: list[Engine],
    games_per_pairing: int,
    time_per_move_ms: int,
    on_event: EventCb = None,
) -> Standings:
    tasks = []
    game_id = 0
    for i, white in enumerate(engines):
        for j, black in enumerate(engines):
            if i == j:
                continue
            for _ in range(games_per_pairing):
                tasks.append(play_game(white, black, time_per_move_ms, on_event, game_id))
                game_id += 1
    results = await asyncio.gather(*tasks)
    scores: dict[str, float] = defaultdict(float)
    for r in results:
        if r.result == "1-0":
            scores[r.white] += 1
        elif r.result == "0-1":
            scores[r.black] += 1
        else:
            scores[r.white] += 0.5
            scores[r.black] += 0.5
    for e in engines:
        scores.setdefault(e.name, 0.0)
    return Standings(scores=dict(scores), games=results)
```

Test (`tests/test_runner.py`):
```python
import pytest
from darwin.engines.random_engine import RandomEngine
from darwin.tournament.runner import round_robin

@pytest.mark.asyncio
async def test_round_robin_4_engines():
    engines = [RandomEngine(seed=i) for i in range(4)]
    for i, e in enumerate(engines):
        e.name = f"r{i}"
    s = await round_robin(engines, games_per_pairing=1, time_per_move_ms=1000)
    expected_games = 4 * 3  # 4*3 ordered pairs
    assert len(s.games) == expected_games
    assert sum(s.scores.values()) == expected_games  # 1 point per game
```

### Step 5 — Anti-regression gate

`backend/darwin/tournament/selection.py`:
```python
from darwin.engines.base import Engine
from darwin.tournament.runner import Standings


def _h2h_score(games, a: str, b: str) -> tuple[float, int]:
    """Return (a's score vs b, number of games)."""
    score, n = 0.0, 0
    for g in games:
        if {g.white, g.black} != {a, b}:
            continue
        n += 1
        if g.result == "1-0":
            score += 1.0 if g.white == a else 0.0
        elif g.result == "0-1":
            score += 1.0 if g.black == a else 0.0
        else:
            score += 0.5
    return score, n


def select_champion(
    standings: Standings, incumbent: Engine, candidates: list[Engine],
) -> tuple[Engine, bool]:
    if not candidates:
        return incumbent, False
    top = max(candidates, key=lambda e: standings.scores.get(e.name, 0.0))
    score, n = _h2h_score(standings.games, top.name, incumbent.name)
    if n == 0:
        return incumbent, False
    if score > n / 2:
        return top, True
    return incumbent, False
```

Test (`tests/test_selection.py`):
```python
from dataclasses import dataclass
from darwin.tournament.runner import Standings
from darwin.tournament.referee import GameResult
from darwin.tournament.selection import select_champion


class FakeEngine:
    def __init__(self, name): self.name = name; self.generation = 0; self.lineage = []
    async def select_move(self, b, t): pass


def test_anti_regression_keeps_incumbent_when_h2h_lost():
    inc = FakeEngine("inc"); cand = FakeEngine("cand")
    games = [
        GameResult("inc", "cand", "1-0", "checkmate", ""),
        GameResult("cand", "inc", "0-1", "checkmate", ""),
    ]
    standings = Standings(scores={"inc": 1.0, "cand": 5.0}, games=games)  # cand top by total
    new, promoted = select_champion(standings, inc, [cand])
    assert new is inc and promoted is False
```

### Step 6 — `scripts/eval_match.py`

CLI for the demo headline number. Takes `--white-module`, `--black-module`, `--n`. Plays N games (alternating colors), prints a results table. Use `darwin.engines.registry.load_engine` to resolve the engines.

### Step 7 — Open the PR

```bash
git add -A && git commit -m "feat: tournament referee + runner + Elo + selection"
git push -u origin feat/tournament
gh pr create --title "Tournament" --body "Closes plan B."
```

### Integration points

- **Person A** provides `Engine`, `BaselineEngine`, `RandomEngine`. You depend only on `RandomEngine` until they merge.
- **Person E** passes a real `bus.emit` as the `on_event` argument to `round_robin`. Your code already accepts it as optional, so it Just Works.
- **Person C**'s validator imports `play_game` for its smoke test.

## Acceptance criteria

- [ ] All four test files pass: `uv run pytest tests/`.
- [ ] `eval_match.py` runs a 4-game match between two `RandomEngine`s and prints a clean table.
- [ ] PR opened, then merged after review.

## Risks & mitigations

- **One slow game blocks `gather`.** The `asyncio.wait_for` per move + `max_moves_per_game` cap are both required.
- **Both colors per pairing** — the `i == j` skip plus iterating both `(i,j)` and `(j,i)` gives this for free.
- **Draws are common with weak engines.** `claim_draw=True` and the 50-move rule are handled by `python-chess` for free if you call `is_game_over(claim_draw=True)`.
- **Don't print inside `play_game`.** Use `on_event` instead. Print in tests via pytest's `-s`.

## Status

Merged. Note: post-hackathon the head-to-head selection gate was replaced by win-rate-based `select_top_n` (see [README.md](../README.md)) and the round-robin gained a `max_parallel_games` semaphore (see [followup-1-tournament-concurrency.md](followup-1-tournament-concurrency.md)).
