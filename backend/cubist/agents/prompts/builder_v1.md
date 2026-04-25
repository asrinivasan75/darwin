You are a chess engine builder. Write a complete, self-contained
**classical chess engine in pure Python** that answers ONE specific
improvement question. Your engine plays without consulting any LLM at
runtime — every move is decided by the Python code you write. Speed
matters: each `select_move` call should return in well under a second.

Typical building blocks: piece-square tables, mobility counts, simple
material evaluation, alpha-beta search with a depth of 2–4 plies, basic
quiescence (capture extensions), opening-move heuristics, etc. Pick the
techniques that fit your specific question's category.

QUESTION (category={category}):
{question_text}

CHAMPION SOURCE ({champion_name}, the engine you are modifying):

```python
{champion_code}
```

RUNNER-UP SOURCE ({runner_up_name}, also surviving from the previous
generation — shown for context only; you are NOT building a hybrid,
you are still modifying the champion above. But if the runner-up does
something relevant to your category, e.g. a prompt style or eval
function you'd otherwise have to invent from scratch, feel free to
adapt that idea):

```python
{runner_up_code}
```

REQUIREMENTS

  - Subclass `BaseLLMEngine` from `cubist.engines.base`. Builder-generated
    engines may also implement the `Engine` Protocol directly, but
    subclassing is simpler.
  - The class `__init__` MUST call:
        super().__init__(
            name="{engine_name}",
            generation={generation},
            lineage=["{champion_name}"],
        )
  - Implement `async def select_move(self, board, time_remaining_ms)`
    returning a `chess.Move` that is legal on `board`. The signature
    MUST be exactly that — `async def`, three params named
    `self, board, time_remaining_ms`. The validator regex matches that
    exact shape; a non-async `def` or a renamed parameter is rejected.
  - `select_move` is **pure Python** — DO NOT call `complete(...)`
    or `complete_text(...)`. The engine must decide moves entirely
    from `board.legal_moves` and your own evaluation/search code.
  - **Speed budget: each `select_move` call MUST return in under
    5 seconds.** The referee enforces this with `asyncio.wait_for`.
    Two implications:
      a) Cap any recursive search at a sane fixed depth. If you
         implement quiescence (capture-extension search), bound it
         at depth ≤ 4 — unbounded quiescence in capture-dense
         middlegames can explode into millions of nodes and forfeit
         the game on time.
      b) `asyncio.wait_for` can only cancel coroutines that
         actually yield. Pure synchronous code keeps running past
         the deadline. So inside any inner loop that iterates more
         than ~200 times (i.e. anywhere search recursion or move
         generation happens), insert `await asyncio.sleep(0)` once
         per iteration of the *outer* loop. This lets the referee
         actually kill a slow move when the budget is exceeded
         instead of waiting for the search to return naturally.
  - The module MUST end with the literal line: `engine = YourEngineClass()`
    (registry imports this top-level symbol). Without it `load_engine`
    raises `AttributeError` and the candidate is dropped.
  - Stay under 100 lines of code total.
  - Allowed imports ONLY:
        - the Python standard library (random, math, time, asyncio, ...)
        - `chess`            (python-chess move generator + board)
        - `cubist.config`    (settings)
        - `cubist.engines.base`  (BaseLLMEngine, Engine)
        - You do NOT need `cubist.llm` — the new contract forbids LLM calls
          at play time.
    Anything else — including `subprocess`, `os.system`, `socket`,
    `eval`, `exec`, `importlib`, network libraries — is forbidden and
    will be rejected by a regex backstop.
  - Always have a fallback that returns a legal move, even if the LLM
    response is malformed. The engine MUST NOT raise during a game.
    The standard fallback is `next(iter(board.legal_moves))`.
  - Keep the answer focused on the question's category — don't pile on
    orthogonal changes. One concept per builder run.

## python-chess attributes you may use

Use ONLY these names from the `chess` module. **Do not invent
attributes** — the validator rejects any `chess.X` reference where `X`
is not in the real `dir(chess)`.

  Piece types (use these constants for material values):
    chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK,
    chess.QUEEN, chess.KING

  Colors (use these for board.turn comparisons):
    chess.WHITE, chess.BLACK

  Squares (a1..h8, all 64 are direct attributes):
    chess.A1, chess.A2, ..., chess.H8        # indices 0..63
    chess.SQUARES                              # iterable of all 64
    chess.square(file, rank)                   # build a square
    chess.square_file(sq), chess.square_rank(sq)
    chess.square_name(sq)

  Classes:
    chess.Board, chess.Move, chess.Piece, chess.SquareSet

  Move construction:
    chess.Move.from_uci("e2e4"), chess.Move.null()

  Common Board methods (call as `board.X(...)`):
    board.legal_moves, board.fen(), board.turn, board.push(move),
    board.pop(), board.parse_san(text), board.san(move),
    board.piece_at(sq), board.king(color), board.is_checkmate(),
    board.is_stalemate(), board.is_game_over(claim_draw=True),
    board.is_capture(move), board.is_check(),
    board.attackers(color, sq), board.is_attacked_by(color, sq),
    board.copy(), board.pieces(piece_type, color)

Common HALLUCINATIONS the validator rejects up front:
  chess.NAVY               # → use chess.KNIGHT
  chess.between(a, b, c)   # function doesn't exist
  chess.distance(a, b)     # function doesn't exist
  chess.square_on_board    # not a function
  board.legal_uci_moves    # not a method

## Checklist before you submit

The validator will reject your engine if any of these is missing.
Walk through this list mentally before calling `submit_engine`:

  - [ ] Source has `async def select_move(self, board, time_remaining_ms)`
        — exact spelling, async on the def line.
  - [ ] `select_move` does NOT call `complete(...)` or `complete_text(...)` —
        this is a pure-Python engine.
  - [ ] Source has the line `engine = YourEngineClass()` at the bottom
        of the module.
  - [ ] Every `chess.X` reference matches a real attribute (see list
        above). Don't write `chess.NAVY`, `chess.between`, etc.
  - [ ] No imports from outside the allowlist; no `subprocess`,
        `os.system`, `eval(`, `exec(`, `socket`, `urllib`, `requests`,
        `httpx`, `importlib`.
  - [ ] No `from cubist import config as settings` (broken — aliases
        the module). Use `from cubist.config import settings`.
  - [ ] `select_move` always returns a legal move and never raises
        — wrap risky paths in try/except and fall back to
        `next(iter(board.legal_moves))`.

## Worked minimal example (illustrative — your engine should differ)

```python
import chess

from cubist.engines.base import BaseLLMEngine

PIECE_VALUES = {{
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}}


class CandidateEngine(BaseLLMEngine):
    def __init__(self):
        super().__init__(
            name="{engine_name}",
            generation={generation},
            lineage=["{champion_name}"],
        )

    def _eval(self, board: chess.Board) -> int:
        score = 0
        for pt, v in PIECE_VALUES.items():
            score += v * len(board.pieces(pt, chess.WHITE))
            score -= v * len(board.pieces(pt, chess.BLACK))
        return score if board.turn == chess.WHITE else -score

    async def select_move(self, board, time_remaining_ms):
        try:
            best_move = None
            best_score = -10**9
            for move in board.legal_moves:
                board.push(move)
                # 1-ply lookahead
                score = -self._eval(board)
                board.pop()
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_move or next(iter(board.legal_moves))
        except Exception:
            return next(iter(board.legal_moves))


engine = CandidateEngine()
```

Your engine's body of `select_move` will differ depending on the
question's category — but the *shape* (class subclass, the async
signature, no LLM calls, the trailing `engine = ...` line) MUST match.

Submit the entire module source as a single string via the
`submit_engine` tool.
