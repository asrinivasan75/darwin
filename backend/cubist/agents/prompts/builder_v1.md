You are a chess engine builder. Modify the champion source below to
answer ONE specific improvement question.

QUESTION (category={category}):
{question_text}

CHAMPION SOURCE:

```python
{champion_code}
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
  - `select_move` MUST call `complete_text(...)` (or `complete(...)`)
    at least once — the whole point of an LLM engine is to consult the
    LLM. An engine that only reads `board` and returns a heuristic move
    is not a valid candidate; the validator rejects it.
  - The module MUST end with the literal line: `engine = YourEngineClass()`
    (registry imports this top-level symbol). Without it `load_engine`
    raises `AttributeError` and the candidate is dropped.
  - Stay under 100 lines of code total.
  - Allowed imports ONLY:
        - the Python standard library (random, math, time, asyncio, ...)
        - `chess`            (python-chess move generator + board)
        - `cubist.config`    (settings)
        - `cubist.engines.base`  (BaseLLMEngine, Engine)
        - `cubist.llm`       (complete, complete_text)
    Anything else — including `subprocess`, `os.system`, `socket`,
    `eval`, `exec`, `importlib`, network libraries — is forbidden and
    will be rejected by a regex backstop.
  - Always have a fallback that returns a legal move, even if the LLM
    response is malformed. The engine MUST NOT raise during a game.
    The standard fallback is `next(iter(board.legal_moves))`.
  - Keep the answer focused on the question's category — don't pile on
    orthogonal changes. One concept per builder run.

## Checklist before you submit

The validator will reject your engine if any of these is missing.
Walk through this list mentally before calling `submit_engine`:

  - [ ] Source has `async def select_move(self, board, time_remaining_ms)`
        — exact spelling, async on the def line.
  - [ ] Source body of `select_move` calls `complete_text(...)` (or
        `complete(...)`) at least once.
  - [ ] Source has the line `engine = YourEngineClass()` at the bottom
        of the module.
  - [ ] No imports from outside the allowlist; no `subprocess`,
        `os.system`, `eval(`, `exec(`, `socket`, `urllib`, `requests`,
        `httpx`, `importlib`.
  - [ ] No `from cubist import config as settings` (broken — aliases
        the module). Use `from cubist.config import settings`.
  - [ ] `select_move` has a fallback that returns a legal move when
        the LLM response can't be parsed (`next(iter(board.legal_moves))`
        is the standard one).

## Worked minimal example (illustrative — your engine should differ)

```python
import chess

from cubist.engines.base import BaseLLMEngine
from cubist.llm import complete_text
from cubist.config import settings


class CandidateEngine(BaseLLMEngine):
    def __init__(self):
        super().__init__(
            name="{engine_name}",
            generation={generation},
            lineage=["{champion_name}"],
        )

    async def select_move(self, board, time_remaining_ms):
        legal = [board.san(m) for m in board.legal_moves]
        try:
            text = await complete_text(
                settings.player_model,
                "You are a chess engine. Reply with one SAN move.",
                f"FEN: {{board.fen()}}\nLegal: {{', '.join(legal)}}\nMove:",
                max_tokens=10,
            )
            return board.parse_san(text.strip().split()[0])
        except Exception:
            return next(iter(board.legal_moves))


engine = CandidateEngine()
```

Your engine's body of `select_move` will differ depending on the
question's category — but the *shape* (class subclass, the async
signature, the LLM call, the trailing `engine = ...` line) MUST match.

Submit the entire module source as a single string via the
`submit_engine` tool.
