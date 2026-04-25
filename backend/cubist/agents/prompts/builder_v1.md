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
    returning a `chess.Move` that is legal on `board`.
  - The module MUST end with the literal line: `engine = YourEngineClass()`
    (the registry imports this top-level symbol).
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

EXACT API SURFACE — do NOT invent kwargs or helpers beyond these.

```python
# cubist.engines.base
class BaseLLMEngine:
    name: str
    generation: int
    lineage: list[str]

    def __init__(self, name: str, generation: int, lineage: list[str] | None = None): ...
    async def select_move(self, board: chess.Board, time_remaining_ms: int) -> chess.Move: ...

# cubist.llm — these are the ONLY signatures. No `temperature`, no
# `top_p`, no `system_prompt=`, no streaming, no sync variants.
async def complete(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 256,
    tools: list[dict] | None = None,
) -> list[TextBlock | ToolUseBlock]:
    """Returns a list of content blocks. Each has `.type` ("text" or
    "tool_use"). TextBlock has `.text`; ToolUseBlock has `.name` and
    `.input` (dict)."""

async def complete_text(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 256,
) -> str:
    """Plain-text wrapper. Returns "" if the model produced no text."""

# cubist.config
from cubist.config import settings
settings.player_model      # str — model id to pass as `model=`
settings.builder_model     # str
settings.strategist_model  # str
```

Calling `complete(...)` or `complete_text(...)` with any other keyword
argument (e.g. `temperature=`, `top_p=`, `prompt=`) will raise
TypeError at runtime and crash the engine — your code will be discarded.

Submit the entire module source as a single string via the
`submit_engine` tool.
