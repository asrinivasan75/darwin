You are the chess-engine builder, on a second pass. You wrote the
code below for one specific improvement question. An adversarial
reviewer has critiqued it. Your job is to apply the critique:
produce a **revised** complete module that addresses the reviewer's
concerns while keeping the engine focused on the original question's
category.

Apply the critique pragmatically:

  - Fix concrete bugs and forfeit risks the reviewer named.
  - Re-anchor on the question's category if the reviewer says you
    drifted off-scope.
  - DO NOT pile on orthogonal new ideas the reviewer didn't ask for.
    If the original code was solid, return something very close to it.
  - DO NOT delete the category's core technique just because the
    reviewer flagged a related bug — fix the bug, keep the technique.

QUESTION (category={category}):
{question_text}

ADVERSARY CRITIQUE (read this carefully, address it directly):

{critique}

ORIGINAL ENGINE CODE (your previous attempt — revise it, don't
rewrite from scratch):

```python
{original_code}
```

CHAMPION SOURCE ({champion_name}, the engine being modified — for
context, in case you need to re-anchor):

```python
{champion_code}
```

REQUIREMENTS (unchanged from your first pass)

  - Subclass `BaseLLMEngine` from `darwin.engines.base`.
  - The class `__init__` MUST call:
        super().__init__(
            name="{engine_name}",
            generation={generation},
            lineage=["{champion_name}"],
        )
    Keep the same `name`, `generation`, and `lineage` as the original
    code — this is a revision, not a new candidate.
  - Implement `async def select_move(self, board, time_remaining_ms)`
    returning a legal `chess.Move`. Exact signature, async on the
    def line.
  - `select_move` is **pure Python** — no `complete(...)` /
    `complete_text(...)` calls.
  - Each `select_move` must return in under 5 seconds. Cap recursive
    search depths; insert `await asyncio.sleep(0)` in any inner loop
    that iterates more than ~200 times so the referee can cancel.
  - End the module with: `engine = YourEngineClass()`
  - Allowed imports ONLY: stdlib, `chess`, `darwin.config`,
    `darwin.engines.base`. No `subprocess`, `os.system`, `socket`,
    `eval`, `exec`, `importlib`, network libs.
  - Always have a fallback that returns a legal move:
    `next(iter(board.legal_moves))`.
  - Use ONLY real `chess.X` attributes — the same hallucination
    backstops apply (no `chess.NAVY`, `chess.between`, etc.).

Submit the entire revised module source as a single string via the
`submit_engine` tool. Submit ONLY the revised module — no commentary,
no JSON wrapper, no markdown fences inside the tool input.
