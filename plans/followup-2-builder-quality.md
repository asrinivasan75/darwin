# Follow-up 2 — Builder quality gate + prompt fix

**Owner:** TBD  •  **Branch:** `followup/builder-quality`

## Goal

Stop the builder from emitting silently-broken engines. Every engine produced so far contains the same import bug:

```python
from darwin import config as settings     # ← imports the MODULE
...
text = await complete_text(settings.player_model, ...)   # AttributeError
```

`darwin.config` is a module; `settings` is the `Settings()` instance *inside* that module. Accessing `settings.player_model` raises `AttributeError`, which the generated engine's `except Exception: return next(iter(board.legal_moves))` swallows — so every generated engine silently skips the LLM and plays the first legal move forever. The validator's smoke game passes (the engine "works") but the engine is useless in a real tournament.

## Scope

- `backend/darwin/agents/prompts/builder_v1.md`
- `backend/darwin/agents/builder.py` (`FORBIDDEN` regex + validator tweaks)
- `backend/tests/test_builder.py`

## Frozen contracts touched

None. `engines/base.py` (Engine Protocol) and the `submit_engine` tool schema stay untouched.

## Deliverables

### 1. Fix the builder prompt

`backend/darwin/agents/prompts/builder_v1.md` needs a concrete, correct import example. Explicit is better than implicit — show the exact line to use and forbid the broken one. Add a section:

```markdown
## Imports (MUST copy verbatim)

Use exactly these imports — no others:

    import chess
    from darwin.engines.base import BaseLLMEngine
    from darwin.llm import complete_text
    from darwin.config import settings

NEVER write `from darwin import config as settings` — that imports the
module, not the settings instance, and will raise AttributeError on
every access.
```

### 2. Tighten the validator

`backend/darwin/agents/builder.py::validate_engine` plays one game vs `RandomEngine`. That's too lenient — an engine whose `select_move` always raises will pass because the fallback (`next(iter(legal))`) returns a legal move.

Change the validator to:

1. Play `N = 6` full moves (3 per side) against `RandomEngine`.
2. On each move, **assert the engine's `select_move` returned without going through its exception handler**. Hard to detect from outside — easier: inspect the engine module source for common anti-patterns before accepting it.

Specifically, reject any source that matches:

```python
BANNED_IMPORTS = [
    r"from\s+darwin\s+import\s+config\s+as\s+settings",
    r"import\s+darwin\.config\s+as\s+settings",
]
```

These are almost always the broken pattern.

### 3. Add a "liveness probe"

After loading the engine, call `select_move` once on the starting position. If the return value is deterministically `next(iter(legal))` (i.e. `Nh3` for White, `Nh6` for Black), that's a red flag — the LLM is being bypassed. Compare to baseline's response on the same FEN; if they differ, the engine is genuinely using the LLM.

## Acceptance criteria

- [ ] Builder prompt explicitly shows `from darwin.config import settings`.
- [ ] At least one regenerated engine from a real Gemini call uses `settings.player_model` correctly.
- [ ] Validator rejects any source containing the banned import patterns.
- [ ] New test `test_builder.py::test_validator_rejects_bad_config_import` covers it.

## Risks & mitigations

- **Liveness probe false positives.** A genuinely-strong opening like `Nh6` (Black) or the LLM legitimately picking the first legal move would trip the probe. Mitigation: compare to baseline's response on the same FEN — only flag when *both* engines' fallbacks would have produced the same move.
- **Banned-import regex churn.** New broken patterns may appear from the LLM. Mitigation: keep the regex list small and append-only; surface validator rejections in logs so operators can spot new failure modes.

## Status

Merged. `FORBIDDEN` regex in [backend/darwin/agents/builder.py](../backend/darwin/agents/builder.py) covers `from darwin import config as settings`; the canonical import line is documented in [backend/darwin/agents/prompts/builder_v1.md](../backend/darwin/agents/prompts/builder_v1.md).
