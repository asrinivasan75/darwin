# Follow-up 3 — Champion resume in `run_generation_task`

**Owner:** TBD  •  **Branch:** `followup/champion-resume`

## Goal

Make API-triggered generations actually advance the lineage. `backend/darwin/orchestration/generation.py::run_generation_task` hard-codes baseline as the champion for every API-triggered run:

```python
async def run_generation_task() -> None:
    from darwin.engines.baseline import engine as baseline
    ...
    await run_generation(baseline, next_number)
```

So if gen 2 promotes `gen2-evaluation-f00df6`, hitting `POST /api/generations/run` again expecting gen 3 restarts from baseline-v0 instead of continuing the lineage. The whole "self-improving" premise breaks after one generation. The CLI path (`darwin.orchestration.run.main`) already threads the champion forward correctly; the API path does not.

## Scope

- `backend/darwin/orchestration/generation.py` (the loader branch)
- `backend/darwin/engines/registry.py` (if `load_engine` needs broadening)
- `backend/tests/test_orchestration.py` (new file if needed)

## Frozen contracts touched

None. `Engine` Protocol in `engines/base.py` and `EngineRow` in `storage/models.py` stay untouched.

## Deliverables

### 1. Load the reigning champion from the DB

Right after deriving `next_number`, look up the previous `GenerationRow.champion_after`, find its `EngineRow` by name, and load the engine via the registry:

```python
with get_session() as s:
    last_gen = s.exec(
        select(GenerationRow).order_by(GenerationRow.number.desc())
    ).first()
    if last_gen is None:
        # First run — start from baseline
        from darwin.engines.baseline import engine as champion
    else:
        row = s.exec(
            select(EngineRow).where(EngineRow.name == last_gen.champion_after)
        ).one()
        champion = load_engine(row.code_path)
    next_number = (last_gen.number + 1) if last_gen else 1

await run_generation(champion, next_number)
```

### 2. Make `load_engine` handle both forms

`EngineRow.code_path` currently holds two shapes:

- Baseline: dotted module name `darwin.engines.baseline`
- Generated engines: filesystem path `backend/darwin/engines/generated/gen2_*.py`

Ensure `darwin.engines.registry.load_engine(...)` accepts either form. If it doesn't today, add a branch: if the string starts with `/` or ends with `.py`, treat as file path; otherwise treat as dotted module.

### 3. Persist the promoted engine row

`run_generation` already inserts an `EngineRow` when a new champion is promoted. Double-check `code_path` is the **absolute** file path, not a relative one — otherwise step 2's `load_engine` call from a different cwd will fail.

## Acceptance criteria

- [ ] Running gen 2 then gen 3 back-to-back via the API (no process restart) shows gen 3's champion_before == gen 2's champion_after, not "baseline-v0".
- [ ] `load_engine` accepts both dotted module names and filesystem paths.
- [ ] `EngineRow.code_path` stores absolute paths for generated engines.
- [ ] New test `test_orchestration.py::test_run_generation_task_resumes_from_last_champion` covers it (mock the strategist + builder, assert the second call's champion is the first call's winner).

## Risks & mitigations

- **Relative-path regression.** A future caller running from a different cwd will silently fail to load the candidate. Mitigation: `code_path` is stored absolute at insert time, and `load_engine` rejects relative paths explicitly.
- **Stale `EngineRow` after `state.cleared`.** Wiping the DB drops every row except baseline, so the next run correctly starts from baseline rather than a dangling lineage.

## Status

Merged. `run_generation_task` in [backend/darwin/orchestration/generation.py](../backend/darwin/orchestration/generation.py) reconstructs incumbents from the DB (top-2 by win rate), and [backend/darwin/engines/registry.py](../backend/darwin/engines/registry.py) handles both dotted-module and absolute-path forms.
