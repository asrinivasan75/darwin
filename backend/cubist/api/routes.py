"""REST routes. Person E owns.

Thin read-only adapters over the three SQLModel tables plus two write
endpoints that drive the orchestrator. The live move/event stream is
served separately over `/ws` (see `server.py`).

Routes:
    GET  /api/engines            all engines ever seen, oldest-first implicit
    GET  /api/generations        all generations, ordered by number
    GET  /api/games[?gen=N]      games, optionally filtered to one generation
    POST /api/generations/run    cancel any in-flight generation; start fresh
    POST /api/generations/stop   cancel any in-flight generation; no replacement
    POST /api/state/clear        cancel + wipe DB + delete generated engines
"""

import logging
from pathlib import Path

from fastapi import APIRouter
from sqlmodel import delete, select

from cubist.storage.db import get_session
from cubist.storage.models import EngineRow, GameRow, GenerationRow

log = logging.getLogger("cubist.api")
router = APIRouter()


@router.get("/engines")
def list_engines():
    """Return every engine row (baseline + every promoted/candidate engine)."""
    with get_session() as s:
        return s.exec(select(EngineRow)).all()


@router.get("/generations")
def list_generations():
    """Return every generation ordered by generation number."""
    with get_session() as s:
        return s.exec(select(GenerationRow).order_by(GenerationRow.number)).all()


@router.get("/games")
def list_games(gen: int | None = None):
    """Return games. If `gen` is provided, filter to that generation."""
    with get_session() as s:
        q = select(GameRow)
        if gen is not None:
            q = q.where(GameRow.generation == gen)
        return s.exec(q).all()


@router.post("/generations/run")
async def run():
    """Cancel any in-flight generation and start a fresh one.

    Two rapid Run-button clicks no longer race each other — the second
    request cancels the first task (emitting ``generation.cancelled``)
    before kicking off its own. The task runs inside the FastAPI event
    loop so its emitted events reach the same `bus` the `/ws` clients
    are subscribed to. We do not wait for completion — a single
    generation can take minutes.
    """
    from cubist.orchestration.generation import start_or_replace_generation_task

    await start_or_replace_generation_task()
    return {"started": True}


@router.post("/generations/stop")
async def stop():
    """Cancel the in-flight generation, if any. Idempotent.

    Returns ``{"stopped": True}`` if a task was actually cancelled,
    ``{"stopped": False}`` if there was nothing running. The frontend
    fires this on Stop-button click and on ``beforeunload`` (via
    ``navigator.sendBeacon``) so closing/reloading the dashboard tab
    doesn't leave a generation churning the LLM in the background.
    """
    from cubist.orchestration.generation import stop_current_generation_task

    stopped = await stop_current_generation_task()
    return {"stopped": stopped}


@router.post("/state/clear")
async def clear_state():
    """Wipe all engines/games/generations and on-disk generated engines.

    Stronger than ``/stop``: stops the current generation if any, then
    deletes every row in the three tables, every ``.py`` under
    ``engines/generated/`` (these are the LLM-built candidates), and
    every ``.txt`` under ``engines/generated/_failures/``. Finally
    broadcasts a ``state.cleared`` event so connected dashboards drop
    their accumulated event log and show an empty UI matching the now-
    empty backend state.

    The baseline engine is NOT touched — it lives in
    ``cubist.engines.baseline`` and is loaded directly by the
    orchestrator, not from the engines table.
    """
    from cubist.api.websocket import bus
    from cubist.orchestration.generation import stop_current_generation_task

    stopped = await stop_current_generation_task()

    with get_session() as s:
        # Order doesn't matter — there are no FKs between these tables.
        s.exec(delete(GameRow))
        s.exec(delete(EngineRow))
        s.exec(delete(GenerationRow))
        s.commit()

    # Hardcoded relative to this file's location — same convention as
    # builder.py's GENERATED_DIR. Avoids importing builder just for the
    # constant (would pull in the LLM SDK chain on a clear).
    generated_dir = Path(__file__).parent.parent / "engines" / "generated"
    failed_dir = generated_dir / "_failures"
    deleted_engines = 0
    deleted_failures = 0
    if generated_dir.exists():
        for p in generated_dir.glob("*.py"):
            # Skip __init__.py if present — that's a package marker, not
            # a generated candidate.
            if p.name == "__init__.py":
                continue
            p.unlink()
            deleted_engines += 1
    if failed_dir.exists():
        for p in failed_dir.glob("*.txt"):
            p.unlink()
            deleted_failures += 1

    log.info(
        "state cleared: stopped_running=%s deleted_engines=%d deleted_failures=%d",
        stopped, deleted_engines, deleted_failures,
    )
    await bus.emit({"type": "state.cleared"})
    return {
        "cleared": True,
        "stopped_running": stopped,
        "deleted_engine_files": deleted_engines,
        "deleted_failure_files": deleted_failures,
    }
