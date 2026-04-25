"""REST routes. Person E owns.

Thin read-only adapters over the three SQLModel tables plus one write
endpoint that kicks off a background generation task. The live move/event
stream is served separately over `/ws` (see `server.py`).

Routes:
    GET  /api/engines           all engines ever seen, oldest-first implicit
    GET  /api/generations       all generations, ordered by number
    GET  /api/games[?gen=N]     games, optionally filtered to one generation
    POST /api/generations/run   fire-and-forget: trigger one generation
"""

import asyncio

from fastapi import APIRouter
from sqlmodel import select

from cubist.storage.db import get_session
from cubist.storage.models import EngineRow, GameRow, GenerationRow

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
    """Start one generation in the background and return immediately.

    The task runs inside the FastAPI event loop so its emitted events reach
    the same `bus` the `/ws` clients are subscribed to. We do not wait for
    completion — a single generation can take minutes.
    """
    from cubist.orchestration.generation import run_generation_task

    asyncio.create_task(run_generation_task())
    return {"started": True}
