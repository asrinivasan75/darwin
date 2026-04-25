"""The big loop: one generation end-to-end.

strategist -> 2 builders (parallel) -> validator -> tournament -> selection
-> persist + emit events.
"""

import asyncio
import inspect
import json
import logging
from datetime import datetime

log = logging.getLogger("cubist.orchestration")

from cubist.agents.builder import build_engine, validate_engine
from cubist.agents.strategist import propose_questions
from cubist.api.websocket import bus
from cubist.config import settings
from cubist.engines.base import Engine
from cubist.engines.registry import load_engine
from cubist.storage.db import get_session
from cubist.storage.models import EngineRow, GameRow, GenerationRow
from cubist.tournament.runner import round_robin
from cubist.tournament.selection import select_champion


def _read_source(engine: Engine) -> str:
    return inspect.getsource(type(engine))


async def run_generation(champion: Engine, generation_number: int) -> Engine:
    await bus.emit(
        {
            "type": "generation.started",
            "number": generation_number,
            "champion": champion.name,
        }
    )

    questions = await propose_questions(_read_source(champion), [])
    for q in questions:
        await bus.emit(
            {
                "type": "strategist.question",
                "index": q.index,
                "category": q.category,
                "text": q.text,
            }
        )

    paths = await asyncio.gather(
        *[
            build_engine(_read_source(champion), champion.name, generation_number, q)
            for q in questions
        ],
        return_exceptions=True,
    )

    candidates: list[Engine] = []
    for q, p in zip(questions, paths):
        if isinstance(p, Exception):
            log.error(
                "build_engine raised q=%d category=%s err=%r",
                q.index, q.category, p,
            )
            await bus.emit(
                {
                    "type": "builder.completed",
                    "question_index": q.index,
                    "engine_name": "-",
                    "ok": False,
                    "error": str(p),
                }
            )
            continue
        ok, err = await validate_engine(p)
        name = p.stem
        if not ok:
            log.error(
                "validator rejected q=%d category=%s engine=%s reason=%r",
                q.index, q.category, name, err,
            )
        else:
            log.info("validator accepted q=%d category=%s engine=%s", q.index, q.category, name)
        await bus.emit(
            {
                "type": "builder.completed",
                "question_index": q.index,
                "engine_name": name,
                "ok": ok,
                "error": err,
            }
        )
        if ok:
            candidates.append(load_engine(str(p)))

    # If every candidate fell through, ``round_robin([champion])`` will
    # schedule zero games (i==j filter). Surface this loudly so the
    # operator knows why the dashboard goes silent after builder events.
    if not candidates:
        log.error(
            "generation %d has 0 candidates — every builder failed or "
            "was rejected by the validator. Tournament will schedule 0 games. "
            "Check engines/generated/_failures/ for raw model responses.",
            generation_number,
        )

    standings = await round_robin(
        [champion, *candidates],
        games_per_pairing=settings.games_per_pairing,
        time_per_move_ms=settings.time_per_move_ms,
        on_event=bus.emit,
    )
    new_champion, promoted = select_champion(standings, champion, candidates)

    with get_session() as s:
        gen_row = GenerationRow(
            number=generation_number,
            champion_before=champion.name,
            champion_after=new_champion.name,
            strategist_questions_json=json.dumps(
                [{"category": q.category, "text": q.text} for q in questions]
            ),
            finished_at=datetime.utcnow(),
        )
        s.add(gen_row)
        for g in standings.games:
            s.add(
                GameRow(
                    generation=generation_number,
                    white_name=g.white,
                    black_name=g.black,
                    pgn=g.pgn,
                    result=g.result,
                    termination=g.termination,
                )
            )
        if promoted:
            existing = s.get(EngineRow, new_champion.name)
            if existing is None:
                from sqlmodel import select

                existing = s.exec(
                    select(EngineRow).where(EngineRow.name == new_champion.name)
                ).first()
            if existing is None:
                s.add(
                    EngineRow(
                        name=new_champion.name,
                        generation=generation_number,
                        parent_name=champion.name,
                        code_path=f"cubist.engines.generated.{new_champion.name}",
                    )
                )
        s.commit()

    await bus.emit(
        {
            "type": "generation.finished",
            "number": generation_number,
            "new_champion": new_champion.name,
            "elo_delta": 0.0,
            "promoted": promoted,
        }
    )
    return new_champion


async def run_generation_task() -> None:
    """Triggered by the API. Loads current champion from DB, runs one generation.

    Wrapped in a top-level try/except so a crash in any sub-step is logged
    with a full traceback AND surfaced to the UI via a terminal
    ``generation.finished`` event (``promoted=False``). Without this the
    asyncio Task just dies, the dashboard hangs on "running", and we have
    to grep honcho's stdout to find out what went wrong.
    """
    from cubist.engines.baseline import engine as baseline

    with get_session() as s:
        from sqlmodel import select

        last = s.exec(
            select(GenerationRow).order_by(GenerationRow.number.desc())
        ).first()
        next_number = (last.number + 1) if last else 1

    log.info("run_generation_task starting generation=%d", next_number)
    try:
        await run_generation(baseline, next_number)
        log.info("run_generation_task finished generation=%d", next_number)
    except Exception:
        log.exception("run_generation_task crashed generation=%d", next_number)
        await bus.emit(
            {
                "type": "generation.finished",
                "number": next_number,
                "new_champion": baseline.name,
                "elo_delta": 0.0,
                "promoted": False,
            }
        )
