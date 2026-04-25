"""The big loop: one generation end-to-end.

strategist -> 2 builders (parallel) -> validator -> tournament -> selection
-> persist + emit events.
"""

import asyncio
import inspect
import json
import logging
from datetime import datetime

from cubist.agents.builder import build_engine, validate_engine
from cubist.agents.strategist import propose_questions
from cubist.api.websocket import bus
from cubist.config import settings
from cubist.engines.base import Engine
from cubist.engines.registry import load_engine
from cubist.storage.db import get_session
from cubist.storage.models import EngineRow, GameRow, GenerationRow
from cubist.tournament.elo import update_elo
from cubist.tournament.runner import round_robin
from cubist.tournament.selection import select_top_n

log = logging.getLogger("cubist.orchestration")


def _read_source(engine: Engine) -> str:
    return inspect.getsource(type(engine))


async def run_generation(
    incumbents: list[Engine], generation_number: int
) -> list[Engine]:
    """Run one generation: strategist → builders → tournament → selection.

    ``incumbents[0]`` is the *primary* champion — its source is shown to
    the strategist and used as the seed for builders, and it's the
    ``champion_before`` field that gets persisted. Any extra incumbents
    (``incumbents[1:]``) are runners-up from the previous generation and
    they participate in the round-robin alongside the new candidates,
    but they are not seeds for new builds. Returns the list of top-2
    engines coming out of this generation, for the orchestrator to seed
    the next run.
    """
    if not incumbents:
        raise ValueError("run_generation requires at least one incumbent")
    primary = incumbents[0]
    runner_up = incumbents[1] if len(incumbents) > 1 else None

    await bus.emit(
        {
            "type": "generation.started",
            "number": generation_number,
            "champion": primary.name,
        }
    )

    # Both the strategist and the builder see the champion AND the
    # runner-up. The champion is the seed they're modifying; the runner-
    # up is shown as context — a strong alternative design from the same
    # gen, useful for the LLM to compare approaches without forcing a
    # hybrid.
    primary_src = _read_source(primary)
    runner_up_src = _read_source(runner_up) if runner_up else None
    runner_up_name = runner_up.name if runner_up else None

    questions = await propose_questions(
        primary_src, [], runner_up_code=runner_up_src
    )
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
            build_engine(
                primary_src,
                primary.name,
                generation_number,
                q,
                runner_up_code=runner_up_src,
                runner_up_name=runner_up_name,
            )
            for q in questions
        ],
        return_exceptions=True,
    )

    candidates: list[Engine] = []
    candidate_paths: dict[str, str] = {}
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
        # ``p.stem`` is the safe filename (underscored: gen1_book_abc).
        # ``engine.name`` and game.finished.white/black use the hyphenated
        # form (gen1-book-abc). Emit the hyphenated form so the Bracket
        # join lines up.
        name = p.stem.replace("_", "-")
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
            eng = load_engine(str(p))
            candidate_paths[eng.name] = str(p.resolve())
            candidates.append(eng)

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

    # The tournament cohort is every incumbent + every accepted candidate.
    # With 2 incumbents and up to 4 candidates that's 6 engines and
    # 6*5*games_per_pairing = 60 games concurrently — the
    # max_parallel_games semaphore caps actual concurrency to keep
    # Gemini quotas in line.
    cohort = [*incumbents, *candidates]
    standings = await round_robin(
        cohort,
        games_per_pairing=settings.games_per_pairing,
        time_per_move_ms=settings.time_per_move_ms,
        on_event=bus.emit,
    )

    # Score-based selection with random tiebreak. The primary incumbent
    # is the "anti-regression baseline" — anything else in the cohort is
    # a candidate from the selector's perspective (so the runner-up
    # incumbent gets re-evaluated each gen, just like the new builds).
    others = [e for e in cohort if e.name != primary.name]
    top = select_top_n(standings, primary, others, n=2)
    new_champion = top[0]
    promoted = new_champion.name != primary.name

    # ── Elo update ──────────────────────────────────────────────────────
    # Standard chess Elo, K=32 — the typical hackathon-friendly value
    # (USCF-style). Each engine in the cohort gets one rating update per
    # game it played. New candidates start at 1500 (the chess midpoint);
    # baseline-v0 is also seeded at 1500 by ``scripts/seed_baseline.py``.
    cohort_names = [e.name for e in cohort]
    with get_session() as s:
        from sqlmodel import select as _select

        existing_rows = s.exec(
            _select(EngineRow).where(EngineRow.name.in_(cohort_names))
        ).all()
        ratings: dict[str, float] = {row.name: row.elo for row in existing_rows}

    for name in cohort_names:
        ratings.setdefault(name, 1500.0)

    pre_ratings = dict(ratings)
    for game in standings.games:
        if game.result == "1-0":
            score_a = 1.0
        elif game.result == "0-1":
            score_a = 0.0
        else:
            score_a = 0.5
        new_w, new_b = update_elo(
            ratings[game.white],
            ratings[game.black],
            score_a,
        )
        ratings[game.white] = new_w
        ratings[game.black] = new_b

    # Champion's Elo delta = post-tournament Elo of the new champion
    # minus its pre-tournament Elo. For a fresh promotion this includes
    # the candidate's whole gain (it walked in at 1500 and walked out
    # at whatever it earned). For a retention this is just the seasoned
    # champion's drift.
    elo_delta = ratings[new_champion.name] - pre_ratings[new_champion.name]
    log.info(
        "elo updates gen=%d primary=%s -> %.1f, new_champion=%s -> %.1f (delta=%.1f)",
        generation_number, primary.name, ratings[primary.name],
        new_champion.name, ratings[new_champion.name], elo_delta,
    )

    with get_session() as s:
        from sqlmodel import select as _select

        gen_row = GenerationRow(
            number=generation_number,
            champion_before=primary.name,
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

        # Persist EngineRows for every accepted candidate (not just the
        # promoted one) so the orchestrator can load the runner-up next
        # gen by name. Dedupe on name — a second appearance under the
        # same name is treated as a no-op.
        for cand in candidates:
            existing = s.exec(
                _select(EngineRow).where(EngineRow.name == cand.name)
            ).first()
            if existing is not None:
                continue
            path = candidate_paths.get(cand.name)
            if path is None:
                continue
            s.add(
                EngineRow(
                    name=cand.name,
                    generation=generation_number,
                    parent_name=primary.name,
                    code_path=path,
                    elo=ratings.get(cand.name, 1500.0),
                )
            )

        # Update Elo for engines that already have a row (baseline,
        # previously-promoted champions, prior runners-up).
        for row in s.exec(
            _select(EngineRow).where(EngineRow.name.in_(cohort_names))
        ).all():
            row.elo = ratings[row.name]
            s.add(row)

        s.commit()

    await bus.emit(
        {
            "type": "generation.finished",
            "number": generation_number,
            "new_champion": new_champion.name,
            "elo_delta": round(elo_delta, 1),
            "promoted": promoted,
            # Full cohort Elo so the frontend can plot every engine,
            # not just the champion. Round to 1 decimal so we don't
            # ship 16-digit floats to a chart that displays integers.
            "ratings": {n: round(r, 1) for n, r in ratings.items()},
        }
    )
    return top


async def run_generation_task() -> None:
    """Triggered by the API. Loads current champion from DB, runs one generation.

    Wrapped in a top-level try/except so a crash in any sub-step is logged
    with a full traceback AND surfaced to the UI via a terminal
    ``generation.finished`` event (``promoted=False``). Without this the
    asyncio Task just dies, the dashboard hangs on "running", and we have
    to grep honcho's stdout to find out what went wrong.

    ``asyncio.CancelledError`` is treated specially: a ``generation.cancelled``
    event is emitted before the cancellation propagates, so the frontend
    can clear its in-progress panels.
    """
    with get_session() as s:
        from sqlmodel import select

        last = s.exec(
            select(GenerationRow).order_by(GenerationRow.number.desc())
        ).first()
        next_number = (last.number + 1) if last else 1

        if last is None:
            from cubist.engines.baseline import engine as baseline
            incumbents: list[Engine] = [baseline]
        else:
            # Reconstruct previous-gen scores from GameRow so we can
            # carry the top-2 forward. Score = wins + 0.5*draws.
            prev_games = s.exec(
                select(GameRow).where(GameRow.generation == last.number)
            ).all()
            scores: dict[str, float] = {}
            for g in prev_games:
                scores.setdefault(g.white_name, 0.0)
                scores.setdefault(g.black_name, 0.0)
                if g.result == "1-0":
                    scores[g.white_name] += 1.0
                elif g.result == "0-1":
                    scores[g.black_name] += 1.0
                else:
                    scores[g.white_name] += 0.5
                    scores[g.black_name] += 0.5

            # The new champion is non-negotiable — it always seeds the
            # next gen. If there's a runner-up, it joins as a second
            # incumbent. Order by score desc then by name (deterministic
            # tiebreak here is fine — the random tiebreak happened in
            # ``select_top_n`` when promotion was decided).
            ranked_names = sorted(
                scores.keys(),
                key=lambda n: (-scores[n], n),
            )
            # Champion always first.
            top_names: list[str] = [last.champion_after]
            for n in ranked_names:
                if n == last.champion_after:
                    continue
                top_names.append(n)
                if len(top_names) >= 2:
                    break

            incumbents = []
            for name in top_names:
                row = s.exec(
                    select(EngineRow).where(EngineRow.name == name)
                ).first()
                if row is None:
                    if name == "baseline-v0":
                        from cubist.engines.baseline import engine as baseline
                        incumbents.append(baseline)
                    else:
                        log.warning(
                            "skipping incumbent %s — no EngineRow found",
                            name,
                        )
                    continue
                try:
                    incumbents.append(load_engine(row.code_path))
                except Exception as e:
                    log.warning(
                        "skipping incumbent %s — load_engine failed: %r",
                        name, e,
                    )

            # Defensive: if every load failed, fall back to baseline so
            # the generation can still run rather than silently dying.
            if not incumbents:
                log.error(
                    "could not load any incumbent from gen=%d; falling back "
                    "to baseline-v0", last.number,
                )
                from cubist.engines.baseline import engine as baseline
                incumbents = [baseline]

    log.info(
        "run_generation_task starting generation=%d incumbents=%s",
        next_number, [e.name for e in incumbents],
    )
    try:
        await run_generation(incumbents, next_number)
        log.info("run_generation_task finished generation=%d", next_number)
    except asyncio.CancelledError:
        log.warning("run_generation_task cancelled generation=%d", next_number)
        # Emit a terminal event so the dashboard knows to stop showing
        # "Waiting for strategist…" / live-board placeholders. We swallow
        # any error from the bus emit (rare, but the queue may be torn
        # down at server-shutdown time) so cancellation always propagates.
        try:
            await bus.emit(
                {"type": "generation.cancelled", "number": next_number}
            )
        except Exception:  # pragma: no cover — best-effort emit
            pass
        raise
    except Exception:
        log.exception("run_generation_task crashed generation=%d", next_number)
        await bus.emit(
            {
                "type": "generation.finished",
                "number": next_number,
                "new_champion": incumbents[0].name,
                "elo_delta": 0.0,
                "promoted": False,
            }
        )


# ---------------------------------------------------------------------------
# Cancellation API — used by /api/generations/run (replace) and
# /api/generations/stop (cancel only).
# ---------------------------------------------------------------------------

# Module-level handle to the currently-running generation task, if any.
# Single-process / single-worker assumption: this matches the deploy setup
# (uvicorn with one worker; honcho composes one backend process).
_current_task: asyncio.Task[None] | None = None
_task_lock = asyncio.Lock()


async def _await_cancellation(task: asyncio.Task[None]) -> None:
    """Await a cancelled task, swallowing the standard cancellation exception.

    Useful so the caller doesn't need its own try/except around
    ``await task`` after ``task.cancel()``.
    """
    try:
        await task
    except (asyncio.CancelledError, Exception):
        # Cancelled is the expected path; any other exception is already
        # logged by the task's own try/except. No re-raise here — we
        # specifically want to ignore cancellation cleanup errors.
        pass


async def start_or_replace_generation_task() -> None:
    """Cancel any in-flight generation, then start a new one.

    Mounted to ``POST /api/generations/run``. Two clicks of the dashboard's
    Run button no longer race each other — the second click cancels the
    first generation cleanly (emits ``generation.cancelled``) before
    starting the second.
    """
    global _current_task
    async with _task_lock:
        if _current_task is not None and not _current_task.done():
            log.info("preempting in-flight generation task before starting new one")
            _current_task.cancel()
            await _await_cancellation(_current_task)
        _current_task = asyncio.create_task(run_generation_task())


async def stop_current_generation_task() -> bool:
    """Cancel the in-flight generation, if any.

    Mounted to ``POST /api/generations/stop`` and called by the frontend's
    ``beforeunload`` ``sendBeacon`` on page reload. Returns ``True`` if a
    task was cancelled, ``False`` if there was nothing to cancel — the
    endpoint surfaces this as ``{"stopped": bool}`` so the dashboard's
    Stop button can grey out when there's nothing running.
    """
    global _current_task
    if _current_task is None or _current_task.done():
        return False
    _current_task.cancel()
    await _await_cancellation(_current_task)
    return True
