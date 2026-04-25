"""Round-robin scheduler with bounded parallel game execution.

Local mode caps concurrency with ``settings.max_parallel_games`` so a
slow LLM provider can't make every game time out at once. Modal mode
fans every game out to its own container (real OS-level parallel,
GIL-free) which is the right choice for CPU-bound pure-code engines.
Toggle via ``settings.tournament_backend`` (``local`` | ``modal``).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Awaitable, Callable

from cubist.config import settings
from cubist.engines.base import Engine
from cubist.tournament.referee import GameResult, play_game

log = logging.getLogger("cubist.tournament.runner")

EventCb = Callable[[dict], Awaitable[None]] | None


async def warm_modal_pool(n: int = 4) -> None:
    """Bring up ``n`` warm Modal containers ahead of the tournament.

    No-op when ``tournament_backend != 'modal'``. Best-effort: any Modal
    error (auth, network, autoscaler API rename) is logged and swallowed
    so a warm-pool failure can never abort a generation. Returns
    immediately — Modal spins up the containers in the background while
    we run the strategist + builder + smoke phases.
    """
    if settings.tournament_backend != "modal":
        return
    try:
        import modal
        f = modal.Function.from_name(
            "cubist-tournament", "play_game_remote"
        )
        await f.update_autoscaler.aio(min_containers=n)
        log.info("modal warm-pool set to min_containers=%d", n)
    except Exception as e:
        log.warning(
            "modal warm-pool call failed: %r — proceeding without pre-warm", e
        )


async def cool_modal_pool() -> None:
    """Drain the warm pool back to 0 idle containers. No-op for local."""
    if settings.tournament_backend != "modal":
        return
    try:
        import modal
        f = modal.Function.from_name(
            "cubist-tournament", "play_game_remote"
        )
        await f.update_autoscaler.aio(min_containers=0)
        log.info("modal warm-pool cooled (min_containers=0)")
    except Exception as e:
        log.warning("modal cool-down call failed: %r", e)


@dataclass
class Standings:
    scores: dict[str, float]  # engine name -> total points
    games: list[GameResult]


def _build_pairings(engines: list[Engine], games_per_pairing: int):
    """Enumerate every (white, black, game_id) tuple. White=row, Black=col,
    skip i==j (engine vs itself), expand by games_per_pairing."""
    pairings = []
    game_id = 0
    for i, white in enumerate(engines):
        for j, black in enumerate(engines):
            if i == j:
                continue
            for _ in range(games_per_pairing):
                pairings.append((white, black, game_id))
                game_id += 1
    return pairings


def _tally(engines: list[Engine], results: list[GameResult]) -> Standings:
    scores: dict[str, float] = defaultdict(float)
    for result in results:
        if result.result == "1-0":
            scores[result.white] += 1.0
        elif result.result == "0-1":
            scores[result.black] += 1.0
        else:
            scores[result.white] += 0.5
            scores[result.black] += 0.5
    for engine in engines:
        scores.setdefault(engine.name, 0.0)
    return Standings(scores=dict(scores), games=results)


async def _round_robin_local(
    pairings: list[tuple[Engine, Engine, int]],
    time_per_move_ms: int,
    on_event: EventCb,
) -> list[GameResult]:
    sem = asyncio.Semaphore(settings.max_parallel_games)

    async def _guarded(white: Engine, black: Engine, game_id: int) -> GameResult:
        async with sem:
            return await play_game(
                white, black, time_per_move_ms, on_event, game_id
            )

    return await asyncio.gather(
        *[_guarded(w, b, gid) for (w, b, gid) in pairings]
    )


async def _round_robin_modal(
    pairings: list[tuple[Engine, Engine, int]],
    time_per_move_ms: int,
    on_event: EventCb,
) -> list[GameResult]:
    """Dispatch every game to a Modal container, with live event streaming.

    Each container pushes ``game.move`` / ``game.finished`` events into a
    shared ``modal.Queue`` as they happen. We run a drainer task that
    pulls from the queue and re-emits via ``on_event`` in parallel with
    the games. Net effect: dashboard sees moves in real time, just like
    the local backend, but the actual chess CPU runs on Modal's hardware.
    """
    import modal

    # Look up deployed objects. Importing them from modal_runner would
    # only give us un-hydrated references because the local process
    # isn't inside an ``app.run()`` context.
    play_game_remote = modal.Function.from_name(
        "cubist-tournament", "play_game_remote"
    )
    events_queue = modal.Queue.from_name(
        "cubist-events", create_if_missing=True
    )

    # Drain any stale events from a previous run (orphaned by cancel,
    # crash, or a slow consumer). We don't want gen N's leftovers to
    # bleed into gen N+1's dashboard.
    drained_stale = 0
    while True:
        try:
            await asyncio.wait_for(events_queue.get.aio(), timeout=0.05)
            drained_stale += 1
        except (asyncio.TimeoutError, modal.exception.NotFoundError):
            break
        except Exception:
            break
    if drained_stale:
        log.info("modal-tournament drained %d stale events from queue", drained_stale)

    # Each game needs the FULL MODULE source of both engines. See note
    # in _full_source: we read the actual .py file rather than
    # inspect.getsource(type(e)) so imports survive the trip.
    def _full_source(e: Engine) -> str:
        path = inspect.getsourcefile(type(e))
        if path is None:
            return inspect.getsource(type(e))
        with open(path, "r") as f:
            return f.read()

    args = [
        (
            _full_source(w),
            w.name,
            _full_source(b),
            b.name,
            time_per_move_ms,
            gid,
        )
        for (w, b, gid) in pairings
    ]

    cohort_size = len({w.name for w, _, _ in pairings} | {b.name for _, b, _ in pairings})
    log.info(
        "modal-tournament dispatching %d games (cohort_size=%d) with live event streaming",
        len(args), cohort_size,
    )

    # Drainer: pull events from the Modal queue and re-emit on our bus
    # in real time. Stops when the cancel signal fires after all games
    # complete + a small tail-drain window.
    cancel_drainer = asyncio.Event()

    async def drain_events() -> None:
        while not cancel_drainer.is_set():
            try:
                # Pull up to BATCH events at once. Containers send them
                # in batches of 5 via ``put_many``, so reading one at a
                # time forces the queue server to do as many round-trips
                # as we'd save by batching.
                batch = await asyncio.wait_for(
                    events_queue.get_many.aio(10), timeout=0.2
                )
                if on_event is not None:
                    for ev in batch:
                        await on_event(ev)
            except asyncio.TimeoutError:
                # No events right now — loop and check cancel flag.
                continue
            except Exception as e:
                log.warning("modal-tournament drainer error: %r", e)
                await asyncio.sleep(0.1)

    drainer = asyncio.create_task(drain_events())

    try:
        results: list[GameResult] = []
        # starmap.aio yields one result per game as it completes. The
        # function returns just the GameResult fields now — events
        # already flowed via the queue while the game was in flight.
        async for ret in play_game_remote.starmap.aio(args):
            results.append(
                GameResult(
                    white=ret["white"],
                    black=ret["black"],
                    result=ret["result"],
                    termination=ret["termination"],
                    pgn=ret["pgn"],
                )
            )
    finally:
        # Tail-drain: a couple of events may still be in flight from
        # the last container. Give them a beat to land before stopping.
        await asyncio.sleep(0.5)
        cancel_drainer.set()
        try:
            await asyncio.wait_for(drainer, timeout=2.0)
        except asyncio.TimeoutError:
            drainer.cancel()
            try:
                await drainer
            except (asyncio.CancelledError, Exception):
                pass

    return results


async def round_robin(
    engines: list[Engine],
    games_per_pairing: int,
    time_per_move_ms: int,
    on_event: EventCb = None,
) -> Standings:
    if games_per_pairing < 0:
        raise ValueError("games_per_pairing must be non-negative")
    if settings.max_parallel_games < 1:
        raise ValueError("max_parallel_games must be at least 1")

    pairings = _build_pairings(engines, games_per_pairing)

    backend = settings.tournament_backend
    if backend == "modal":
        results = await _round_robin_modal(pairings, time_per_move_ms, on_event)
    elif backend == "local":
        results = await _round_robin_local(pairings, time_per_move_ms, on_event)
    else:
        raise ValueError(
            f"unknown tournament_backend={backend!r}; expected 'local' or 'modal'"
        )

    return _tally(engines, results)
