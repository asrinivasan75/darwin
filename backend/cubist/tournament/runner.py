"""Round-robin scheduler with parallel game execution.

Runs all pairings concurrently via asyncio.gather. Both colors per pairing.
Returns a scoreboard. See plans/person-b-tournament.md.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Awaitable, Callable

from cubist.engines.base import Engine
from cubist.tournament.referee import GameResult, play_game

log = logging.getLogger("cubist.tournament")

EventCb = Callable[[dict], Awaitable[None]] | None


@dataclass
class Standings:
    scores: dict[str, float]  # engine name -> total points
    games: list[GameResult]


async def round_robin(
    engines: list[Engine],
    games_per_pairing: int,
    time_per_move_ms: int,
    on_event: EventCb = None,
) -> Standings:
    if games_per_pairing < 0:
        raise ValueError("games_per_pairing must be non-negative")

    tasks: list[asyncio.Task[GameResult]] = []
    pairings: list[tuple[Engine, Engine]] = []
    game_id = 0
    for i, white in enumerate(engines):
        for j, black in enumerate(engines):
            if i == j:
                continue
            for _ in range(games_per_pairing):
                tasks.append(
                    asyncio.create_task(
                        play_game(white, black, time_per_move_ms, on_event, game_id)
                    )
                )
                pairings.append((white, black))
                game_id += 1

    raw = await asyncio.gather(*tasks, return_exceptions=True)
    results: list[GameResult] = []
    for (white, black), entry in zip(pairings, raw):
        if isinstance(entry, BaseException):
            # play_game has its own top-level guard, so this branch should
            # be unreachable. Keep the safety net anyway: a raised exception
            # here would otherwise abort the whole generation.
            log.exception(
                "round_robin: play_game raised white=%s black=%s err=%r",
                white.name,
                black.name,
                entry,
            )
            results.append(
                GameResult(
                    white=white.name,
                    black=black.name,
                    result="0-1",
                    termination="error",
                    pgn="",
                    error=f"runner: {type(entry).__name__}: {entry}",
                )
            )
        else:
            results.append(entry)

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
