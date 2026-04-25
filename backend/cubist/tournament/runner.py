"""Round-robin scheduler with parallel game execution.

Runs all pairings concurrently via asyncio.gather. Both colors per pairing.
Returns a scoreboard. See plans/person-b-tournament.md.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Awaitable, Callable

from cubist.engines.base import Engine
from cubist.tournament.referee import GameResult, play_game

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

    tasks = []
    game_id = 0
    for i, white in enumerate(engines):
        for j, black in enumerate(engines):
            if i == j:
                continue
            for _ in range(games_per_pairing):
                tasks.append(play_game(white, black, time_per_move_ms, on_event, game_id))
                game_id += 1

    results = await asyncio.gather(*tasks)
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
