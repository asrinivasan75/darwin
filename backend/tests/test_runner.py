import asyncio

import pytest

from cubist.engines.random_engine import RandomEngine
from cubist.tournament.runner import round_robin


def test_round_robin_4_engines():
    engines = [RandomEngine(seed=i) for i in range(4)]
    for i, engine in enumerate(engines):
        engine.name = f"r{i}"

    standings = asyncio.run(round_robin(engines, games_per_pairing=1, time_per_move_ms=1000))

    expected_games = 4 * 3
    assert len(standings.games) == expected_games
    assert sum(standings.scores.values()) == expected_games
    assert set(standings.scores) == {"r0", "r1", "r2", "r3"}


def test_round_robin_rejects_negative_games_per_pairing():
    with pytest.raises(ValueError):
        asyncio.run(round_robin([], games_per_pairing=-1, time_per_move_ms=1000))
