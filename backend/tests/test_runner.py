import asyncio

import pytest

from cubist.config import settings
from cubist.engines.random_engine import RandomEngine
from cubist.tournament.referee import GameResult
from cubist.tournament.runner import round_robin


@pytest.fixture(autouse=True)
def _force_local_backend(monkeypatch):
    """Tests in this file exercise the local asyncio path. The .env may
    set ``TOURNAMENT_BACKEND=modal`` for the dev server; force-override
    to ``local`` so the tests don't try to dispatch to Modal."""
    monkeypatch.setattr(settings, "tournament_backend", "local")


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


@pytest.mark.asyncio
async def test_round_robin_caps_in_flight_games(monkeypatch):
    current = 0
    max_seen = 0

    async def fake_play_game(white, black, time_per_move_ms, on_event=None, game_id=0):
        nonlocal current, max_seen
        current += 1
        max_seen = max(max_seen, current)
        await asyncio.sleep(0.01)
        current -= 1
        return GameResult(white.name, black.name, "1/2-1/2", "draw", "")

    import cubist.tournament.runner as runner

    monkeypatch.setattr(runner, "play_game", fake_play_game)
    monkeypatch.setattr(runner.settings, "max_parallel_games", 2)

    engines = [RandomEngine(seed=i) for i in range(4)]
    for i, engine in enumerate(engines):
        engine.name = f"r{i}"

    standings = await round_robin(engines, games_per_pairing=1, time_per_move_ms=1000)

    assert len(standings.games) == 12
    assert max_seen == 2
