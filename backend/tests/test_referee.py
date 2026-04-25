import asyncio

import chess

from cubist.engines.random_engine import RandomEngine
from cubist.tournament.referee import play_game


class IllegalEngine:
    name = "illegal"
    generation = 0
    lineage: list[str] = []

    async def select_move(self, board: chess.Board, time_remaining_ms: int) -> chess.Move:
        return chess.Move.null()


def test_two_random_engines_finish():
    a = RandomEngine(seed=1)
    a.name = "a"
    b = RandomEngine(seed=2)
    b.name = "b"

    result = asyncio.run(play_game(a, b, time_per_move_ms=1000))

    assert result.result in ("1-0", "0-1", "1/2-1/2")
    assert result.pgn.startswith("[Event")


def test_illegal_move_loses_and_emits_finished_event():
    black = RandomEngine(seed=1)
    black.name = "black"
    events = []

    async def on_event(event: dict) -> None:
        events.append(event)

    result = asyncio.run(play_game(IllegalEngine(), black, 1000, on_event=on_event, game_id=7))

    assert result.result == "0-1"
    assert result.termination == "illegal_move"
    assert events[-1]["type"] == "game.finished"
    assert events[-1]["game_id"] == 7
    assert events[-1]["result"] == "0-1"
