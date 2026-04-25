import chess

from cubist.tournament.referee import GameResult
from cubist.tournament.runner import Standings
from cubist.tournament.selection import select_champion


class FakeEngine:
    def __init__(self, name: str) -> None:
        self.name = name
        self.generation = 0
        self.lineage: list[str] = []

    async def select_move(self, board: chess.Board, time_remaining_ms: int) -> chess.Move:
        return next(iter(board.legal_moves))


def test_anti_regression_keeps_incumbent_when_h2h_lost():
    incumbent = FakeEngine("inc")
    candidate = FakeEngine("cand")
    games = [
        GameResult("inc", "cand", "1-0", "checkmate", ""),
        GameResult("cand", "inc", "0-1", "checkmate", ""),
    ]
    standings = Standings(scores={"inc": 1.0, "cand": 5.0}, games=games)

    new_champion, promoted = select_champion(standings, incumbent, [candidate])

    assert new_champion is incumbent
    assert promoted is False


def test_promotes_top_candidate_that_wins_h2h():
    incumbent = FakeEngine("inc")
    candidate = FakeEngine("cand")
    games = [
        GameResult("cand", "inc", "1-0", "checkmate", ""),
        GameResult("inc", "cand", "0-1", "checkmate", ""),
    ]
    standings = Standings(scores={"inc": 0.0, "cand": 2.0}, games=games)

    new_champion, promoted = select_champion(standings, incumbent, [candidate])

    assert new_champion is candidate
    assert promoted is True


def test_tied_h2h_does_not_promote():
    incumbent = FakeEngine("inc")
    candidate = FakeEngine("cand")
    games = [
        GameResult("cand", "inc", "1/2-1/2", "draw", ""),
        GameResult("inc", "cand", "1/2-1/2", "draw", ""),
    ]
    standings = Standings(scores={"inc": 1.0, "cand": 5.0}, games=games)

    new_champion, promoted = select_champion(standings, incumbent, [candidate])

    assert new_champion is incumbent
    assert promoted is False
