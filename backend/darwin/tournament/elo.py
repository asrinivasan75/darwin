"""Elo rating updates."""

from collections.abc import Iterable
from typing import Protocol


class RatedGame(Protocol):
    white: str
    black: str
    result: str


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A against player B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k: float = 32.0,
) -> tuple[float, float]:
    """Standard Elo: score_a in {0, 0.5, 1}. Returns (new_a, new_b)."""
    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1 - expected_a
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * ((1 - score_a) - expected_b)
    return new_a, new_b


def update_ratings_for_games(
    ratings: dict[str, float],
    games: Iterable[RatedGame],
    k: float = 32.0,
) -> dict[str, float]:
    """Apply one rating-period Elo update for a completed set of games.

    Expected scores are computed from the ratings at the start of the
    tournament, then actual-vs-expected deltas are accumulated across all
    games. This is order-independent, unlike applying ``update_elo`` to
    each game as async results arrive.
    """
    start = dict(ratings)
    deltas = {name: 0.0 for name in start}

    for game in games:
        white = game.white
        black = game.black
        if white not in start:
            start[white] = 1500.0
            deltas[white] = 0.0
        if black not in start:
            start[black] = 1500.0
            deltas[black] = 0.0

        if game.result == "1-0":
            score_white = 1.0
        elif game.result == "0-1":
            score_white = 0.0
        else:
            score_white = 0.5

        expected_white = expected_score(start[white], start[black])
        deltas[white] += k * (score_white - expected_white)
        deltas[black] += k * ((1 - score_white) - (1 - expected_white))

    return {name: start[name] + delta for name, delta in deltas.items()}
