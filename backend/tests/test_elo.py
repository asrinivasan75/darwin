import pytest

from darwin.tournament.elo import expected_score, update_elo, update_ratings_for_games


class Game:
    def __init__(self, white: str, black: str, result: str):
        self.white = white
        self.black = black
        self.result = result


def test_expected_score_equal_ratings_is_half():
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_400_point_favorite_is_about_91_percent():
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)


def test_draw_at_equal_rating_unchanged():
    a, b = update_elo(1500, 1500, 0.5)
    assert a == pytest.approx(1500)
    assert b == pytest.approx(1500)


def test_win_increases_winner_and_decreases_loser():
    a, b = update_elo(1500, 1500, 1.0)
    assert a > 1500
    assert b < 1500


def test_rating_delta_is_zero_sum():
    a, b = update_elo(1600, 1400, 0.0)
    assert (a + b) == pytest.approx(3000)


def test_rating_period_update_is_order_independent():
    ratings = {"a": 1600.0, "b": 1500.0, "c": 1400.0}
    games = [
        Game("a", "b", "1-0"),
        Game("b", "c", "1/2-1/2"),
        Game("c", "a", "0-1"),
    ]

    forward = update_ratings_for_games(ratings, games)
    backward = update_ratings_for_games(ratings, list(reversed(games)))

    assert forward == pytest.approx(backward)


def test_rating_period_update_accumulates_against_starting_ratings():
    ratings = {"a": 1500.0, "b": 1500.0}
    games = [
        Game("a", "b", "1-0"),
        Game("b", "a", "0-1"),
    ]

    updated = update_ratings_for_games(ratings, games)

    assert updated["a"] == pytest.approx(1532.0)
    assert updated["b"] == pytest.approx(1468.0)
    assert sum(updated.values()) == pytest.approx(sum(ratings.values()))
