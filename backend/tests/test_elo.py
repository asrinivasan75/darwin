import pytest

from cubist.tournament.elo import update_elo


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
