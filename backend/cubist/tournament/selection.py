"""Anti-regression gate.

The top scorer in the tournament becomes the new champion ONLY IF it beats
the prior champion in their head-to-head subset. Otherwise champion persists.
"""

from cubist.engines.base import Engine
from cubist.tournament.runner import Standings


def _h2h_score(games, a: str, b: str) -> tuple[float, int]:
    score = 0.0
    games_played = 0
    for game in games:
        if {game.white, game.black} != {a, b}:
            continue
        games_played += 1
        if game.result == "1-0":
            score += 1.0 if game.white == a else 0.0
        elif game.result == "0-1":
            score += 1.0 if game.black == a else 0.0
        else:
            score += 0.5
    return score, games_played


def select_champion(
    standings: Standings,
    incumbent: Engine,
    candidates: list[Engine],
) -> tuple[Engine, bool]:
    """Returns (new_champion, promoted). promoted=False means incumbent persisted."""
    if not candidates:
        return incumbent, False

    top = max(candidates, key=lambda engine: standings.scores.get(engine.name, 0.0))
    score, games_played = _h2h_score(standings.games, top.name, incumbent.name)
    if games_played == 0:
        return incumbent, False
    if score > games_played / 2:
        return top, True
    return incumbent, False
