import chess
import pytest

import cubist.engines.baseline as baseline
from cubist.engines.baseline import engine


@pytest.mark.asyncio
async def test_baseline_returns_legal_move(monkeypatch):
    async def fake_complete_text(*args, **kwargs):
        return "e4"

    monkeypatch.setattr(baseline, "_complete_text", fake_complete_text)
    board = chess.Board()
    move = await engine.select_move(board, 10000)
    assert move in board.legal_moves


@pytest.mark.asyncio
async def test_baseline_falls_back_on_bad_response(monkeypatch):
    async def fake_complete_text(*args, **kwargs):
        return "not-a-move"

    monkeypatch.setattr(baseline, "_complete_text", fake_complete_text)
    board = chess.Board()
    move = await engine.select_move(board, 10000)
    assert move in board.legal_moves
