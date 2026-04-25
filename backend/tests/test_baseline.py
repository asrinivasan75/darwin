import inspect

import chess
import pytest

import cubist.engines.baseline as baseline
from cubist.engines.baseline import engine


@pytest.mark.asyncio
async def test_baseline_returns_legal_move():
    board = chess.Board()
    move = await engine.select_move(board, 10000)
    assert move in board.legal_moves


@pytest.mark.asyncio
async def test_baseline_finds_mate_in_one():
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 w - - 0 1")
    move = await engine.select_move(board, 10000)
    board.push(move)
    assert board.is_checkmate()


@pytest.mark.asyncio
async def test_baseline_prefers_winning_material():
    board = chess.Board("4k3/8/8/8/8/8/4q3/4R1K1 w - - 0 1")
    move = await engine.select_move(board, 10000)
    assert board.piece_at(move.to_square).piece_type == chess.QUEEN


def test_baseline_has_no_llm_or_anthropic_dependency():
    source = inspect.getsource(baseline)
    assert "cubist.llm" not in source
    assert "anthropic" not in source
    assert "complete_text" not in source
