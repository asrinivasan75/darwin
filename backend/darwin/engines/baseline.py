"""Generation-0 local chess engine.

This baseline intentionally does not call an LLM or any external API. It gives
the tournament and builder pipelines a deterministic, cheap incumbent built on
ordinary chess heuristics: terminal detection, material, mobility, and a small
two-ply alpha-beta search.
"""

from __future__ import annotations

import math

import chess

from darwin.engines.base import BaseLLMEngine

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
MATE_SCORE = 1_000_000
SEARCH_DEPTH = 2


class BaselineEngine(BaseLLMEngine):
    def __init__(self) -> None:
        super().__init__(name="baseline-v0", generation=0, lineage=[])

    async def select_move(
        self,
        board: chess.Board,
        time_remaining_ms: int,
    ) -> chess.Move:
        """Return the best legal move found by a small deterministic search."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("cannot select a move from a position with no legal moves")

        maximizing = board.turn == chess.WHITE
        best_move = legal_moves[0]
        best_score = -math.inf if maximizing else math.inf

        for move in self._ordered_moves(board):
            board.push(move)
            score = self._search(board, SEARCH_DEPTH - 1, -math.inf, math.inf)
            board.pop()
            if maximizing and score > best_score:
                best_score = score
                best_move = move
            elif not maximizing and score < best_score:
                best_score = score
                best_move = move

        return best_move

    def _search(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        if depth == 0:
            return self._evaluate(board)
        if board.is_insufficient_material() or board.halfmove_clock >= 100:
            return 0

        moves = self._ordered_moves(board)
        if not moves:
            if board.is_check():
                return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
            return 0

        if board.turn == chess.WHITE:
            value = -math.inf
            for move in moves:
                board.push(move)
                value = max(value, self._search(board, depth - 1, alpha, beta))
                board.pop()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = math.inf
        for move in moves:
            board.push(move)
            value = min(value, self._search(board, depth - 1, alpha, beta))
            board.pop()
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _evaluate(self, board: chess.Board) -> float:
        """Evaluate from White's perspective."""
        if board.is_checkmate():
            return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
        if board.is_insufficient_material() or board.halfmove_clock >= 100:
            return 0

        material = 0
        for piece_type, value in PIECE_VALUES.items():
            material += len(board.pieces(piece_type, chess.WHITE)) * value
            material -= len(board.pieces(piece_type, chess.BLACK)) * value

        saved_turn = board.turn
        board.turn = chess.WHITE
        white_mobility = board.legal_moves.count()
        board.turn = chess.BLACK
        black_mobility = board.legal_moves.count()
        board.turn = saved_turn

        return material + 2 * (white_mobility - black_mobility)

    def _ordered_moves(self, board: chess.Board) -> list[chess.Move]:
        def priority(move: chess.Move) -> tuple[int, int, int]:
            captured = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            capture_gain = 0
            if captured is not None and attacker is not None:
                capture_gain = PIECE_VALUES[captured.piece_type] - PIECE_VALUES[attacker.piece_type]
            promotion = PIECE_VALUES.get(move.promotion, 0)
            return (int(board.gives_check(move)), promotion, capture_gain)

        return sorted(board.legal_moves, key=priority, reverse=True)


engine = BaselineEngine()
