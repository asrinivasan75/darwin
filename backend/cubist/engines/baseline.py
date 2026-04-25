"""Generation-0 LLM chess engine."""

import chess

from cubist.config import settings
from cubist.engines.base import BaseLLMEngine

SYSTEM = (
    "You are a chess engine. Reply with EXACTLY ONE legal move in standard "
    "algebraic notation (SAN). No prose, no explanation, just the move."
)


async def _complete_text(model: str, system: str, user: str, max_tokens: int) -> str:
    from cubist.llm import complete_text

    return await complete_text(model, system, user, max_tokens=max_tokens)


class BaselineEngine(BaseLLMEngine):
    def __init__(self) -> None:
        super().__init__(name="baseline-v0", generation=0, lineage=[])

    async def select_move(
        self,
        board: chess.Board,
        time_remaining_ms: int,
    ) -> chess.Move:
        legal = [board.san(move) for move in board.legal_moves]
        user = (
            f"FEN: {board.fen()}\n"
            f"Move number: {board.fullmove_number}\n"
            f"Side to move: {'White' if board.turn else 'Black'}\n"
            f"Legal moves: {', '.join(legal)}\n"
            f"Your move:"
        )
        try:
            text = await _complete_text(settings.player_model, SYSTEM, user, max_tokens=10)
            san = text.strip().split()[0] if text.strip() else ""
            return board.parse_san(san)
        except Exception:
            return next(iter(board.legal_moves))


engine = BaselineEngine()
