import asyncio

import chess

from cubist.engines.baseline import engine


async def main() -> None:
    board = chess.Board()
    while not board.is_game_over() and board.fullmove_number < 10:
        move = await engine.select_move(board, 10000)
        print(board.san(move))
        board.push(move)
    print("done:", board.result())


asyncio.run(main())
