"""Replay persisted generations over the WebSocket bus.

Demo-day insurance: if the live API is rate-limited or otherwise flaky
during the demo, we replay a pre-recorded run from SQLite through the same
event bus the frontend already listens to. From the audience's perspective,
this looks identical to a live generation.

It reads `GenerationRow` and `GameRow` from the database and re-emits the
same event sequence the orchestrator would have emitted — `generation.started`,
one `strategist.question` per recorded question, one `game.finished` per
recorded game, and a final `generation.finished`. Per-move streaming is
not reconstructed (we never persisted move-by-move FENs); the frontend
already tolerates a game arriving via `game.finished` alone.

Usage:
    # Replay every persisted generation, in order.
    uv run python scripts/replay.py

    # Replay a specific generation.
    uv run python scripts/replay.py --gen 1

    # Override inter-event pacing (seconds).
    uv run python scripts/replay.py --question-delay 1.5 --game-delay 3.0

The script only emits events; it does NOT write to the database. Safe to
run as many times as needed without polluting state.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from sqlmodel import select  # noqa: E402

from cubist.api.websocket import bus  # noqa: E402
from cubist.storage.db import get_session  # noqa: E402
from cubist.storage.models import GameRow, GenerationRow  # noqa: E402


async def replay_generation(
    gen: GenerationRow,
    games: list[GameRow],
    question_delay: float,
    game_delay: float,
) -> None:
    """Re-emit one generation's worth of events with realistic pacing."""
    await bus.emit(
        {
            "type": "generation.started",
            "number": gen.number,
            "champion": gen.champion_before,
        }
    )
    await asyncio.sleep(question_delay)

    questions = json.loads(gen.strategist_questions_json or "[]")
    for i, q in enumerate(questions):
        await bus.emit(
            {
                "type": "strategist.question",
                "index": i,
                "category": q.get("category", "prompt"),
                "text": q.get("text", ""),
            }
        )
        await asyncio.sleep(question_delay)

    for g in games:
        await bus.emit(
            {
                "type": "game.finished",
                "game_id": g.id or 0,
                "result": g.result,
                "termination": g.termination,
                "pgn": g.pgn,
                "white": g.white_name,
                "black": g.black_name,
            }
        )
        await asyncio.sleep(game_delay)

    promoted = gen.champion_after != gen.champion_before
    await bus.emit(
        {
            "type": "generation.finished",
            "number": gen.number,
            "new_champion": gen.champion_after,
            "elo_delta": 0.0,
            "promoted": promoted,
        }
    )


async def main(gen_filter: int | None, question_delay: float, game_delay: float) -> None:
    """Replay one or all persisted generations over the event bus."""
    with get_session() as s:
        q = select(GenerationRow).order_by(GenerationRow.number)
        if gen_filter is not None:
            q = q.where(GenerationRow.number == gen_filter)
        generations = list(s.exec(q).all())
        if not generations:
            print("no generations to replay; run `scripts/run_generation.py` first")
            return

        for gen in generations:
            games = list(
                s.exec(
                    select(GameRow).where(GameRow.generation == gen.number)
                ).all()
            )
            print(
                f"replaying generation {gen.number}: "
                f"{gen.champion_before} -> {gen.champion_after} ({len(games)} games)"
            )
            await replay_generation(gen, games, question_delay, game_delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay persisted generations over WS.")
    parser.add_argument(
        "--gen",
        type=int,
        default=None,
        help="Replay only this generation number. Omit to replay all.",
    )
    parser.add_argument(
        "--question-delay",
        type=float,
        default=1.5,
        help="Seconds to wait between strategist questions (default: 1.5).",
    )
    parser.add_argument(
        "--game-delay",
        type=float,
        default=3.0,
        help="Seconds to wait between game.finished events (default: 3.0).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.gen, args.question_delay, args.game_delay))
