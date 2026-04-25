"""CLI entrypoint: run N generations end-to-end.

Usage:
    uv run python -m cubist.orchestration.run --generations 3

Each generation runs the full loop: strategist -> 5 builders -> validator ->
round-robin tournament -> anti-regression selection -> DB persist -> WS emit.
The DB is initialized on startup if the schema is missing. The baseline
engine (from Person A) is used as the starting champion unless a later
generation is already persisted.
"""

import argparse
import asyncio

from cubist.engines.baseline import engine as baseline
from cubist.orchestration.generation import run_generation
from cubist.storage.db import init_db


async def main(generations: int) -> None:
    """Run `generations` rounds back-to-back, threading the new champion forward."""
    init_db()
    champion = baseline
    for n in range(1, generations + 1):
        champion = await run_generation(champion, n)
    print(f"final champion: {champion.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run N generations of Cubist.")
    parser.add_argument(
        "--generations",
        type=int,
        default=1,
        help="Number of generations to run (default: 1).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.generations))
