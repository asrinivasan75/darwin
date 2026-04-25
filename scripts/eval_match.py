"""Head-to-head N-game match between two engines.

Usage: uv run python scripts/eval_match.py --white baseline-v0 --black gen3-champ --n 10
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from cubist.config import settings  # noqa: E402
from cubist.engines.base import Engine  # noqa: E402
from cubist.engines.random_engine import RandomEngine  # noqa: E402
from cubist.engines.registry import load_engine  # noqa: E402
from cubist.tournament.referee import GameResult, play_game  # noqa: E402

RANDOM_REFS = {"random", "random_engine", "cubist.engines.random_engine"}


def _load_module_engine(module_ref: str) -> Engine:
    if module_ref.endswith(".py") or "/" in module_ref:
        path = Path(module_ref).expanduser().resolve()
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {module_ref}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)

    engine = getattr(module, "engine", None)
    if engine is None:
        raise AttributeError(f"{module_ref} has no top-level `engine` symbol")
    if not isinstance(engine, Engine):
        raise TypeError(f"{module_ref}.engine does not satisfy Engine Protocol")
    return engine


def _load_engine(ref: str, label: str, seed: int) -> Engine:
    if ref in RANDOM_REFS:
        engine = RandomEngine(seed=seed)
        engine.name = f"random-{label}"
        return engine

    try:
        return load_engine(ref)
    except NotImplementedError:
        return _load_module_engine(ref)


def _score_result(scores: dict[str, float], result: GameResult) -> None:
    if result.result == "1-0":
        scores[result.white] += 1.0
    elif result.result == "0-1":
        scores[result.black] += 1.0
    else:
        scores[result.white] += 0.5
        scores[result.black] += 0.5


async def _run_match(
    white_engine: Engine,
    black_engine: Engine,
    games: int,
    time_per_move_ms: int,
) -> list[GameResult]:
    results: list[GameResult] = []
    for game_id in range(games):
        if game_id % 2 == 0:
            result = await play_game(white_engine, black_engine, time_per_move_ms, game_id=game_id)
        else:
            result = await play_game(black_engine, white_engine, time_per_move_ms, game_id=game_id)
        results.append(result)
    return results


def _print_table(results: list[GameResult], white_name: str, black_name: str) -> None:
    scores = {white_name: 0.0, black_name: 0.0}
    for result in results:
        _score_result(scores, result)

    print(f"{'Game':>4}  {'White':<24}  {'Black':<24}  {'Result':<7}  Termination")
    print("-" * 78)
    for i, result in enumerate(results, start=1):
        print(
            f"{i:>4}  {result.white:<24}  {result.black:<24}  "
            f"{result.result:<7}  {result.termination}"
        )
    print("-" * 78)
    print(f"{white_name}: {scores[white_name]:.1f}")
    print(f"{black_name}: {scores[black_name]:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an alternating-color engine match.")
    parser.add_argument("--white", "--white-module", dest="white", default="random")
    parser.add_argument("--black", "--black-module", dest="black", default="random")
    parser.add_argument("--n", type=int, default=4, help="number of games to play")
    parser.add_argument("--time-per-move-ms", type=int, default=settings.time_per_move_ms)
    args = parser.parse_args()

    if args.n < 1:
        raise SystemExit("--n must be at least 1")

    white_engine = _load_engine(args.white, "white", seed=1)
    black_engine = _load_engine(args.black, "black", seed=2)
    if white_engine is black_engine:
        raise SystemExit("white and black resolved to the same engine object")
    if white_engine.name == black_engine.name:
        white_engine.name = f"{white_engine.name}-white"
        black_engine.name = f"{black_engine.name}-black"

    results = asyncio.run(_run_match(white_engine, black_engine, args.n, args.time_per_move_ms))
    _print_table(results, white_engine.name, black_engine.name)


if __name__ == "__main__":
    main()
