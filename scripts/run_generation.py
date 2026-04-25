"""One-off: run a single generation from the command line.

Thin wrapper around `cubist.orchestration.run` for the case where you want
to kick off exactly one generation without constructing a full argparse
invocation. Intended for demo-day muscle memory:

    uv run python scripts/run_generation.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from cubist.orchestration.run import main  # noqa: E402


if __name__ == "__main__":
    asyncio.run(main(generations=1))
