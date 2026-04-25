"""Initialize the database and insert the baseline engine row. Idempotent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from cubist.engines.baseline import BaselineEngine  # noqa: E402,F401
from cubist.storage.db import get_session, init_db  # noqa: E402
from cubist.storage.models import EngineRow  # noqa: E402
from sqlmodel import select  # noqa: E402


def main() -> None:
    init_db()
    with get_session() as s:
        existing = s.exec(
            select(EngineRow).where(EngineRow.name == "baseline-v0")
        ).first()
        if existing:
            print("baseline already seeded:", existing.name)
            return
        row = EngineRow(
            name="baseline-v0",
            generation=0,
            parent_name=None,
            code_path="cubist.engines.baseline",
        )
        s.add(row)
        s.commit()
        print("seeded baseline-v0")


if __name__ == "__main__":
    main()
