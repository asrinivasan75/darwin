"""SQLite session helpers. Person E owns.

The database URL is read from `Settings.database_url` (defaults to
`sqlite:///./cubist.db` — a file next to the working directory).

`init_db()` is idempotent and should be called once at process start
(CLI runner, seed script, tests) before any `get_session()` usage.
`get_session()` returns a short-lived SQLModel session; prefer
`with get_session() as s:` so the connection is closed deterministically.
"""

from sqlmodel import Session, SQLModel, create_engine

from cubist.config import settings

_engine = create_engine(settings.database_url, echo=False)


def init_db() -> None:
    """Create all tables declared on the SQLModel metadata. No-op if present."""
    SQLModel.metadata.create_all(_engine)


def get_session() -> Session:
    """Return a new SQLModel session bound to the shared engine.

    Callers should use this as a context manager so the underlying
    connection is released even if the caller raises mid-transaction.
    """
    return Session(_engine)
