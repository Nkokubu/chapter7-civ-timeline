from __future__ import annotations
import os
from typing import Iterator
from contextlib import contextmanager

from sqlmodel import Session, SQLModel, create_engine

# Default to local SQLite file inside repo
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///db/civ.db")

# SQLite needs check_same_thread=False for FastAPI dev
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("SQL_ECHO", "0") == "1",
    connect_args=connect_args,
)


def init_db() -> None:
    """Create tables if they don't exist (for tests/dev).
    In production, use Alembic migrations.
    """
    from app.models.core import SQLModel  # ensure models are imported
    SQLModel.metadata.create_all(engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()