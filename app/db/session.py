# app/db/session.py
import os
from contextlib import contextmanager
from sqlmodel import SQLModel, Session, create_engine

# Use the same DB for app + scripts. Adjust if you keep your db elsewhere.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/civ.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("SQL_ECHO", "0") == "1",
    connect_args=connect_args,
)

@contextmanager
def get_session():
    with Session(engine) as s:
        yield s

def create_all():
    SQLModel.metadata.create_all(engine)
