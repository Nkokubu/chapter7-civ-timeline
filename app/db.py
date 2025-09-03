# app/db.py
from sqlmodel import SQLModel, create_engine, Session
import os

DB_URL = os.getenv("DB_URL", "sqlite:///./data/app.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

def get_session():
    with Session(engine) as session:
        yield session
