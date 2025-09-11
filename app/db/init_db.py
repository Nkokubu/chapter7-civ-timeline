# app/db/init_db.py
from sqlmodel import SQLModel
from app.db.session import engine

if __name__ == "__main__":
    SQLModel.metadata.create_all(engine)
    print("✅ Tables created (or already exist).")
