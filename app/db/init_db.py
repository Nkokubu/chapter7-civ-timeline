from sqlmodel import SQLModel
from app.db.session import engine
from app.models import Civilization, Event  # ensure models are imported

def init_db():
    SQLModel.metadata.create_all(engine)

if __name__ == "__main__":
    init_db()