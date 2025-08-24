from sqlmodel import create_engine, Session

DATABASE_URL = "sqlite:///db/civ.db"
engine = create_engine(DATABASE_URL, echo=False)

def get_session():
    return Session(engine)