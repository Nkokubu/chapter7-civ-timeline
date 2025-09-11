from sqlmodel import select
from app.db.session import get_session
from app.models import Civilization, Event

with get_session() as s:
    rome = s.exec(select(Civilization).where(Civilization.slug=="roman")).first()
    print("Rome sources:", [(src.key, src.title) for src in (rome.sources if rome else [])])

    ev = s.exec(select(Event).where(Event.title=="Edict of Milan", Event.year==313)).first()
    print("Edict sources:", [(src.key, src.title) for src in (ev.sources if ev else [])])
