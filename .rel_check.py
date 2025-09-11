from sqlmodel import select
from app.db.session import get_session
from app.models import Civilization, Event, Source

with get_session() as s:
    civ = s.exec(select(Civilization)).first()
    print("sample civ:", civ.name if civ else None, "| events:", len(civ.events) if civ else 0)

    ev = s.exec(select(Event)).first()
    print("sample event:", ev.title if ev else None, "| sources:", [ (src.key, src.title) for src in ev.sources ] if ev else [])
