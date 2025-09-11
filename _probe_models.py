import inspect
from sqlmodel import SQLModel
from app.models import CivSourceLink, Source, Civilization, Event
from app.db.session import engine
from sqlalchemy import inspect as sai

print("CivSourceLink defined in:", inspect.getsourcefile(CivSourceLink))
print("CivSourceLink columns (ORM view):", list(CivSourceLink.__table__.c.keys()))
print("Has civ_id attr?", hasattr(CivSourceLink, "civ_id"))
print("Has civilization_id attr?", hasattr(CivSourceLink, "civilization_id"))

insp = sai(engine)
print("DB civ_source_link columns (DB view):", [c["name"] for c in insp.get_columns("civ_source_link")])
