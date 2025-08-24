from sqlalchemy import func
from sqlmodel import select

# If your editor/pytest can't find 'app', uncomment the two lines below.
# import sys, pathlib
# sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from app.db.session import get_session
from app.models.event import Event

def test_seed_has_at_least_60_events():
    with get_session() as s:
        total = s.exec(select(func.count(Event.id))).one()
        if isinstance(total, tuple):  # some drivers return a 1-tuple
            total = total[0]
        assert total >= 60
