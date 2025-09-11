from sqlmodel import select
from app.db.session import get_session
from app.models.civilization import Civilization
from app.models.event import Event
from app.models.source import Source, CivSourceLink, EventSourceLink

with get_session() as s:
    srcs = s.exec(select(Source)).all()
    civs = s.exec(select(Civilization).order_by(Civilization.name)).all()
    evs  = s.exec(select(Event)).all()
    civ_links = s.exec(select(CivSourceLink)).all()
    ev_links  = s.exec(select(EventSourceLink)).all()

    print(f"Sources: {len(srcs)} | Civ-source links: {len(civ_links)} | Event-source links: {len(ev_links)}")
    print(f"Civilizations: {len(civs)} | Events: {len(evs)}")
    print("\nSources per civilization:")
    for c in civs:
        print(f" - {c.name} [{c.slug}]: {len(c.sources)} source(s) -> {[s.key for s in c.sources]}")

    # sample events missing sources
    no_src = [e for e in evs if not e.sources]
    print(f"\nEvents without sources: {len(no_src)}")
    for e in no_src[:15]:
        cname = e.civilization.name if e.civilization else "?"
        print(f"   · {e.year:>5}  {cname}: {e.title}")
