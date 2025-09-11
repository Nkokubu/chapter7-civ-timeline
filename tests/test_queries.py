# tests/test_queries.py
from sqlmodel import Session
from app.models.civilization import Civilization
from app.models.event import Event
from streamlit_app import _split_csv, _norm_kind, _event_score

def test_helpers_split_norm_score():
    assert _split_csv(["a, b", "c"]) == ["a","b","c"]
    assert _norm_kind("Tech") == "tech"
    assert _norm_kind("unknown") == "other"

    e = Event(title="Royal Road", kind="economy", tags="trade, road", year=-500)
    s = _event_score(e)
    assert isinstance(s, int) and s >= 1

def test_basic_query_filters(session: Session, monkeypatch):
    # Build tiny dataset
    rome = Civilization(slug="roman", name="Rome", region="Europe", start_year=-27, end_year=476)
    han  = Civilization(slug="han", name="Han", region="East Asia", start_year=-202, end_year=220)
    session.add(rome); session.add(han); session.commit()
    session.refresh(rome); session.refresh(han)

    session.add(Event(title="Edict of Milan", year=313, civilization_id=rome.id, kind="religion", tags="edict,christianity"))
    session.add(Event(title="Paper Invented", year=105, civilization_id=han.id, kind="tech", tags="invention,tech"))
    session.commit()

    # monkeypatch get_session used in _query_events
    import streamlit_app as app
    from contextlib import contextmanager
    @contextmanager
    def _get():
        yield session
    monkeypatch.setattr(app, "get_session", _get, raising=False)

    evs_all = app._query_events((-1000, 500), [], [])
    assert len(evs_all) == 2

    evs_eu = app._query_events((-1000, 500), ["Europe"], [])
    assert len(evs_eu) == 1 and evs_eu[0].title == "Edict of Milan"

    evs_tag = app._query_events((-1000, 500), [], ["tech"])
    assert len(evs_tag) == 1 and evs_tag[0].title == "Paper Invented"
