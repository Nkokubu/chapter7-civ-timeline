# tests/test_ingest.py
from sqlmodel import select
import scripts.ingest_seeds as ing

def test_upsert_and_link(session, monkeypatch):
    monkeypatch.setattr(ing, "load_civs", lambda: [
        {"slug":"roman","name":"Roman Empire","region":"Europe","start_year":-27,"end_year":476,"lat":41.9,"lon":12.5},
        {"slug":"han","name":"Han dynasty","region":"East Asia","start_year":-202,"end_year":220,"lat":34.3,"lon":108.9},
    ])
    monkeypatch.setattr(ing, "load_events", lambda: [
        {"civ_slug":"roman","title":"Edict of Milan","year":313,"kind":"religion","summary":"Religious toleration","tags":"edict,christianity"},
        {"civ_slug":"han","title":"Paper Invented","year":105,"kind":"tech","summary":"Cai Lun improves paper","tags":"invention,tech"},
    ])
    monkeypatch.setattr(ing, "load_sources", lambda: [
        {"key":"wikipedia_rome","title":"Rome (Wikipedia)","url":"https://en.wikipedia.org/wiki/Roman_Empire"},
        {"key":"wikipedia_han","title":"Han (Wikipedia)","url":"https://en.wikipedia.org/wiki/Han_dynasty"},
    ])
    monkeypatch.setattr(ing, "load_civ_source_links", lambda: [
        {"civ_slug":"roman","source_key":"wikipedia_rome"},
        {"civ_slug":"han","source_key":"wikipedia_han"},
    ])
    monkeypatch.setattr(ing, "load_event_source_links", lambda: [
        {"civ_slug":"roman","event_title":"Edict of Milan","year":313,"source_key":"wikipedia_rome"},
        {"civ_slug":"han","event_title":"Paper Invented","year":105,"source_key":"wikipedia_han"},
    ])

    slug_to_civ = ing.upsert_civilizations(session)
    assert set(slug_to_civ.keys()) == {"roman","han"}

    ins, skip = ing.insert_events(session, slug_to_civ)
    assert ins == 2 and skip == 0

    by_key = ing.upsert_sources(session)
    assert set(by_key.keys()) == {"wikipedia_rome","wikipedia_han"}

    lookup = ing._build_slug_lookup(slug_to_civ)
    made_c, skipped_c = ing.link_civ_sources(session, lookup, by_key)
    made_e, skipped_e = ing.link_event_sources(session, lookup, by_key)
    assert made_c == 2 and made_e == 2

    # verify joins work
    from app.models.event import Event
    from app.models.source import Source
    e = session.exec(select(Event).where(Event.title=="Edict of Milan")).one()
    assert e.sources and isinstance(e.sources[0], Source)
