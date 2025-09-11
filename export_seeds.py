from pathlib import Path
import csv
from sqlmodel import select
from sqlalchemy.orm import selectinload

from app.db.session import get_session
from app.models.civilization import Civilization
from app.models.event import Event
from app.models.source import Source, CivSourceLink, EventSourceLink

OUTDIR = Path("data/seeds")
OUTDIR.mkdir(parents=True, exist_ok=True)

def _clean(s):
    if s is None:
        return ""
    return " ".join(str(s).split())  # collapse whitespace/newlines

def export_events():
    path = OUTDIR / "events_full_export.csv"
    with get_session() as s:
        rows = s.exec(
            select(Event)
            .options(selectinload(Event.civilization))
            .order_by(Event.year, Event.title)
        ).all()

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["civ_slug", "title", "year", "kind", "tags", "summary"])
        total = 0
        for e in rows:
            slug = (e.civilization.slug if e.civilization else "") or ""
            w.writerow([
                slug,
                _clean(e.title),
                int(e.year) if e.year is not None else "",
                _clean(e.kind),
                _clean(e.tags),
                _clean(e.summary),
            ])
            total += 1
    return path, total

def export_sources():
    path = OUTDIR / "sources_full_export.csv"
    with get_session() as s:
        srcs = s.exec(select(Source).order_by(Source.key)).all()

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "title", "url", "author", "year", "publisher", "note"])
        total = 0
        for src in srcs:
            w.writerow([
                _clean(src.key),
                _clean(src.title),
                _clean(src.url),
                _clean(src.author),
                src.year if getattr(src, "year", None) is not None else "",
                _clean(getattr(src, "publisher", "")),
                _clean(getattr(src, "note", "")),
            ])
            total += 1
    return path, total

def export_civ_source_links():
    path = OUTDIR / "civ_sources_full_export.csv"
    with get_session() as s:
        # load civs and sources for pretty slugs/keys
        civ_by_id = {c.id: c for c in s.exec(select(Civilization)).all()}
        src_by_id = {x.id: x for x in s.exec(select(Source)).all()}
        links = s.exec(select(CivSourceLink)).all()

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["civ_slug", "source_key"])
        total = 0
        for ln in links:
            civ = civ_by_id.get(ln.civilization_id)
            src = src_by_id.get(ln.source_id)
            if not civ or not src:
                continue
            w.writerow([_clean(civ.slug), _clean(src.key)])
            total += 1
    return path, total

def export_event_source_links():
    path = OUTDIR / "event_sources_full_export.csv"
    with get_session() as s:
        # We need event title/year and civ slug
        evs = s.exec(
            select(Event)
            .options(selectinload(Event.civilization))
        ).all()
        ev_by_id = {e.id: e for e in evs}
        src_by_id = {x.id: x for x in s.exec(select(Source)).all()}
        links = s.exec(select(EventSourceLink)).all()

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["civ_slug", "event_title", "year", "source_key"])
        total = 0
        for ln in links:
            ev = ev_by_id.get(ln.event_id)
            src = src_by_id.get(ln.source_id)
            if not ev or not src or not ev.civilization:
                continue
            w.writerow([
                _clean(ev.civilization.slug),
                _clean(ev.title),
                int(ev.year) if ev.year is not None else "",
                _clean(src.key)
            ])
            total += 1
    return path, total

if __name__ == "__main__":
    ep, ec = export_events()
    sp, sc = export_sources()
    cp, cc = export_civ_source_links()
    lp, lc = export_event_source_links()
    print(f"Exported events   : {ec:>4} -> {ep}")
    print(f"Exported sources  : {sc:>4} -> {sp}")
    print(f"Civ-source links  : {cc:>4} -> {cp}")
    print(f"Event-source links: {lc:>4} -> {lp}")
