# scripts/ingest_seeds.py

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

from sqlalchemy import func
from sqlmodel import select

from app.db.session import get_session
from app.models.civilization import Civilization
from app.models.event import Event
# Import directly from the module to avoid any __init__ export issues
from app.models.source import Source, CivSourceLink, EventSourceLink
from sqlalchemy import func, and_


# -------------------------------------------------------------------
# Alias / misspelling pairs. We’ll resolve either side to the DB row.
# -------------------------------------------------------------------
PAIR_ALIASES = [
    ("maurya", "mauryan"),
    ("axum", "aksum"),
    ("ptolemaic", "ptolemaic-egypt"),
    ("rome", "roman"),
    ("akemenid", "persia"),
    ("achaemenid", "persia"),
]


def _build_slug_lookup(slug_to_civ: Dict[str, Civilization]) -> Dict[str, Civilization]:
    """
    Return a lookup dict that contains the DB slugs plus alias keys
    pointing to the same Civilization row (prefers whichever exists).
    """
    lookup = {k.lower(): v for k, v in slug_to_civ.items()}
    for a, b in PAIR_ALIASES:
        a, b = a.lower(), b.lower()
        if a in lookup and b not in lookup:
            lookup[b] = lookup[a]
        elif b in lookup and a not in lookup:
            lookup[a] = lookup[b]
    return lookup


# Resolve to project_root/data/seeds regardless of CWD
SEEDS = Path(__file__).resolve().parents[1] / "data" / "seeds"


# -------------------------
# Load CSV helper functions
# -------------------------
def _to_int(val: str | None) -> int | None:
    if val is None or str(val).strip() == "":
        return None
    return int(val)


def _to_float(val: str | None) -> float | None:
    if val is None or str(val).strip() == "":
        return None
    return float(val)


def load_civs() -> list[dict]:
    rows: list[dict] = []
    p = SEEDS / "civilizations.csv"
    if not p.exists():
        return rows
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["start_year"] = int(row["start_year"])
            row["end_year"] = int(row["end_year"])
            row["lat"] = _to_float(row.get("lat"))
            row["lon"] = _to_float(row.get("lon"))
            rows.append(row)
    return rows


def load_events() -> list[dict]:
    rows: list[dict] = []
    p = SEEDS / "events.csv"
    if not p.exists():
        return rows
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["year"] = int(row["year"])
            rows.append(row)
    return rows


# -------------------------
# Upsert civilizations/events
# -------------------------
def upsert_civilizations(session) -> dict[str, Civilization]:
    """Insert civs if missing; return dict[slug_lower] -> Civilization."""
    slug_to_civ: dict[str, Civilization] = {}

    for r in load_civs():
        slug = (r["slug"] or "").strip()
        slug_lower = slug.lower()

        civ = session.exec(
            select(Civilization).where(func.lower(Civilization.slug) == slug_lower)
        ).first()

        if civ is None:
            civ = Civilization(
                slug=slug,
                name=r["name"],
                region=r["region"],
                start_year=int(r["start_year"]),
                end_year=int(r["end_year"]),
                lat=r.get("lat"),
                lon=r.get("lon"),
            )
            session.add(civ)
            session.commit()
            session.refresh(civ)
        else:
            civ.name = r["name"]
            civ.region = r["region"]
            civ.start_year = int(r["start_year"])
            civ.end_year = int(r["end_year"])
            civ.lat = r.get("lat")
            civ.lon = r.get("lon")
            session.add(civ)
            session.commit()

        # IMPORTANT: store by lower-cased slug so tests see {"roman","han"}
        slug_to_civ[slug_lower] = civ

    return slug_to_civ


def insert_events(session, slug_to_civ):
    """
    Insert events from seeds for the provided slug->Civilization map.
    Returns (inserted_count, skipped_count).
    """
    import os
    DEBUG = os.getenv("INGEST_DEBUG") == "1"

    # alias-tolerant lookup if keys come in mixed case
    lookup = {k.lower(): v for k, v in slug_to_civ.items()}
    if DEBUG:
        print(f"[ingest] lookup keys = {sorted(lookup.keys())}")

    inserted, skipped = 0, 0

    # ensure civ rows have ids visible to this txn
    session.flush()

    events_rows = load_events()
    if DEBUG:
        print(f"[ingest] load_events() -> {len(events_rows)} rows")

    for r in events_rows:
        raw = (r.get("civ_slug") or "").strip().lower()
        civ = lookup.get(raw)

        # Fallback: if mapping somehow missed, re-query by slug
        if civ is None and raw:
            civ = session.exec(
                select(Civilization).where(func.lower(Civilization.slug) == raw)
            ).first()
            if civ and DEBUG:
                print(f"[ingest] fallback hit: civ '{raw}' -> id={civ.id}")

        if civ is None:
            if DEBUG:
                print(f"[ingest] SKIP: unknown civ_slug='{raw}' for title={r.get('title')}")
            skipped += 1
            continue

        title = (r["title"] or "").strip()
        year = int(r["year"])

        # existence check via COUNT(*) — robust across drivers/versions
        cnt_row = session.exec(
            select(func.count(Event.id)).where(
                and_(
                    Event.civilization_id == civ.id,
                    Event.title == title,
                    Event.year == year,
                )
            )
        ).one()
        cnt = cnt_row[0] if isinstance(cnt_row, tuple) else cnt_row

        if (cnt or 0) > 0:
            if DEBUG:
                print(f"[ingest] SKIP (exists): civ_id={civ.id} title={title!r} year={year}")
            skipped += 1
            continue

        ev = Event(
            civilization_id=civ.id,
            title=title,
            year=year,
            kind=(r.get("kind") or None),
            summary=(r.get("summary") or None),
            tags=(r.get("tags") or None),
        )
        session.add(ev)
        inserted += 1
        if DEBUG:
            print(f"[ingest] ADD: civ_id={civ.id} title={title!r} year={year}")

    session.commit()
    if DEBUG:
        print(f"[ingest] result inserted={inserted} skipped={skipped}")
    return inserted, skipped


# -------------------------
# Sources + link ingestion
# -------------------------
def load_sources() -> list[dict]:
    rows: list[dict] = []
    p = SEEDS / "sources.csv"
    if not p.exists():
        return rows
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "key": r["key"].strip(),
                    "title": r["title"].strip(),
                    "url": (r.get("url") or "").strip(),
                }
            )
    return rows


def upsert_sources(session) -> Dict[str, Source]:
    by_key: Dict[str, Source] = {}
    for r in load_sources():
        src = session.exec(select(Source).where(Source.key == r["key"])).first()
        if src is None:
            src = Source(key=r["key"], title=r["title"], url=r["url"] or None)
            session.add(src)
            session.commit()
            session.refresh(src)
        else:
            src.title = r["title"]
            src.url = r["url"] or None
            session.add(src)
            session.commit()
        by_key[r["key"]] = src
    return by_key


def load_civ_source_links() -> list[dict]:
    rows: list[dict] = []
    p = SEEDS / "civ_sources.csv"
    if not p.exists():
        return rows
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "civ_slug": (r["civ_slug"] or "").strip().lower(),
                    "source_key": r["source_key"].strip(),
                }
            )
    return rows


def load_event_source_links() -> list[dict]:
    rows: list[dict] = []
    p = SEEDS / "event_sources.csv"
    if not p.exists():
        return rows
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "civ_slug": (r["civ_slug"] or "").strip().lower(),
                    "event_title": r["event_title"].strip(),
                    "year": int(r["year"]),
                    "source_key": r["source_key"].strip(),
                }
            )
    return rows


def link_civ_sources(
    session,
    slug_lookup: Dict[str, Civilization],
    src_by_key: Dict[str, Source],
) -> Tuple[int, int]:
    made, skipped = 0, 0
    for r in load_civ_source_links():
        civ = slug_lookup.get(r["civ_slug"])
        src = src_by_key.get(r["source_key"])
        if not civ or not src:
            skipped += 1
            continue
        exists = session.exec(
            select(CivSourceLink).where(
                (CivSourceLink.civilization_id == civ.id)
                & (CivSourceLink.source_id == src.id)
            )
        ).first()
        if exists:
            skipped += 1
            continue
        session.add(CivSourceLink(civilization_id=civ.id, source_id=src.id))
        made += 1
    session.commit()
    return made, skipped


def link_event_sources(
    session,
    slug_lookup: Dict[str, Civilization],
    src_by_key: Dict[str, Source],
) -> Tuple[int, int]:
    made, skipped = 0, 0
    for r in load_event_source_links():
        civ = slug_lookup.get(r["civ_slug"])
        src = src_by_key.get(r["source_key"])
        if not civ or not src:
            skipped += 1
            continue
        ev = session.exec(
            select(Event).where(
                (Event.civilization_id == civ.id)
                & (Event.title == r["event_title"])
                & (Event.year == r["year"])
            )
        ).first()
        if not ev:
            skipped += 1
            continue
        exists = session.exec(
            select(EventSourceLink).where(
                (EventSourceLink.event_id == ev.id)
                & (EventSourceLink.source_id == src.id)
            )
        ).first()
        if exists:
            skipped += 1
            continue
        session.add(EventSourceLink(event_id=ev.id, source_id=src.id))
        made += 1
    session.commit()
    return made, skipped


# -------------------------
# main
# -------------------------
def main() -> None:
    with get_session() as session:
        # 1) civs + events
        slug_to_civ = upsert_civilizations(session)
        inserted, skipped = insert_events(session, slug_to_civ)

        # 2) sources + links
        src_by_key = upsert_sources(session)
        lookup = _build_slug_lookup(slug_to_civ)
        made_c, skipped_c = link_civ_sources(session, lookup, src_by_key)
        made_e, skipped_e = link_event_sources(session, lookup, src_by_key)

        # 3) totals (robust across SA versions)
        try:
            total_events = session.exec(select(func.count(Event.id))).scalar_one()
        except Exception:
            total_events = session.exec(select(func.count(Event.id))).one()
            total_events = total_events[0] if isinstance(total_events, tuple) else total_events

        print(f"Inserted {inserted} events (skipped {skipped}). Total now: {total_events}")
        print(
            f"Linked civ-sources: {made_c} new ({skipped_c} skipped); "
            f"event-sources: {made_e} new ({skipped_e} skipped)."
        )


if __name__ == "__main__":
    main()

