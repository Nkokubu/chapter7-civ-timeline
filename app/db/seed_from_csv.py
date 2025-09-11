# app/db/seed_from_csv.py
import csv, os
from pathlib import Path
from typing import Dict, Any, List, Optional

from sqlmodel import select
from app.db.session import get_session
from app.models.civilization import Civilization
from app.models.event import Event

# Source model is optional in your app; load if present
try:
    from app.models.source import Source  # noqa: F401
    HAVE_SOURCE = True
except Exception:
    HAVE_SOURCE = False

# app/db/seed_from_csv.py  (top of file, after imports)
from sqlmodel import SQLModel
from app.models.civilization import Civilization
from app.models.event import Event
try:
    from app.models.source import Source
    HAVE_SOURCE = True
except Exception:
    HAVE_SOURCE = False

from app.db.session import engine, get_session

# 🔴 Ensure tables exist before any queries
SQLModel.metadata.create_all(engine)

SEED_DIR = Path("data/seeds")

def _load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        rows = []
        for r in rdr:
            # strip keys/values
            rows.append({(k or "").strip(): (v or "").strip() for k, v in r.items()})
        return rows

def _to_int(val: str) -> Optional[int]:
    try:
        return int(val)
    except Exception:
        return None

def seed_civilizations():
    rows = _load_csv(SEED_DIR / "civilizations.csv")
    if not rows:
        print("⚠️  No civilizations.csv found in data/seeds — skipping.")
        return

    inserted = updated = skipped = 0
    with get_session() as s:
        # cache by slug
        existing = { (c.slug or "").lower(): c for c in s.exec(select(Civilization)).all() }
        for r in rows:
            slug = (r.get("slug") or "").lower()
            if not slug:
                skipped += 1; continue

            c = existing.get(slug)
            payload = dict(
                name = r.get("name") or None,
                slug = slug,
                region = r.get("region") or None,
                start_year = _to_int(r.get("start_year", "")),
                end_year   = _to_int(r.get("end_year", "")),
            )

            # coordinates: latitude/longitude or lat/lon
            lat = r.get("latitude") or r.get("lat")
            lon = r.get("longitude") or r.get("lon")
            if lat and lon:
                try:
                    payload["latitude"]  = float(lat)
                    payload["longitude"] = float(lon)
                except Exception:
                    pass

            if c:
                # update minimal fields
                for k, v in payload.items():
                    setattr(c, k, v)
                updated += 1
            else:
                c = Civilization(**payload)
                s.add(c)
                inserted += 1
        s.commit()
    print(f"✅ Civilizations: inserted={inserted}, updated={updated}, skipped_rows={skipped}")

def seed_events():
    rows = _load_csv(SEED_DIR / "events.csv")
    if not rows:
        print("⚠️  No events.csv found in data/seeds — skipping.")
        return

    inserted = skipped = 0
    with get_session() as s:
        civs = { (c.slug or "").lower(): c for c in s.exec(select(Civilization)).all() }
        for r in rows:
            title = r.get("title") or ""
            year  = _to_int(r.get("year", ""))
            civ_slug = (r.get("civ_slug") or r.get("civilization_slug") or "").lower()

            if not title or year is None or not civ_slug:
                skipped += 1; continue

            civ = civs.get(civ_slug)
            if not civ:
                skipped += 1; continue

            # avoid duplicates by (civ_id, title, year)
            exists = s.exec(
                select(Event).where(
                    Event.civilization_id == civ.id,
                    Event.title == title,
                    Event.year == year
                )
            ).first()
            if exists:
                continue

            ev = Event(
                civilization_id=civ.id,
                title=title,
                year=year,
                kind=(r.get("kind") or None),
                tags=(r.get("tags") or None),
                summary=(r.get("summary") or None),
            )
            s.add(ev)
            inserted += 1
        s.commit()
    print(f"✅ Events: inserted={inserted}, skipped_rows={skipped}")

def seed_sources():
    if not HAVE_SOURCE:
        print("ℹ️  Source model not present — skipping sources.")
        return

    rows = _load_csv(SEED_DIR / "sources.csv")
    if not rows:
        print("ℹ️  No sources.csv found — skipping.")
        return

    from app.models.source import Source  # type: ignore

    inserted = updated = 0
    with get_session() as s:
        existing = { (src.key or "").lower(): src for src in s.exec(select(Source)).all() }
        for r in rows:
            key = (r.get("key") or "").lower()
            if not key:
                continue
            title = r.get("title") or None
            url   = r.get("url") or None
            src = existing.get(key)
            if src:
                if title is not None: src.title = title
                if url   is not None: src.url   = url
                updated += 1
            else:
                s.add(Source(key=key, title=title, url=url))
                inserted += 1
        s.commit()
    print(f"✅ Sources: inserted={inserted}, updated={updated}")

def link_civilization_sources():
    """Optional file: data/seeds/civilization_sources.csv with columns: civ_slug,source_key"""
    if not HAVE_SOURCE:
        return
    path = SEED_DIR / "civilization_sources.csv"
    rows = _load_csv(path)
    if not rows:
        return

    from app.models.source import Source  # type: ignore
    linked = skipped = 0
    with get_session() as s:
        civs = { (c.slug or "").lower(): c for c in s.exec(select(Civilization)).all() }
        srcs = { (src.key or "").lower(): src for src in s.exec(select(Source)).all() }

        for r in rows:
            slug = (r.get("civ_slug") or "").lower()
            key  = (r.get("source_key") or "").lower()
            civ, src = civs.get(slug), srcs.get(key)
            if not civ or not src:
                skipped += 1; continue
            if src not in civ.sources:
                civ.sources.append(src)
                linked += 1
        s.commit()
    print(f"🔗 Civilization↔Source links: linked={linked}, skipped={skipped}")

def link_event_sources():
    """Optional file: data/seeds/event_sources.csv with columns: civ_slug,event_title,event_year,source_key"""
    if not HAVE_SOURCE:
        return
    path = SEED_DIR / "event_sources.csv"
    rows = _load_csv(path)
    if not rows:
        return

    from app.models.source import Source  # type: ignore
    linked = skipped = 0
    with get_session() as s:
        civs = { (c.slug or "").lower(): c for c in s.exec(select(Civilization)).all() }
        srcs = { (src.key or "").lower(): src for src in s.exec(select(Source)).all() }

        for r in rows:
            slug = (r.get("civ_slug") or "").lower()
            title = r.get("event_title") or ""
            year  = _to_int(r.get("event_year", ""))
            key   = (r.get("source_key") or "").lower()
            civ = civs.get(slug)
            src = srcs.get(key)
            if not civ or not src or not title or year is None:
                skipped += 1; continue

            ev = s.exec(
                select(Event).where(
                    Event.civilization_id == civ.id,
                    Event.title == title,
                    Event.year == year
                )
            ).first()
            if not ev:
                skipped += 1; continue
            if src not in ev.sources:
                ev.sources.append(src)
                linked += 1
        s.commit()
    print(f"🔗 Event↔Source links: linked={linked}, skipped={skipped}")

def run_all():
    if not SEED_DIR.exists():
        print("❌ data/seeds directory not found.")
        return
    seed_civilizations()
    seed_events()
    seed_sources()            # optional
    link_civilization_sources()  # optional
    link_event_sources()         # optional
    print("🎉 Done seeding from CSV.")

if __name__ == "__main__":
    run_all()
