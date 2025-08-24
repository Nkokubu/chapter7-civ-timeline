import csv
from pathlib import Path
from typing import Dict
from sqlalchemy import func

from sqlmodel import select

from app.db.session import get_session
from app.models.civilization import Civilization
from app.models.event import Event

SEEDS = Path("data/seeds")

def load_civs():
    rows = []
    with (SEEDS / "civilizations.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # coerce ints/floats
            row["start_year"] = int(row["start_year"])
            row["end_year"]   = int(row["end_year"])
            row["lat"] = float(row["lat"]) if row.get("lat") else None
            row["lon"] = float(row["lon"]) if row.get("lon") else None
            rows.append(row)
    return rows

def load_events():
    rows = []
    with (SEEDS / "events.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["year"] = int(row["year"])
            rows.append(row)
    return rows

def upsert_civilizations(session) -> Dict[str, Civilization]:
    """Insert civs if missing; return dict[slug] -> Civilization."""
    slug_to_civ: Dict[str, Civilization] = {}

    for r in load_civs():
        # Try to find by slug
        civ = session.exec(select(Civilization).where(Civilization.slug == r["slug"])).first()
        if civ is None:
            civ = Civilization(
                slug=r["slug"],
                name=r["name"],
                region=r["region"],
                start_year=r["start_year"],
                end_year=r["end_year"],
                lat=r.get("lat"),
                lon=r.get("lon"),
            )
            session.add(civ)
            session.commit()
            session.refresh(civ)
        else:
            # optional: update basic fields to keep CSV as truth in dev
            civ.name = r["name"]
            civ.region = r["region"]
            civ.start_year = r["start_year"]
            civ.end_year = r["end_year"]
            civ.lat = r.get("lat")
            civ.lon = r.get("lon")
            session.add(civ)
            session.commit()
        slug_to_civ[r["slug"]] = civ

    return slug_to_civ

def insert_events(session, slug_to_civ):
    inserted, skipped = 0, 0
    for r in load_events():
        civ = slug_to_civ.get(r["civ_slug"])
        if not civ:
            print(f"!! Missing civ for event: {r['title']} (slug={r['civ_slug']}) — skipping")
            skipped += 1
            continue

        # Idempotency: check duplicate (same civ, title, and year)
        exists = session.exec(
            select(Event).where(
                (Event.civilization_id == civ.id) &
                (Event.title == r["title"]) &
                (Event.year == r["year"])
            )
        ).first()

        if exists:
            skipped += 1
            continue

        ev = Event(
            civilization_id=civ.id,
            title=r["title"],
            year=r["year"],
            kind=r.get("kind"),
            summary=r.get("summary"),
            tags=r.get("tags"),
        )
        session.add(ev)
        inserted += 1

    session.commit()
    return inserted, skipped

def main():
    with get_session() as session:
        slug_to_civ = upsert_civilizations(session)
        inserted, skipped = insert_events(session, slug_to_civ)
        total = session.exec(select(func.count(Event.id))).one() # Aggregate count (fast + correct)
        # If your SQLAlchemy returns a 1-tuple, unwrap it:
        if isinstance(total, tuple):
             total = total[0]
        print(f"Inserted {inserted} events (skipped {skipped}). Total now: {total}")

if __name__ == "__main__":
    main()