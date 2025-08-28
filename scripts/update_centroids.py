# scripts/update_centroids.py
import csv
from pathlib import Path
from sqlmodel import select
from app.db.session import get_session
from app.models.civilization import Civilization

CSV = Path("data/seeds/civ_centroids.csv")

def main():
    if not CSV.exists():
        print(f"Missing {CSV}")
        return
    with get_session() as s, CSV.open() as f:
        reader = csv.DictReader(f)
        updated = 0
        for r in reader:
            slug = r["slug"].strip()
            lat = float(r["latitude"])
            lon = float(r["longitude"])
            civ = s.exec(select(Civilization).where(Civilization.slug == slug)).first()
            if not civ:
                print(f"Skip (no civ): {slug}")
                continue
            civ.latitude = lat
            civ.longitude = lon
            s.add(civ)
            updated += 1
        s.commit()
        print(f"Updated {updated} civilizations.")

if __name__ == "__main__":
    main()
