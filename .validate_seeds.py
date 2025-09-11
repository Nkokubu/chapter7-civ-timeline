from pathlib import Path
import csv
from sqlmodel import select
from app.db.session import get_session
from app.models.civilization import Civilization
from app.models.event import Event
from app.models.source import Source

SEEDS = Path("data/seeds")

# same alias idea used in ingest
PAIR_ALIASES = [
    ("maurya", "mauryan"),
    ("axum", "aksum"),
    ("ptolemaic", "ptolemaic-egypt"),
    ("rome", "roman"),
    ("akemenid", "persia"),
    ("achaemenid", "persia"),
]

def load_csv(name):
    p = SEEDS / name
    if not p.exists(): return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))

def build_slug_lookup():
    with get_session() as s:
        civs = s.exec(select(Civilization)).all()
    base = { (c.slug or "").lower(): c for c in civs }
    # add aliases that point to whichever exists
    look = dict(base)
    for a, b in PAIR_ALIASES:
        a, b = a.lower(), b.lower()
        if a in base and b not in look: look[b] = base[a]
        if b in base and a not in look: look[a] = base[b]
    return look

def main():
    src_rows = load_csv("sources.csv")
    civ_rows = load_csv("civ_sources.csv")
    ev_rows  = load_csv("event_sources.csv")

    with get_session() as s:
        srcs = s.exec(select(Source)).all()
        src_keys_db = {s.key for s in srcs}

    src_keys_csv = {r["key"].strip() for r in src_rows}
    if not src_rows:
        print("WARN: sources.csv is empty or missing.")
    # keys present in civ/event link CSVs but NOT in sources.csv
    link_src_keys = { (r.get("source_key") or "").strip() for r in (civ_rows + ev_rows) if (r.get("source_key") or "").strip() }
    missing_in_sources_csv = sorted(link_src_keys - src_keys_csv)
    if missing_in_sources_csv:
        print("MISSING in sources.csv:", missing_in_sources_csv)

    # keys present in sources.csv but not yet in DB (ingest not run or failed)
    not_in_db = sorted(src_keys_csv - src_keys_db)
    if not_in_db:
        print("Present in sources.csv but NOT in DB yet:", not_in_db)

    slug_lookup = build_slug_lookup()
    # civ_sources.csv checks
    bad_civ_rows = []
    for r in civ_rows:
        slug = (r.get("civ_slug") or "").strip().lower()
        skey = (r.get("source_key") or "").strip()
        if slug not in slug_lookup:
            bad_civ_rows.append(f"(unknown civ slug: {slug}, source_key={skey})")
        if skey not in src_keys_csv:
            bad_civ_rows.append(f"(unknown source key in sources.csv: {skey}, civ_slug={slug})")
    if bad_civ_rows:
        print("civ_sources.csv issues:")
        for m in bad_civ_rows:
            print("  -", m)

    # event_sources.csv checks (title+year must exist for that civ)
    with get_session() as s:
        bad_ev_rows = []
        for r in ev_rows:
            slug = (r.get("civ_slug") or "").strip().lower()
            title = (r.get("event_title") or "").strip()
            year_s = (r.get("year") or "").strip()
            skey = (r.get("source_key") or "").strip()
            try:
                year = int(year_s)
            except:
                bad_ev_rows.append(f"(bad year '{year_s}' for {slug} / {title})")
                continue
            civ = slug_lookup.get(slug)
            if not civ:
                bad_ev_rows.append(f"(unknown civ slug: {slug} for event '{title}')")
                continue
            if skey not in src_keys_csv:
                bad_ev_rows.append(f"(unknown source key in sources.csv: {skey} for event '{title}')")
            ev = s.exec(select(Event).where(
                Event.civilization_id == civ.id,
                Event.title == title,
                Event.year == year
            )).first()
            if not ev:
                bad_ev_rows.append(f"(no event match in DB: civ={slug}, title='{title}', year={year})")
        if bad_ev_rows:
            print("event_sources.csv issues:")
            for m in bad_ev_rows:
                print("  -", m)

    if not (missing_in_sources_csv or not_in_db or bad_civ_rows or bad_ev_rows):
        print("✅ Seeds look consistent. If links still don't appear, re-run: poetry run python -m scripts.ingest_seeds")

if __name__ == "__main__":
    main()
