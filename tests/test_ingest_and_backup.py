# tests/test_ingest_and_backup.py
import os
from pathlib import Path
import sqlite3

def test_ingest_roundtrip(tmp_path, monkeypatch):
    # point DATABASE_URL to a temp DB
    db_file = tmp_path / "app.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")

    # run migrations to create schema
    from alembic.config import Config
    from alembic import command
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_file}")
    command.upgrade(cfg, "head")

    # write minimal seeds into a temp seeds dir and monkeypatch the module constant
    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    (seeds_dir / "civilizations.csv").write_text(
        "slug,name,region,start_year,end_year,lat,lon\n"
        "roman,Roman Republic/Empire,Europe,-500,500,41.9,12.5\n"
    , encoding="utf-8")
    (seeds_dir / "events.csv").write_text(
        "civ_slug,title,year,kind,summary,tags\n"
        "roman,Edict of Milan,313,religion,Legalizes Christianity,edict,christianity\n"
    , encoding="utf-8")
    (seeds_dir / "sources.csv").write_text(
        "key,title,url\nwikipedia_rome,Roman Empire (Wikipedia),https://en.wikipedia.org/wiki/Roman_Empire\n"
    , encoding="utf-8")
    (seeds_dir / "civ_sources.csv").write_text(
        "civ_slug,source_key\nroman,wikipedia_rome\n"
    , encoding="utf-8")
    (seeds_dir / "event_sources.csv").write_text(
        "civ_slug,event_title,year,source_key\nroman,Edict of Milan,313,wikipedia_rome\n"
    , encoding="utf-8")

    # run ingest with patched SEEDS path
    import scripts.ingest_seeds as ingest
    ingest.SEEDS = seeds_dir  # override module constant
    ingest.main()

    # verify data present
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cur.execute("select count(*) from civilizations")
    assert cur.fetchone()[0] >= 1
    cur.execute("select count(*) from events")
    assert cur.fetchone()[0] >= 1
    cur.execute("select count(*) from sources")
    assert cur.fetchone()[0] >= 1
    con.close()

def test_backup_script(tmp_path, monkeypatch):
    # make a tiny DB file to back up
    db_file = tmp_path / "app.db"
    import sqlite3
    sqlite3.connect(db_file).close()

    # env + cwd for script
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    backups = tmp_path / "backups"
    backups.mkdir()
    monkeypatch.chdir(tmp_path)

    # run the script
    from scripts import backup_db
    backup_db.main()

    # assert a .sqlite3 file was created under backups/
    files = list(backups.glob("backup_*.sqlite3"))
    assert files, "No backup file created"
    assert files[0].stat().st_size >= 0
