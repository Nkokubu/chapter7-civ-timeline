# tests/test_migrations.py
from alembic.config import Config
from alembic import command
import sqlite3

def test_migrations_sqlite(tmp_path):
    db_file = tmp_path / "migrate_test.db"

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_file}")
    command.upgrade(cfg, "head")

    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = {row[0] for row in cur.fetchall()}
    con.close()

    # core tables should exist
    assert {"civilizations", "events", "sources", "civ_source_link", "event_source_link"}.issubset(tables)
