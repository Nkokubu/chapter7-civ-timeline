from __future__ import annotations
import os
from pathlib import Path
from sqlalchemy import create_engine, inspect
import subprocess

def test_migrations_sqlite(tmp_path):
    db_dir = Path("db"); db_dir.mkdir(exist_ok=True)
    db_path = db_dir / "test_migrations.db"
    if db_path.exists():
        db_path.unlink()

    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    res = subprocess.run(["alembic", "upgrade", "head"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    eng = create_engine(os.environ["DATABASE_URL"], connect_args={"check_same_thread": False})
    tables = set(inspect(eng).get_table_names())
    assert {"civilization", "event", "tag", "eventtag"} <= tables