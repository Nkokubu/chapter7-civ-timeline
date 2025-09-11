# scripts/backup_db.py
from __future__ import annotations
import os, sqlite3, time
from pathlib import Path

def _detect_sqlite_path() -> Path:
    # Try importing the engine your app actually uses
    try:
        from app.db.session import engine
        url = engine.url
        if url.get_backend_name() == "sqlite":
            db = url.database  # may be relative or absolute
            if db:
                return Path(db).resolve()
    except Exception:
        pass

    # Fallback: use DATABASE_URL if set
    url = os.getenv("DATABASE_URL", "")
    if url.startswith("sqlite:///"):
        p = url.replace("sqlite:///", "")
        return Path(p).resolve()

    # Final fallback: your previous default
    return Path("db/civ.db").resolve()

def backup_sqlite(src: Path, dest_dir: Path, keep: int = 14) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = dest_dir / f"backup_{stamp}.sqlite3"
    with sqlite3.connect(str(src)) as src_conn, sqlite3.connect(str(dst)) as dst_conn:
        src_conn.backup(dst_conn)
    files = sorted(dest_dir.glob("backup_*.sqlite3"))
    for old in files[:-keep]:
        try: old.unlink()
        except Exception: pass
    return dst

if __name__ == "__main__":
    src = _detect_sqlite_path()
    if not src.exists():
        raise SystemExit(f"DB not found: {src}")
    dst = backup_sqlite(src, Path("backups"))
    print(f"Backup created: {dst}")

