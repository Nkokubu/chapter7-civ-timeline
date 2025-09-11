# alembic/env.py

import os
import sys
from pathlib import Path
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlmodel import SQLModel  # ✅ get SQLModel from the library, not your app
from app.models import civilization, event, source  # noqa: F401 (side-effect import to populate metadata)
# target_metadata = SQLModel.metadata

# Ensure project root is on sys.path (so "app" imports work)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import your model modules so their tables are registered on SQLModel.metadata
# (Side-effect imports; we don't use the names directly.)
from app.models import civilization, event  # noqa: F401

# ---- Alembic config ----
config = context.config

# Allow tests / env to override the DB URL; default to your dev SQLite file
db_url = os.getenv("DATABASE_URL", "sqlite:///db/civ.db")
config.set_main_option("sqlalchemy.url", db_url)

# Configure logging from alembic.ini if present
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Point Alembic at SQLModel's metadata (now populated by the imports above)
target_metadata = SQLModel.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # render_as_batch helps with SQLite ALTER TABLE migrations (optional)
        render_as_batch=True if url.startswith("sqlite") else False,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True if db_url.startswith("sqlite") else False,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
