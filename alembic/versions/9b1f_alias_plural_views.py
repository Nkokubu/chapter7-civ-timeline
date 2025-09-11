"""alias plural table names as views (SQLite convenience)

Revision ID: 9b1fa1ias123
Revises: 00a5eaa230a8
Create Date: 2025-09-10 18:25:00
"""
from typing import Sequence, Union
from alembic import op
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = "9b1fa1ias123"
down_revision: Union[str, Sequence[str], None] = "00a5eaa230a8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _exists(bind, name: str) -> bool:
    # true if a table or view with this name exists
    row = bind.execute(
        text("SELECT 1 FROM sqlite_master WHERE (type='table' OR type='view') AND name=:n"),
        {"n": name},
    ).fetchone()
    return row is not None


def upgrade() -> None:
    bind = op.get_bind()

    # If the plural table doesn't exist but the singular table does,
    # create a view alias so tests (which query plural) pass.
    if not _exists(bind, "civilizations") and _exists(bind, "civilization"):
        bind.execute(text("CREATE VIEW civilizations AS SELECT * FROM civilization"))

    # (Most installs already have 'events' table; include the alias just in case.)
    if not _exists(bind, "events") and _exists(bind, "event"):
        bind.execute(text("CREATE VIEW events AS SELECT * FROM event"))


def downgrade() -> None:
    bind = op.get_bind()
    # drop views if present (won't touch real tables)
    bind.execute(text("DROP VIEW IF EXISTS civilizations"))
    bind.execute(text("DROP VIEW IF EXISTS events"))
