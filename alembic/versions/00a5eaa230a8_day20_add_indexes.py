"""day20 add indexes (robust to singular/plural table names)"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "00a5eaa230a8"
down_revision: Union[str, Sequence[str], None] = "7344c619d220"  # keep your chain
branch_labels = None
depends_on = None


def _find_table(candidates: list[str]) -> str:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing = set(insp.get_table_names())
    for name in candidates:
        if name in existing:
            return name
    # return empty string instead of raising; caller can skip if not found
    return ""


def _colnames(table: str) -> set[str]:
    if not table:
        return set()
    bind = op.get_bind()
    insp = sa.inspect(bind)
    return {c["name"] for c in insp.get_columns(table)}


def _has_index(table: str, name: str) -> bool:
    if not table:
        return False
    bind = op.get_bind()
    insp = sa.inspect(bind)
    return any(ix["name"] == name for ix in insp.get_indexes(table))


def _ensure_index(table: str, cols: list[str], name: str, unique: bool = False):
    if not table:
        return
    if _has_index(table, name):
        return
    op.create_index(name, table, cols, unique=unique)


def upgrade() -> None:
    events = _find_table(["events", "event"])
    civs   = _find_table(["civilizations", "civilization"])
    srcs   = _find_table(["sources", "source"])

    ev_cols = _colnames(events)
    cv_cols = _colnames(civs)
    sr_cols = _colnames(srcs)

    # events(year), events(civilization_id)
    if "year" in ev_cols:
        _ensure_index(events, ["year"], "ix_events_year", unique=False)
    if "civilization_id" in ev_cols:
        _ensure_index(events, ["civilization_id"], "ix_events_civilization_id", unique=False)

    # civilizations(region), civilizations(slug) if present
    if "region" in cv_cols:
        _ensure_index(civs, ["region"], "ix_civilizations_region", unique=False)
    if "slug" in cv_cols:
        _ensure_index(civs, ["slug"], "ix_civilizations_slug", unique=False)

    # sources(key) helpful lookup (unique if your model enforces it)
    if "key" in sr_cols:
        _ensure_index(srcs, ["key"], "ux_sources_key", unique=True)


def downgrade() -> None:
    # Drop indexes if they exist (don’t error if missing)
    for table in ( _find_table(["events","event"]),
                   _find_table(["civilizations","civilization"]),
                   _find_table(["sources","source"]) ):
        if not table:
            continue
        bind = op.get_bind()
        insp = sa.inspect(bind)
        for ix in insp.get_indexes(table):
            if ix["name"] in {
                "ix_events_year",
                "ix_events_civilization_id",
                "ix_civilizations_region",
                "ix_civilizations_slug",
                "ux_sources_key",
            }:
                op.drop_index(ix["name"], table_name=table)
