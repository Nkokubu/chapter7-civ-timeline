"""add civ centroid lat/lon

Revision ID: 24b67d692f65
Revises: 25c46e35a004
Create Date: 2025-08-28 06:47:48.670090
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "24b67d692f65"
down_revision: Union[str, Sequence[str], None] = "25c46e35a004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _civ_table_name() -> str:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())
    if "civilizations" in tables:
        return "civilizations"
    if "civilization" in tables:
        return "civilization"
    raise RuntimeError("Neither 'civilizations' nor 'civilization' table exists.")


def _column_exists(table: str, col: str) -> bool:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    return any(c["name"] == col for c in insp.get_columns(table))


def upgrade() -> None:
    t = _civ_table_name()
    # Add columns if they don't already exist
    with op.batch_alter_table(t) as batch_op:
        if not _column_exists(t, "latitude"):
            batch_op.add_column(sa.Column("latitude", sa.Float(), nullable=True))
        if not _column_exists(t, "longitude"):
            batch_op.add_column(sa.Column("longitude", sa.Float(), nullable=True))


def downgrade() -> None:
    t = _civ_table_name()
    with op.batch_alter_table(t) as batch_op:
        # Drop only if present
        if _column_exists(t, "longitude"):
            batch_op.drop_column("longitude")
        if _column_exists(t, "latitude"):
            batch_op.drop_column("latitude")
