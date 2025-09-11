"""add sources and links

Revision ID: 7344c619d220
Revises: 2fb73dd75d46
Create Date: 2025-09-06 06:47:52.336849

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes as sqltypes

# revision identifiers, used by Alembic.
revision: str = '7344c619d220'
down_revision: Union[str, Sequence[str], None] = "2fb73dd75d46"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("author", sa.String(), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("url", sa.String(), nullable=True),
        sa.Column("publisher", sa.String(), nullable=True),
        sa.Column("note", sa.String(), nullable=True),
    )
    with op.batch_alter_table("sources") as batch_op:
        batch_op.create_index(batch_op.f("ix_sources_key"), ["key"], unique=True)

    op.create_table(
        "civ_source_link",
        sa.Column("civ_id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["civ_id"], ["civilizations.id"]),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"]),
        sa.PrimaryKeyConstraint("civ_id", "source_id"),
    )

    op.create_table(
        "event_source_link",
        sa.Column("event_id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"]),
        sa.PrimaryKeyConstraint("event_id", "source_id"),
    )

def downgrade() -> None:
    op.drop_table("event_source_link")
    op.drop_table("civ_source_link")
    with op.batch_alter_table("sources") as batch_op:
        batch_op.drop_index(batch_op.f("ix_sources_key"))
    op.drop_table("sources")
