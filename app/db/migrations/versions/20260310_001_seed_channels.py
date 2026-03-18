# ============================================================
# File: app/db/migrations/versions/20260310_001_seed_channels.py
# Purpose:
#   Seed initial values for channels dictionary table.
#
# Notes:
#   - baseline schema already created the channels table
#   - this migration is intentionally separate from baseline
#   - inserts are idempotent via ON CONFLICT DO NOTHING
# ============================================================

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# Revision identifiers, used by Alembic.
revision = "20260310_001_seed_channels"
down_revision = "c1aaf793328a"
branch_labels = None
depends_on = None


CHANNEL_TYPE_ENUM_NAME = "channel_type_enum"


def upgrade() -> None:
    channel_type_enum = postgresql.ENUM(
        "TELEGRAM",
        "MAX",
        "WEB",
        "TEST_CONSOLE",
        "UNKNOWN",
        name=CHANNEL_TYPE_ENUM_NAME,
        create_type=False,
    )

    channels_table = sa.table(
        "channels",
        sa.column("channel_code", channel_type_enum),
        sa.column("channel_name", sa.Text()),
        sa.column("is_active", sa.Boolean()),
    )

    op.execute(
        postgresql.insert(channels_table).values(
            [
                {
                    "channel_code": "TELEGRAM",
                    "channel_name": "Telegram",
                    "is_active": True,
                },
                {
                    "channel_code": "MAX",
                    "channel_name": "MAX",
                    "is_active": True,
                },
                {
                    "channel_code": "WEB",
                    "channel_name": "Web",
                    "is_active": True,
                },
                {
                    "channel_code": "TEST_CONSOLE",
                    "channel_name": "Test Console",
                    "is_active": True,
                },
                {
                    "channel_code": "UNKNOWN",
                    "channel_name": "Unknown",
                    "is_active": True,
                },
            ]
        ).on_conflict_do_nothing(
            index_elements=["channel_code"]
        )
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            """
            DELETE FROM channels
            WHERE channel_code IN (
                'TELEGRAM',
                'MAX',
                'WEB',
                'TEST_CONSOLE',
                'UNKNOWN'
            )
            """
        )
    )