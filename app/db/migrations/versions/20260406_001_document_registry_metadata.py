from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# Revision identifiers, used by Alembic.
revision = "docregmeta_20260406"
down_revision = "20260310_001_seed_channels"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document_registry",
        sa.Column("document_number", sa.Text(), nullable=True),
    )
    op.add_column(
        "document_registry",
        sa.Column("document_date", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "document_registry",
        sa.Column("service_name_full", sa.Text(), nullable=True),
    )
    op.add_column(
        "document_registry",
        sa.Column("service_name_short", sa.Text(), nullable=True),
    )
    op.add_column(
        "document_registry",
        sa.Column("primary_measure_code", sa.Text(), nullable=True),
    )

    op.create_index(
        "idx_document_registry_document_number",
        "document_registry",
        ["document_number"],
        unique=False,
    )
    op.create_index(
        "idx_document_registry_document_date",
        "document_registry",
        ["document_date"],
        unique=False,
    )
    op.create_index(
        "idx_document_registry_primary_measure_code",
        "document_registry",
        ["primary_measure_code"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "idx_document_registry_primary_measure_code",
        table_name="document_registry",
    )
    op.drop_index(
        "idx_document_registry_document_date",
        table_name="document_registry",
    )
    op.drop_index(
        "idx_document_registry_document_number",
        table_name="document_registry",
    )

    op.drop_column("document_registry", "primary_measure_code")
    op.drop_column("document_registry", "service_name_short")
    op.drop_column("document_registry", "service_name_full")
    op.drop_column("document_registry", "document_date")
    op.drop_column("document_registry", "document_number")