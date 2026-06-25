from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# Revision identifiers, used by Alembic.
revision = "service_registry_20260517"
down_revision = "remove_measure_code_20260514"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Add concrete service registry.

    This is the replacement for the removed coarse measure_code layer:
    one row in service_registry is one concrete public service from
    Актуальный_приказ5.xlsx, not a broad measure group.
    """
    op.create_table(
        "service_registry",
        sa.Column(
            "service_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("service_key", sa.Text(), nullable=False),
        sa.Column("service_name_full", sa.Text(), nullable=False),
        sa.Column("service_name_short", sa.Text(), nullable=False),
        sa.Column("frgu_1", sa.Text(), nullable=True),
        sa.Column("frgu_3", sa.Text(), nullable=True),
        sa.Column("order_details", sa.Text(), nullable=True),
        sa.Column("raw_filename", sa.Text(), nullable=False),
        sa.Column("cleaned_filename", sa.Text(), nullable=False),
        sa.Column("aliases_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("TRUE")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint("service_key <> ''", name="chk_service_registry_service_key_not_empty"),
        sa.CheckConstraint("service_name_full <> ''", name="chk_service_registry_name_full_not_empty"),
        sa.CheckConstraint("service_name_short <> ''", name="chk_service_registry_name_short_not_empty"),
        sa.CheckConstraint("raw_filename <> ''", name="chk_service_registry_raw_filename_not_empty"),
        sa.CheckConstraint("cleaned_filename <> ''", name="chk_service_registry_cleaned_filename_not_empty"),
        sa.UniqueConstraint("service_key", name="uq_service_registry_service_key"),
        sa.UniqueConstraint("raw_filename", name="uq_service_registry_raw_filename"),
        sa.UniqueConstraint("cleaned_filename", name="uq_service_registry_cleaned_filename"),
        sa.UniqueConstraint("frgu_1", name="uq_service_registry_frgu_1"),
        sa.UniqueConstraint("frgu_3", name="uq_service_registry_frgu_3"),
    )
    op.create_index("idx_service_registry_name_short", "service_registry", ["service_name_short"], unique=False)
    op.create_index("idx_service_registry_is_active", "service_registry", ["is_active"], unique=False)

    op.add_column("document_registry", sa.Column("service_key", sa.Text(), nullable=True))
    op.add_column("document_registry", sa.Column("service_frgu_1", sa.Text(), nullable=True))
    op.add_column("document_registry", sa.Column("service_frgu_3", sa.Text(), nullable=True))
    op.create_foreign_key(
        "fk_document_registry_service_key",
        "document_registry",
        "service_registry",
        ["service_key"],
        ["service_key"],
        ondelete="SET NULL",
    )
    op.create_index("idx_document_registry_service_key", "document_registry", ["service_key"], unique=False)
    op.create_index("idx_document_registry_service_frgu_1", "document_registry", ["service_frgu_1"], unique=False)
    op.create_index("idx_document_registry_service_frgu_3", "document_registry", ["service_frgu_3"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_document_registry_service_frgu_3", table_name="document_registry")
    op.drop_index("idx_document_registry_service_frgu_1", table_name="document_registry")
    op.drop_index("idx_document_registry_service_key", table_name="document_registry")
    op.drop_constraint("fk_document_registry_service_key", "document_registry", type_="foreignkey")
    op.drop_column("document_registry", "service_frgu_3")
    op.drop_column("document_registry", "service_frgu_1")
    op.drop_column("document_registry", "service_key")

    op.drop_index("idx_service_registry_is_active", table_name="service_registry")
    op.drop_index("idx_service_registry_name_short", table_name="service_registry")
    op.drop_table("service_registry")
