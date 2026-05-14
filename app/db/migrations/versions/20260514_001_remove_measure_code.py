from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# Revision identifiers, used by Alembic.
revision = "remove_measure_code_20260514"
down_revision = "docregmeta_20260406"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Remove the old coarse measure-code layer.

    Reason:
    - primary_measure_code grouped different public services into broad buckets
      such as edv/subsidy/sanatorium;
    - for the 110-regulation corpus this caused ambiguity instead of helping;
    - the next stable anchor must be concrete service identity, not a measure group.
    """
    op.execute("DROP INDEX IF EXISTS idx_document_registry_primary_measure_code")
    op.execute("ALTER TABLE document_registry DROP COLUMN IF EXISTS primary_measure_code")

    op.execute("DROP INDEX IF EXISTS idx_legal_facts_measure_code")
    op.execute("ALTER TABLE legal_facts DROP COLUMN IF EXISTS measure_code")

    op.execute("DROP INDEX IF EXISTS idx_question_events_measure_code")
    op.execute("ALTER TABLE question_events DROP COLUMN IF EXISTS measure_code")

    op.execute("DROP INDEX IF EXISTS idx_quality_aggregates_daily_measure")
    op.execute("ALTER TABLE quality_aggregates_daily DROP CONSTRAINT IF EXISTS uq_quality_aggregates_daily")
    op.execute("ALTER TABLE quality_aggregates_daily DROP COLUMN IF EXISTS measure_code")
    op.create_unique_constraint(
        "uq_quality_aggregates_daily",
        "quality_aggregates_daily",
        ["aggregate_date", "channel_code", "intent_type"],
    )

    op.execute("DROP TABLE IF EXISTS measure_aliases")


def downgrade() -> None:
    op.add_column(
        "document_registry",
        sa.Column("primary_measure_code", sa.Text(), nullable=True),
    )
    op.create_index(
        "idx_document_registry_primary_measure_code",
        "document_registry",
        ["primary_measure_code"],
        unique=False,
    )

    op.add_column(
        "legal_facts",
        sa.Column("measure_code", sa.Text(), nullable=True),
    )
    op.create_index(
        "idx_legal_facts_measure_code",
        "legal_facts",
        ["measure_code"],
        unique=False,
    )

    op.add_column(
        "question_events",
        sa.Column("measure_code", sa.Text(), nullable=True),
    )
    op.create_index(
        "idx_question_events_measure_code",
        "question_events",
        ["measure_code"],
        unique=False,
    )

    op.execute("ALTER TABLE quality_aggregates_daily DROP CONSTRAINT IF EXISTS uq_quality_aggregates_daily")
    op.add_column(
        "quality_aggregates_daily",
        sa.Column("measure_code", sa.Text(), nullable=True),
    )
    op.create_index(
        "idx_quality_aggregates_daily_measure",
        "quality_aggregates_daily",
        ["measure_code"],
        unique=False,
    )
    op.create_unique_constraint(
        "uq_quality_aggregates_daily",
        "quality_aggregates_daily",
        ["aggregate_date", "channel_code", "intent_type", "measure_code"],
    )

    op.create_table(
        "measure_aliases",
        sa.Column("alias_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("alias", sa.Text(), nullable=False),
        sa.Column("measure_code", sa.Text(), nullable=True),
        sa.Column("canonical_name", sa.Text(), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.ForeignKeyConstraint(["document_id"], ["document_registry.document_id"], ondelete="CASCADE"),
        sa.CheckConstraint("alias <> ''", name="chk_measure_aliases_alias_not_empty"),
        sa.UniqueConstraint("document_id", "alias", name="uq_measure_aliases_document_alias"),
    )
    op.create_index("idx_measure_aliases_document_id", "measure_aliases", ["document_id"], unique=False)
    op.create_index("idx_measure_aliases_alias", "measure_aliases", ["alias"], unique=False)
    op.create_index("idx_measure_aliases_measure_code", "measure_aliases", ["measure_code"], unique=False)
