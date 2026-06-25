from __future__ import annotations

from alembic import op


# Revision identifiers, used by Alembic.
# Keep the revision id <= 32 chars because the current alembic_version.version_num
# column in this project is VARCHAR(32).
revision = "evptr_20260522_001"
down_revision = "service_registry_20260517"
branch_labels = None
depends_on = None


_NEW_CONSTRAINT_NAME = "chk_answer_evidence_document_or_one_precise_pointer"
_OLD_CONSTRAINT_NAME = "chk_answer_evidence_exactly_one_pointer"


_NEW_CHECK_SQL = """
(
    (
        (CASE WHEN block_id IS NOT NULL THEN 1 ELSE 0 END) +
        (CASE WHEN table_id IS NOT NULL THEN 1 ELSE 0 END) +
        (CASE WHEN table_row_id IS NOT NULL THEN 1 ELSE 0 END) +
        (CASE WHEN legal_fact_id IS NOT NULL THEN 1 ELSE 0 END)
    ) = 1
)
OR
(
    document_id IS NOT NULL
    AND block_id IS NULL
    AND table_id IS NULL
    AND table_row_id IS NULL
    AND legal_fact_id IS NULL
)
"""


_OLD_CHECK_SQL = """
(
    (CASE WHEN document_id IS NOT NULL THEN 1 ELSE 0 END) +
    (CASE WHEN block_id IS NOT NULL THEN 1 ELSE 0 END) +
    (CASE WHEN table_id IS NOT NULL THEN 1 ELSE 0 END) +
    (CASE WHEN table_row_id IS NOT NULL THEN 1 ELSE 0 END) +
    (CASE WHEN legal_fact_id IS NOT NULL THEN 1 ELSE 0 END)
) = 1
"""


def _drop_existing_answer_evidence_pointer_checks() -> None:
    """
    Remove old and new pointer-shape checks on answer_evidence_items.

    PostgreSQL can truncate long names, and SQLAlchemy naming conventions can add
    table prefixes. Therefore we drop by safe name patterns instead of one exact
    name only. The migration is intentionally tolerant so it can be rerun after a
    previously failed attempt where DDL may or may not have been rolled back.
    """
    op.execute(
        """
        DO $$
        DECLARE
            r record;
        BEGIN
            FOR r IN
                SELECT conname
                FROM pg_constraint
                WHERE conrelid = 'answer_evidence_items'::regclass
                  AND contype = 'c'
                  AND (
                        conname = 'chk_answer_evidence_exactly_one_pointer'
                     OR conname = 'ck_answer_evidence_items_chk_answer_evidence_exactly_one_pointer'
                     OR conname LIKE 'ck_answer_evidence_items_chk_answer_evidence_exactly_on_%'
                     OR conname = 'chk_answer_evidence_document_or_one_precise_pointer'
                     OR conname = 'ck_answer_evidence_items_chk_answer_evidence_document_or_one_precise_pointer'
                     OR conname LIKE 'ck_answer_evidence_items_chk_answer_evidence_document_or_%'
                  )
            LOOP
                EXECUTE format('ALTER TABLE answer_evidence_items DROP CONSTRAINT %I', r.conname);
            END LOOP;
        END $$;
        """
    )


def upgrade() -> None:
    """
    Allow an evidence row to store document_id as document context together with
    one precise pointer such as table_row_id.
    """
    _drop_existing_answer_evidence_pointer_checks()
    op.create_check_constraint(
        _NEW_CONSTRAINT_NAME,
        "answer_evidence_items",
        _NEW_CHECK_SQL,
    )


def downgrade() -> None:
    _drop_existing_answer_evidence_pointer_checks()
    op.create_check_constraint(
        _OLD_CONSTRAINT_NAME,
        "answer_evidence_items",
        _OLD_CHECK_SQL,
    )
