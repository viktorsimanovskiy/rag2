# ============================================================
# File: app/db/models/documents.py
# Purpose:
#   ORM models for document ingestion, storage, and retrieval base.
#
# Includes:
#   - IngestionJob
#   - DocumentRegistry
#   - DocumentBlock
#   - DocumentTable
#   - DocumentTableRow
#   - LegalFact
#   - MeasureAlias
#
# Notes:
#   - SQLAlchemy 2.x style
#   - PostgreSQL-oriented
#   - JSONB used for flexible metadata
#   - designed to support staging-safe publication model
# ============================================================

from __future__ import annotations

from datetime import datetime, date, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover
    Vector = None  # type: ignore

from app.db.base import Base


# ============================================================
# Helpers
# ============================================================

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# Ingestion job
# ============================================================

class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        Index("idx_ingestion_jobs_status", "status"),
        Index("idx_ingestion_jobs_stage", "stage"),
        Index("idx_ingestion_jobs_started_at", "started_at"),
        Index("idx_ingestion_jobs_file_hash", "file_hash"),
    )

    job_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    stage: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    file_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    original_filename: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    file_hash: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    parser_version: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    schema_version: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    metadata_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    payload_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("NOW()"),
    )

    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    documents: Mapped[list["DocumentRegistry"]] = relationship(
        "DocumentRegistry",
        back_populates="ingestion_job",
        lazy="selectin",
    )


# ============================================================
# Document registry
# ============================================================

class DocumentRegistry(Base):
    __tablename__ = "document_registry"
    __table_args__ = (
        Index("idx_document_registry_status", "status"),
        Index("idx_document_registry_doc_uid_base", "doc_uid_base"),
        Index("idx_document_registry_revision_date", "revision_date"),
        Index("idx_document_registry_file_hash", "file_hash"),
        Index("idx_document_registry_content_hash", "content_hash"),
        Index("idx_document_registry_document_type", "document_type"),
        Index("idx_document_registry_source_type", "source_type"),
        Index("idx_document_registry_source_authority", "source_authority"),
    )

    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    ingestion_job_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("ingestion_jobs.job_id", ondelete="SET NULL"),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    doc_uid_base: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    revision_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    document_name: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    document_type: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    source_type: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    source_authority: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    file_path: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    original_filename: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    file_hash: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    content_hash: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    mime_type: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    extension: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    file_size_bytes: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        nullable=True,
    )

    parser_version: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    schema_version: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    detected_language_code: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    publication_payload_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("NOW()"),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
        server_default=text("NOW()"),
    )

    ingestion_job: Mapped[Optional["IngestionJob"]] = relationship(
        "IngestionJob",
        back_populates="documents",
        lazy="joined",
    )

    blocks: Mapped[list["DocumentBlock"]] = relationship(
        "DocumentBlock",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
        order_by="DocumentBlock.block_order",
    )

    tables: Mapped[list["DocumentTable"]] = relationship(
        "DocumentTable",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )

    table_rows: Mapped[list["DocumentTableRow"]] = relationship(
        "DocumentTableRow",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )

    legal_facts: Mapped[list["LegalFact"]] = relationship(
        "LegalFact",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )

    aliases: Mapped[list["MeasureAlias"]] = relationship(
        "MeasureAlias",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )


# ============================================================
# Document blocks
# ============================================================

class DocumentBlock(Base):
    __tablename__ = "document_blocks"
    __table_args__ = (
        Index("idx_document_blocks_document_id", "document_id"),
        Index("idx_document_blocks_block_type", "block_type"),
        Index("idx_document_blocks_block_order", "document_id", "block_order"),
        Index("idx_document_blocks_appendix_number", "appendix_number"),
        Index("idx_document_blocks_table_number", "table_number"),
    )

    block_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("document_registry.document_id", ondelete="CASCADE"),
        nullable=False,
    )

    block_order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    block_type: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    content_raw: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    content_clean: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    chapter: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    section_number: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    clause_number: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    appendix_number: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    table_number: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    if Vector is not None:
        embedding: Mapped[Optional[list[float]]] = mapped_column(
            Vector(1536),
            nullable=True,
        )
    else:  # pragma: no cover
        embedding: Mapped[Optional[str]] = mapped_column(
            Text,
            nullable=True,
        )

    citation_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    metadata_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    document: Mapped["DocumentRegistry"] = relationship(
        "DocumentRegistry",
        back_populates="blocks",
        lazy="joined",
    )


# ============================================================
# Document tables
# ============================================================

class DocumentTable(Base):
    __tablename__ = "document_tables"
    __table_args__ = (
        Index("idx_document_tables_document_id", "document_id"),
        Index("idx_document_tables_table_type", "table_type"),
        Index("idx_document_tables_table_number", "table_number"),
        Index("idx_document_tables_appendix_number", "appendix_number"),
    )

    table_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("document_registry.document_id", ondelete="CASCADE"),
        nullable=False,
    )

    table_number: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    appendix_number: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    table_type: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    table_title: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    header_schema_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    rows_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    markdown_preview: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    if Vector is not None:
        embedding: Mapped[Optional[list[float]]] = mapped_column(
            Vector(1536),
            nullable=True,
        )
    else:  # pragma: no cover
        embedding: Mapped[Optional[str]] = mapped_column(
            Text,
            nullable=True,
        )

    citation_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    metadata_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    document: Mapped["DocumentRegistry"] = relationship(
        "DocumentRegistry",
        back_populates="tables",
        lazy="joined",
    )

    rows: Mapped[list["DocumentTableRow"]] = relationship(
        "DocumentTableRow",
        back_populates="table",
        cascade="save-update, merge",
        lazy="selectin",
        order_by="DocumentTableRow.row_order",
    )


# ============================================================
# Document table rows
# ============================================================

class DocumentTableRow(Base):
    __tablename__ = "document_table_rows"
    __table_args__ = (
        Index("idx_document_table_rows_document_id", "document_id"),
        Index("idx_document_table_rows_table_id", "table_id"),
        Index("idx_document_table_rows_row_order", "table_id", "row_order"),
    )

    row_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("document_registry.document_id", ondelete="CASCADE"),
        nullable=False,
    )

    table_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("document_tables.table_id", ondelete="SET NULL"),
        nullable=True,
    )

    row_order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    row_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    normalized_row_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    row_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    if Vector is not None:
        embedding: Mapped[Optional[list[float]]] = mapped_column(
            Vector(1536),
            nullable=True,
        )
    else:  # pragma: no cover
        embedding: Mapped[Optional[str]] = mapped_column(
            Text,
            nullable=True,
        )

    citation_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    metadata_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    document: Mapped["DocumentRegistry"] = relationship(
        "DocumentRegistry",
        back_populates="table_rows",
        lazy="joined",
    )

    table: Mapped[Optional["DocumentTable"]] = relationship(
        "DocumentTable",
        back_populates="rows",
        lazy="joined",
    )


# ============================================================
# Legal facts
# ============================================================

class LegalFact(Base):
    __tablename__ = "legal_facts"
    __table_args__ = (
        Index("idx_legal_facts_document_id", "document_id"),
        Index("idx_legal_facts_fact_type", "fact_type"),
        Index("idx_legal_facts_measure_code", "measure_code"),
        Index("idx_legal_facts_subject_category", "subject_category"),
    )

    fact_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("document_registry.document_id", ondelete="CASCADE"),
        nullable=False,
    )

    fact_type: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    measure_code: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    subject_category: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    condition_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    value_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    validity_note: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    citation_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    metadata_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    document: Mapped["DocumentRegistry"] = relationship(
        "DocumentRegistry",
        back_populates="legal_facts",
        lazy="joined",
    )


# ============================================================
# Measure aliases
# ============================================================

class MeasureAlias(Base):
    __tablename__ = "measure_aliases"
    __table_args__ = (
        CheckConstraint("alias <> ''", name="chk_measure_aliases_alias_not_empty"),
        Index("idx_measure_aliases_document_id", "document_id"),
        Index("idx_measure_aliases_alias", "alias"),
        Index("idx_measure_aliases_measure_code", "measure_code"),
        UniqueConstraint("document_id", "alias", name="uq_measure_aliases_document_alias"),
    )

    alias_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("document_registry.document_id", ondelete="CASCADE"),
        nullable=False,
    )

    alias: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    measure_code: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    canonical_name: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    metadata_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    document: Mapped["DocumentRegistry"] = relationship(
        "DocumentRegistry",
        back_populates="aliases",
        lazy="joined",
    )