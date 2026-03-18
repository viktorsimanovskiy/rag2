# ============================================================
# File: app/services/ingestion/document_publisher.py
# Purpose:
#   Publish fully parsed, enriched, and QC-approved documents into
#   the active RAG knowledge base using a staging-first strategy.
#
# Responsibilities:
#   - create/update document registry entry
#   - write blocks, tables, rows, legal facts, aliases
#   - ensure publication is atomic
#   - replace old active revision when business rules require it
#   - prevent partial publication
#
# Design principles:
#   - single transactional publication
#   - staging-first, activate-last
#   - conservative replacement logic
#   - idempotent where possible
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import Select, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ============================================================
# PLACEHOLDER IMPORTS
# Replace these imports with actual ORM models in the project.
# ============================================================
try:
    from app.db.models.documents import (
        DocumentRegistry,
        DocumentBlock,
        DocumentTable,
        DocumentTableRow,
        LegalFact,
        MeasureAlias,
    )  # pragma: no cover
except Exception:  # pragma: no cover
    DocumentRegistry = None  # type: ignore
    DocumentBlock = None  # type: ignore
    DocumentTable = None  # type: ignore
    DocumentTableRow = None  # type: ignore
    LegalFact = None  # type: ignore
    MeasureAlias = None  # type: ignore


# ============================================================
# DTOs
# ============================================================

@dataclass(slots=True)
class DetectedFileInfo:
    file_path: str
    original_filename: str
    extension: str
    mime_type: str
    file_size_bytes: int
    file_hash: str


@dataclass(slots=True)
class NormalizationResult:
    normalized_text: str
    normalized_content_hash: str
    detected_language_code: Optional[str]
    parser_payload_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractionResult:
    document_title: Optional[str]
    doc_uid_base: Optional[str]
    revision_date: Optional[datetime]

    blocks: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    table_rows: list[dict[str, Any]] = field(default_factory=list)

    extraction_payload_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SemanticEnrichmentResult:
    source_authority: Optional[str]
    document_type: Optional[str]
    measure_codes: list[str] = field(default_factory=list)
    legal_facts: list[dict[str, Any]] = field(default_factory=list)
    aliases: list[dict[str, Any]] = field(default_factory=list)
    enrichment_payload_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QcResult:
    passed: bool
    error_code: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    metrics_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentIngestionInput:
    file_path: str
    original_filename: str
    source_type: str
    uploaded_by: Optional[str] = None
    metadata_json: dict[str, Any] = field(default_factory=dict)
    force_reingest: bool = False
    parser_version: str = "parser_v1"
    schema_version: str = "schema_v1"


@dataclass(slots=True)
class PublishInput:
    ingestion_job_id: UUID
    file_info: DetectedFileInfo
    normalized_result: NormalizationResult
    extraction_result: ExtractionResult
    enrichment_result: SemanticEnrichmentResult
    qc_result: QcResult
    input_payload: DocumentIngestionInput


@dataclass(slots=True)
class PublishResult:
    document_id: UUID
    status: str
    publish_payload_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PublicationPlan:
    """
    Internal publication decision.
    """
    target_document_id: UUID
    replace_document_ids: list[UUID] = field(default_factory=list)
    mode: str = "insert_new_active"


# ============================================================
# Exceptions
# ============================================================

class DocumentPublisherError(Exception):
    """Base publisher error."""


class DocumentPublisherDependencyError(DocumentPublisherError):
    """Raised when required ORM models are unavailable."""


class DocumentPublicationValidationError(DocumentPublisherError):
    """Raised when publication input is invalid."""


class DocumentPublicationConflictError(DocumentPublisherError):
    """Raised when active revision conflict cannot be resolved safely."""


# ============================================================
# Publisher
# ============================================================

class DocumentPublisher:
    """
    Transactional publisher for active knowledge base.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def publish(
        self,
        payload: PublishInput,
    ) -> PublishResult:
        """
        Publish one fully prepared document into active storage.

        Flow:
        1. Validate input
        2. Build publication plan
        3. Create/replace document registry row
        4. Write dependent objects
        5. Deactivate replaced revisions
        6. Commit atomically
        """
        self._ensure_required_dependencies()
        self._validate_publish_input(payload)

        plan = await self._build_publication_plan(payload)

        try:
            document = await self._upsert_document_registry(
                payload=payload,
                plan=plan,
            )

            await self._replace_document_children(
                document_id=plan.target_document_id,
                payload=payload,
            )

            if plan.replace_document_ids:
                await self._deactivate_replaced_documents(
                    replace_document_ids=plan.replace_document_ids,
                    replacement_document_id=plan.target_document_id,
                )

            await self.db.commit()
            await self.db.refresh(document)

            publish_payload_json = {
                "publication_mode": plan.mode,
                "replaced_document_ids": [str(x) for x in plan.replace_document_ids],
                "blocks_count": len(payload.extraction_result.blocks or []),
                "tables_count": len(payload.extraction_result.tables or []),
                "table_rows_count": len(payload.extraction_result.table_rows or []),
                "legal_facts_count": len(payload.enrichment_result.legal_facts or []),
                "aliases_count": len(payload.enrichment_result.aliases or []),
                "file_hash": payload.file_info.file_hash,
                "content_hash": payload.normalized_result.normalized_content_hash,
            }

            logger.info(
                "Document published",
                extra={
                    "document_id": str(plan.target_document_id),
                    "publication_mode": plan.mode,
                    "replaced_count": len(plan.replace_document_ids),
                },
            )

            return PublishResult(
                document_id=plan.target_document_id,
                status="published",
                publish_payload_json=publish_payload_json,
            )

        except Exception:
            await self.db.rollback()
            logger.exception(
                "Document publication failed",
                extra={
                    "ingestion_job_id": str(payload.ingestion_job_id),
                    "doc_uid_base": payload.extraction_result.doc_uid_base,
                    "revision_date": (
                        payload.extraction_result.revision_date.isoformat()
                        if payload.extraction_result.revision_date else None
                    ),
                },
            )
            raise

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _ensure_required_dependencies(self) -> None:
        required = [
            DocumentRegistry,
            DocumentBlock,
            DocumentTable,
            DocumentTableRow,
            LegalFact,
            MeasureAlias,
        ]
        if any(model is None for model in required):
            raise DocumentPublisherDependencyError(
                "Document ORM models are required for document publication."
            )

    def _validate_publish_input(self, payload: PublishInput) -> None:
        if payload is None:
            raise DocumentPublicationValidationError("PublishInput must not be None.")

        if not payload.qc_result.passed:
            raise DocumentPublicationValidationError(
                "Cannot publish document that did not pass QC."
            )

        if not payload.file_info.file_hash:
            raise DocumentPublicationValidationError("file_hash is required.")

        if not payload.normalized_result.normalized_content_hash:
            raise DocumentPublicationValidationError("normalized_content_hash is required.")

    # ---------------------------------------------------------
    # Publication plan
    # ---------------------------------------------------------

    async def _build_publication_plan(
        self,
        payload: PublishInput,
    ) -> PublicationPlan:
        """
        Determine whether we:
        - publish a brand new active document
        - replace existing active revision(s)
        - reuse existing inactive/current target slot (not done here for safety)
        """
        doc_uid_base = (payload.extraction_result.doc_uid_base or "").strip() or None
        revision_date = payload.extraction_result.revision_date

        # If we cannot identify the document family safely, create a new active row.
        if not doc_uid_base:
            return PublicationPlan(
                target_document_id=uuid4(),
                replace_document_ids=[],
                mode="insert_new_active_unbound",
            )

        active_docs = await self._find_active_documents_by_uid(doc_uid_base)

        # Same content already active -> conflict-safe duplicate.
        for doc in active_docs:
            if getattr(doc, "content_hash", None) == payload.normalized_result.normalized_content_hash:
                raise DocumentPublicationConflictError(
                    "Active document with the same doc_uid_base and content_hash already exists."
                )

        if not active_docs:
            return PublicationPlan(
                target_document_id=uuid4(),
                replace_document_ids=[],
                mode="insert_new_active",
            )

        # Business rule for current project:
        # "new revision arrives -> old active revision(s) are deactivated"
        # If a revision date exists, prefer replacing same family.
        replace_ids = [getattr(doc, "document_id") for doc in active_docs]

        mode = "replace_active_revision"
        if revision_date is None:
            mode = "replace_active_revision_without_revision_date"

        return PublicationPlan(
            target_document_id=uuid4(),
            replace_document_ids=replace_ids,
            mode=mode,
        )

    async def _find_active_documents_by_uid(
        self,
        doc_uid_base: str,
    ) -> list[Any]:
        stmt: Select[Any] = select(DocumentRegistry).where(
            DocumentRegistry.doc_uid_base == doc_uid_base,
            DocumentRegistry.status == "active",
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # ---------------------------------------------------------
    # Registry upsert
    # ---------------------------------------------------------

    async def _upsert_document_registry(
        self,
        *,
        payload: PublishInput,
        plan: PublicationPlan,
    ) -> Any:
        """
        Create a new active document row.
        We intentionally do not mutate an old active row in place.
        This keeps audit trail simpler and safer.
        """
        document = DocumentRegistry(
            document_id=plan.target_document_id,
            ingestion_job_id=payload.ingestion_job_id,
            status="active",

            doc_uid_base=payload.extraction_result.doc_uid_base,
            revision_date=payload.extraction_result.revision_date,

            document_name=payload.extraction_result.document_title,
            document_type=payload.enrichment_result.document_type,
            source_type=payload.input_payload.source_type,
            source_authority=payload.enrichment_result.source_authority,

            file_path=payload.file_info.file_path,
            original_filename=payload.file_info.original_filename,
            file_hash=payload.file_info.file_hash,
            content_hash=payload.normalized_result.normalized_content_hash,

            mime_type=payload.file_info.mime_type,
            extension=payload.file_info.extension,
            file_size_bytes=payload.file_info.file_size_bytes,

            parser_version=payload.input_payload.parser_version,
            schema_version=payload.input_payload.schema_version,

            detected_language_code=payload.normalized_result.detected_language_code,

            publication_payload_json={
                "publication_mode": plan.mode,
                "parser_payload_json": payload.normalized_result.parser_payload_json,
                "extraction_payload_json": payload.extraction_result.extraction_payload_json,
                "enrichment_payload_json": payload.enrichment_result.enrichment_payload_json,
                "qc_metrics_json": payload.qc_result.metrics_json,
                "warnings": payload.qc_result.warnings,
                "input_metadata": payload.input_payload.metadata_json,
                "uploaded_by": payload.input_payload.uploaded_by,
            },

            created_at=self._utcnow(),
            updated_at=self._utcnow(),
        )

        self.db.add(document)
        await self.db.flush()
        return document

    # ---------------------------------------------------------
    # Child object replacement
    # ---------------------------------------------------------

    async def _replace_document_children(
        self,
        *,
        document_id: UUID,
        payload: PublishInput,
    ) -> None:
        """
        Replace all children for the target document atomically.
        Safe because target document is newly created in current design.
        """
        await self._write_blocks(
            document_id=document_id,
            blocks=payload.extraction_result.blocks or [],
        )

        table_id_map = await self._write_tables(
            document_id=document_id,
            tables=payload.extraction_result.tables or [],
        )

        await self._write_table_rows(
            document_id=document_id,
            rows=payload.extraction_result.table_rows or [],
            table_id_map=table_id_map,
        )

        await self._write_legal_facts(
            document_id=document_id,
            legal_facts=payload.enrichment_result.legal_facts or [],
        )

        await self._write_aliases(
            document_id=document_id,
            aliases=payload.enrichment_result.aliases or [],
        )

    async def _write_blocks(
        self,
        *,
        document_id: UUID,
        blocks: list[dict[str, Any]],
    ) -> None:
        for idx, block in enumerate(blocks, start=1):
            model = DocumentBlock(
                block_id=uuid4(),
                document_id=document_id,
                block_order=self._int_or_default(block.get("block_order"), idx),
                block_type=self._str_or_none(block.get("block_type")) or "paragraph",

                content_raw=self._str_or_none(block.get("content_raw")),
                content_clean=(
                    self._str_or_none(block.get("content_clean"))
                    or self._str_or_none(block.get("content"))
                    or self._str_or_none(block.get("text"))
                ),

                chapter=self._str_or_none(block.get("chapter")),
                section_number=self._str_or_none(block.get("section_number")),
                clause_number=self._str_or_none(block.get("clause_number")),
                appendix_number=self._str_or_none(block.get("appendix_number")),
                table_number=self._str_or_none(block.get("table_number")),

                citation_json=self._dict_or_empty(block.get("citation_json")),
                metadata_json=self._dict_or_empty(block.get("metadata_json")),
            )
            self.db.add(model)

        await self.db.flush()

    async def _write_tables(
        self,
        *,
        document_id: UUID,
        tables: list[dict[str, Any]],
    ) -> dict[str, UUID]:
        """
        Returns mapping:
            original_table_id (string form) -> new persisted UUID
        """
        table_id_map: dict[str, UUID] = {}

        for idx, table in enumerate(tables, start=1):
            new_table_id = uuid4()
            original_table_id = table.get("table_id")
            if original_table_id is not None:
                table_id_map[str(original_table_id)] = new_table_id

            model = DocumentTable(
                table_id=new_table_id,
                document_id=document_id,

                table_number=self._str_or_none(table.get("table_number")) or str(idx),
                appendix_number=self._str_or_none(table.get("appendix_number")),
                table_type=self._str_or_none(table.get("table_type")) or "other",
                table_title=self._str_or_none(table.get("table_title")),

                summary=self._str_or_none(table.get("summary")),
                header_schema_json=self._dict_or_empty(table.get("header_schema_json")),
                rows_count=self._int_or_default(table.get("rows_count"), 0),
                markdown_preview=self._str_or_none(table.get("markdown_preview")),

                citation_json=self._dict_or_empty(table.get("citation_json")),
                metadata_json=self._dict_or_empty(table.get("metadata_json")),
            )
            self.db.add(model)

        await self.db.flush()
        return table_id_map

    async def _write_table_rows(
        self,
        *,
        document_id: UUID,
        rows: list[dict[str, Any]],
        table_id_map: dict[str, UUID],
    ) -> None:
        for idx, row in enumerate(rows, start=1):
            source_table_id = row.get("table_id")
            persisted_table_id = table_id_map.get(str(source_table_id)) if source_table_id is not None else None

            model = DocumentTableRow(
                row_id=uuid4(),
                document_id=document_id,
                table_id=persisted_table_id,

                row_order=self._int_or_default(row.get("row_order"), idx),
                row_json=self._dict_or_empty(row.get("row_json")),
                normalized_row_json=self._dict_or_empty(row.get("normalized_row_json")),
                row_summary=self._str_or_none(row.get("row_summary")),

                citation_json=self._dict_or_empty(row.get("citation_json")),
                metadata_json=self._dict_or_empty(row.get("metadata_json")),
            )
            self.db.add(model)

        await self.db.flush()

    async def _write_legal_facts(
        self,
        *,
        document_id: UUID,
        legal_facts: list[dict[str, Any]],
    ) -> None:
        for fact in legal_facts:
            model = LegalFact(
                fact_id=uuid4(),
                document_id=document_id,

                fact_type=self._str_or_none(fact.get("fact_type")) or "other",
                measure_code=self._str_or_none(fact.get("measure_code")),
                subject_category=self._str_or_none(fact.get("subject_category")),

                condition_json=self._dict_or_empty(fact.get("condition_json")),
                value_json=self._dict_or_empty(fact.get("value_json")),
                validity_note=self._str_or_none(fact.get("validity_note")),

                citation_json=self._dict_or_empty(fact.get("citation_json")),
                metadata_json=self._dict_or_empty(fact.get("metadata_json")),
            )
            self.db.add(model)

        await self.db.flush()

    async def _write_aliases(
        self,
        *,
        document_id: UUID,
        aliases: list[dict[str, Any]],
    ) -> None:
        """
        For current stage we store aliases bound to document.
        Later alias governance can be centralized if needed.
        """
        for alias in aliases:
            model = MeasureAlias(
                alias_id=uuid4(),
                document_id=document_id,
                alias=self._str_or_none(alias.get("alias")) or "",
                measure_code=self._str_or_none(alias.get("measure_code")),
                canonical_name=self._str_or_none(alias.get("canonical_name")),
                metadata_json=self._dict_or_empty(alias.get("metadata_json")),
            )
            self.db.add(model)

        await self.db.flush()

    # ---------------------------------------------------------
    # Revision deactivation
    # ---------------------------------------------------------

    async def _deactivate_replaced_documents(
        self,
        *,
        replace_document_ids: list[UUID],
        replacement_document_id: UUID,
    ) -> None:
        for document_id in replace_document_ids:
            stmt: Select[Any] = select(DocumentRegistry).where(
                DocumentRegistry.document_id == document_id
            )
            result = await self.db.execute(stmt)
            doc = result.scalar_one_or_none()
            if doc is None:
                continue

            doc.status = "replaced"
            doc.updated_at = self._utcnow()

            publication_payload = dict(getattr(doc, "publication_payload_json", {}) or {})
            publication_payload["replaced_by_document_id"] = str(replacement_document_id)
            publication_payload["replaced_at"] = self._utcnow().isoformat()
            doc.publication_payload_json = publication_payload

        await self.db.flush()

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _str_or_none(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _dict_or_empty(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {}

    def _int_or_default(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)