# ============================================================
# File: app/services/ingestion/document_ingestion_pipeline.py
# Purpose:
#   Central orchestration pipeline for document ingestion into the RAG system.
#
# Responsibilities:
#   - accept a source document
#   - identify document format
#   - compute file/content hashes
#   - create and manage ingestion job lifecycle
#   - normalize source content
#   - extract structural objects (blocks, tables, rows, facts)
#   - run QC checks
#   - publish document only after full successful processing
#
# Design principles:
#   - staging-first, publish-later
#   - idempotent processing
#   - conservative failure handling
#   - no partial publication
#   - production-oriented observability
# ============================================================

from __future__ import annotations

import hashlib
import logging
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ============================================================
# PLACEHOLDER IMPORTS
# Replace these imports with actual project implementations.
# ============================================================

try:
    from app.db.models.documents import DocumentRegistry, IngestionJob  # pragma: no cover
except Exception:  # pragma: no cover
    DocumentRegistry = None  # type: ignore
    IngestionJob = None  # type: ignore


# ============================================================
# Protocols / interfaces
# ============================================================

class DocumentNormalizerProtocol:
    async def normalize(self, payload: "NormalizationInput") -> "NormalizationResult":
        raise NotImplementedError


class StructureExtractorProtocol:
    async def extract(self, payload: "ExtractionInput") -> "ExtractionResult":
        raise NotImplementedError


class SemanticEnricherProtocol:
    async def enrich(self, payload: "SemanticEnrichmentInput") -> "SemanticEnrichmentResult":
        raise NotImplementedError


class StructuralQcProtocol:
    async def run_checks(self, payload: "QcInput") -> "QcResult":
        raise NotImplementedError


class DocumentPublisherProtocol:
    async def publish(self, payload: "PublishInput") -> "PublishResult":
        raise NotImplementedError


# ============================================================
# Exceptions
# ============================================================

class DocumentIngestionError(Exception):
    """Base ingestion pipeline error."""


class IngestionValidationError(DocumentIngestionError):
    """Raised when input is invalid."""


class IngestionDependencyError(DocumentIngestionError):
    """Raised when required DB models or services are unavailable."""


class IngestionDuplicateError(DocumentIngestionError):
    """Raised when an already active/current document is detected."""


# ============================================================
# DTOs
# ============================================================

@dataclass(slots=True)
class DocumentIngestionInput:
    """
    Input payload for document ingestion.
    """
    file_path: str
    original_filename: str

    source_type: str
    uploaded_by: Optional[str] = None
    metadata_json: dict[str, Any] = field(default_factory=dict)

    force_reingest: bool = False
    parser_version: str = "parser_v1"
    schema_version: str = "schema_v1"


@dataclass(slots=True)
class DetectedFileInfo:
    """
    Physical file metadata detected before ingestion.
    """
    file_path: str
    original_filename: str
    extension: str
    mime_type: str
    file_size_bytes: int
    file_hash: str


@dataclass(slots=True)
class NormalizationInput:
    """
    Input for source normalization layer.
    """
    file_path: str
    original_filename: str
    extension: str
    mime_type: str
    source_type: str
    metadata_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizationResult:
    """
    Result of normalization.
    """
    normalized_text: str
    normalized_content_hash: str
    detected_language_code: Optional[str]
    parser_payload_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractionInput:
    """
    Input for structural extraction layer.
    """
    file_path: str
    original_filename: str
    normalized_text: str
    source_type: str
    parser_payload_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractionResult:
    """
    Result of structural extraction.
    """
    document_title: Optional[str]
    doc_uid_base: Optional[str]
    revision_date: Optional[datetime]

    blocks: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    table_rows: list[dict[str, Any]] = field(default_factory=list)

    extraction_payload_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SemanticEnrichmentInput:
    """
    Input for semantic enrichment layer.
    """
    normalized_text: str
    extraction_result: ExtractionResult
    source_type: str


@dataclass(slots=True)
class SemanticEnrichmentResult:
    """
    Result of semantic enrichment.
    """
    source_authority: Optional[str]
    document_type: Optional[str]
    measure_codes: list[str] = field(default_factory=list)
    legal_facts: list[dict[str, Any]] = field(default_factory=list)
    aliases: list[dict[str, Any]] = field(default_factory=list)
    enrichment_payload_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QcInput:
    """
    Input for structural quality control.
    """
    normalized_result: NormalizationResult
    extraction_result: ExtractionResult
    enrichment_result: SemanticEnrichmentResult


@dataclass(slots=True)
class QcResult:
    """
    Output of QC checks.
    """
    passed: bool
    error_code: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    metrics_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PublishInput:
    """
    Input for publication layer.
    """
    ingestion_job_id: UUID
    file_info: DetectedFileInfo
    normalized_result: NormalizationResult
    extraction_result: ExtractionResult
    enrichment_result: SemanticEnrichmentResult
    qc_result: QcResult
    input_payload: DocumentIngestionInput


@dataclass(slots=True)
class PublishResult:
    """
    Output of publish layer.
    """
    document_id: UUID
    status: str
    publish_payload_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentIngestionResult:
    """
    Final result of ingestion pipeline.
    """
    ingestion_job_id: UUID
    document_id: Optional[UUID]
    status: str
    file_hash: str
    content_hash: Optional[str]
    warnings: list[str] = field(default_factory=list)
    payload_json: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Pipeline
# ============================================================

class DocumentIngestionPipeline:
    """
    Production-oriented orchestrator for document ingestion.
    """

    def __init__(
        self,
        db: AsyncSession,
        *,
        normalizer: DocumentNormalizerProtocol,
        extractor: StructureExtractorProtocol,
        enricher: SemanticEnricherProtocol,
        qc: StructuralQcProtocol,
        publisher: DocumentPublisherProtocol,
    ) -> None:
        self.db = db
        self.normalizer = normalizer
        self.extractor = extractor
        self.enricher = enricher
        self.qc = qc
        self.publisher = publisher

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    async def ingest_document(
        self,
        payload: DocumentIngestionInput,
    ) -> DocumentIngestionResult:
        """
        Main ingestion flow.

        Steps:
        1. Validate input
        2. Detect file info and compute file hash
        3. Create ingestion job in 'received'
        4. Check idempotency / duplicates
        5. Normalize source
        6. Extract structural objects
        7. Enrich semantically
        8. Run QC
        9. Publish into active dataset
        10. Mark job as completed
        """
        self._validate_input(payload)
        self._ensure_required_dependencies()

        file_info = self._detect_file_info(
            file_path=payload.file_path,
            original_filename=payload.original_filename,
        )

        job = await self._create_ingestion_job(
            payload=payload,
            file_info=file_info,
        )

        try:
            await self._mark_job_stage(job, status="running", stage="idempotency_check")

            duplicate_document = await self._find_duplicate_document(file_info.file_hash)
            if duplicate_document is not None and not payload.force_reingest:
                await self._mark_job_completed(
                    job=job,
                    status="skipped_duplicate",
                    stage="completed",
                    payload_json={
                        "reason": "active_duplicate_by_file_hash",
                        "document_id": str(getattr(duplicate_document, "document_id", "")),
                    },
                )
                return DocumentIngestionResult(
                    ingestion_job_id=job.job_id,
                    document_id=getattr(duplicate_document, "document_id", None),
                    status="skipped_duplicate",
                    file_hash=file_info.file_hash,
                    content_hash=None,
                    warnings=["Документ уже существует в активной базе с тем же file_hash."],
                    payload_json={"reason": "duplicate"},
                )

            await self._mark_job_stage(job, status="running", stage="normalization")

            normalized_result = await self.normalizer.normalize(
                NormalizationInput(
                    file_path=file_info.file_path,
                    original_filename=file_info.original_filename,
                    extension=file_info.extension,
                    mime_type=file_info.mime_type,
                    source_type=payload.source_type,
                    metadata_json=payload.metadata_json,
                )
            )

            await self._mark_job_stage(
                job,
                status="running",
                stage="structure_extraction",
                payload_json={
                    "content_hash": normalized_result.normalized_content_hash,
                },
            )

            extraction_result = await self.extractor.extract(
                ExtractionInput(
                    file_path=file_info.file_path,
                    original_filename=file_info.original_filename,
                    normalized_text=normalized_result.normalized_text,
                    source_type=payload.source_type,
                    parser_payload_json=normalized_result.parser_payload_json,
                )
            )

            await self._mark_job_stage(job, status="running", stage="semantic_enrichment")

            enrichment_result = await self.enricher.enrich(
                SemanticEnrichmentInput(
                    normalized_text=normalized_result.normalized_text,
                    extraction_result=extraction_result,
                    source_type=payload.source_type,
                )
            )

            await self._mark_job_stage(job, status="running", stage="quality_control")

            qc_result = await self.qc.run_checks(
                QcInput(
                    normalized_result=normalized_result,
                    extraction_result=extraction_result,
                    enrichment_result=enrichment_result,
                )
            )

            if not qc_result.passed:
                await self._mark_job_failed(
                    job=job,
                    stage="quality_control",
                    error_message=qc_result.error_code or "qc_failed",
                    payload_json={
                        "warnings": qc_result.warnings,
                        "metrics_json": qc_result.metrics_json,
                    },
                )
                return DocumentIngestionResult(
                    ingestion_job_id=job.job_id,
                    document_id=None,
                    status="failed_qc",
                    file_hash=file_info.file_hash,
                    content_hash=normalized_result.normalized_content_hash,
                    warnings=qc_result.warnings,
                    payload_json={
                        "error_code": qc_result.error_code,
                        "metrics_json": qc_result.metrics_json,
                    },
                )

            await self._mark_job_stage(job, status="running", stage="publish")

            publish_result = await self.publisher.publish(
                PublishInput(
                    ingestion_job_id=job.job_id,
                    file_info=file_info,
                    normalized_result=normalized_result,
                    extraction_result=extraction_result,
                    enrichment_result=enrichment_result,
                    qc_result=qc_result,
                    input_payload=payload,
                )
            )

            await self._mark_job_completed(
                job=job,
                status="completed",
                stage="completed",
                payload_json={
                    "document_id": str(publish_result.document_id),
                    "publish_payload_json": publish_result.publish_payload_json,
                    "warnings": qc_result.warnings,
                },
            )

            logger.info(
                "Document ingestion completed",
                extra={
                    "ingestion_job_id": str(job.job_id),
                    "document_id": str(publish_result.document_id),
                    "file_hash": file_info.file_hash,
                    "content_hash": normalized_result.normalized_content_hash,
                },
            )

            return DocumentIngestionResult(
                ingestion_job_id=job.job_id,
                document_id=publish_result.document_id,
                status="completed",
                file_hash=file_info.file_hash,
                content_hash=normalized_result.normalized_content_hash,
                warnings=qc_result.warnings,
                payload_json={
                    "publish_payload_json": publish_result.publish_payload_json,
                    "qc_metrics_json": qc_result.metrics_json,
                    "document_title": extraction_result.document_title,
                    "doc_uid_base": extraction_result.doc_uid_base,
                },
            )

        except Exception as exc:
            logger.exception(
                "Document ingestion failed",
                extra={
                    "ingestion_job_id": str(job.job_id),
                    "file_path": payload.file_path,
                    "original_filename": payload.original_filename,
                },
            )
            await self._mark_job_failed(
                job=job,
                stage=getattr(job, "stage", "unknown"),
                error_message=str(exc),
                payload_json={"exception_type": exc.__class__.__name__},
            )
            raise

    # --------------------------------------------------------
    # Input / file detection
    # --------------------------------------------------------

    def _validate_input(self, payload: DocumentIngestionInput) -> None:
        if not payload.file_path or not payload.file_path.strip():
            raise IngestionValidationError("file_path must not be empty.")

        if not payload.original_filename or not payload.original_filename.strip():
            raise IngestionValidationError("original_filename must not be empty.")

        if not payload.source_type or not payload.source_type.strip():
            raise IngestionValidationError("source_type must not be empty.")

        file_path = Path(payload.file_path)
        if not file_path.exists():
            raise IngestionValidationError(f"File not found: {payload.file_path}")

        if not file_path.is_file():
            raise IngestionValidationError(f"Path is not a file: {payload.file_path}")

    def _detect_file_info(
        self,
        *,
        file_path: str,
        original_filename: str,
    ) -> DetectedFileInfo:
        path = Path(file_path)
        extension = path.suffix.lower().lstrip(".")
        mime_type = mimetypes.guess_type(original_filename)[0] or "application/octet-stream"
        file_size_bytes = path.stat().st_size
        file_hash = self._compute_file_hash(path)

        return DetectedFileInfo(
            file_path=str(path),
            original_filename=original_filename,
            extension=extension,
            mime_type=mime_type,
            file_size_bytes=file_size_bytes,
            file_hash=file_hash,
        )

    def _compute_file_hash(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    # --------------------------------------------------------
    # Ingestion job lifecycle
    # --------------------------------------------------------

    async def _create_ingestion_job(
        self,
        *,
        payload: DocumentIngestionInput,
        file_info: DetectedFileInfo,
    ) -> Any:
        job = IngestionJob(
            job_id=uuid4(),
            status="received",
            stage="received",
            file_path=file_info.file_path,
            original_filename=file_info.original_filename,
            file_hash=file_info.file_hash,
            parser_version=payload.parser_version,
            schema_version=payload.schema_version,
            started_at=self._utcnow(),
            metadata_json={
                "source_type": payload.source_type,
                "uploaded_by": payload.uploaded_by,
                "input_metadata": payload.metadata_json,
                "mime_type": file_info.mime_type,
                "extension": file_info.extension,
                "file_size_bytes": file_info.file_size_bytes,
            },
        )
        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)
        return job

    async def _mark_job_stage(
        self,
        job: Any,
        *,
        status: str,
        stage: str,
        payload_json: Optional[dict[str, Any]] = None,
    ) -> None:
        job.status = status
        job.stage = stage

        if payload_json:
            current = dict(getattr(job, "payload_json", {}) or {})
            current.update(payload_json)
            job.payload_json = current

        await self.db.commit()
        await self.db.refresh(job)

    async def _mark_job_completed(
        self,
        *,
        job: Any,
        status: str,
        stage: str,
        payload_json: Optional[dict[str, Any]] = None,
    ) -> None:
        job.status = status
        job.stage = stage
        job.finished_at = self._utcnow()

        if payload_json:
            current = dict(getattr(job, "payload_json", {}) or {})
            current.update(payload_json)
            job.payload_json = current

        await self.db.commit()
        await self.db.refresh(job)

    async def _mark_job_failed(
        self,
        *,
        job: Any,
        stage: str,
        error_message: str,
        payload_json: Optional[dict[str, Any]] = None,
    ) -> None:
        job.status = "failed"
        job.stage = stage
        job.error_message = error_message
        job.finished_at = self._utcnow()

        if payload_json:
            current = dict(getattr(job, "payload_json", {}) or {})
            current.update(payload_json)
            job.payload_json = current

        await self.db.commit()
        await self.db.refresh(job)

    # --------------------------------------------------------
    # Idempotency / duplicate checks
    # --------------------------------------------------------

    async def _find_duplicate_document(
        self,
        file_hash: str,
    ) -> Optional[Any]:
        stmt: Select[Any] = select(DocumentRegistry).where(
            DocumentRegistry.file_hash == file_hash,
            DocumentRegistry.status == "active",
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # --------------------------------------------------------
    # Dependency validation
    # --------------------------------------------------------

    def _ensure_required_dependencies(self) -> None:
        if DocumentRegistry is None or IngestionJob is None:
            raise IngestionDependencyError(
                "DocumentRegistry and IngestionJob models are required for ingestion pipeline."
            )

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)