# ============================================================
# File: app/services/ingestion/structural_qc_service.py
# Purpose:
#   Structural quality control for parsed and enriched documents
#   before publication into the active RAG knowledge base.
#
# Responsibilities:
#   - validate structural completeness of extracted document objects
#   - detect critical parsing failures and suspicious anomalies
#   - assess whether the document is safe to publish
#   - produce deterministic QC verdict, warnings, and metrics
#
# Design principles:
#   - conservative by default
#   - deterministic checks first
#   - production-oriented explainability
#   - no publication from this layer
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================
# DTOs
# ============================================================

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
    revision_date: Optional[Any]

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
class QcInput:
    normalized_result: NormalizationResult
    extraction_result: ExtractionResult
    enrichment_result: SemanticEnrichmentResult


@dataclass(slots=True)
class QcResult:
    passed: bool
    error_code: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    metrics_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StructuralQcConfig:
    """
    Thresholds for deterministic QC checks.

    These values are intentionally conservative for early production.
    Tune them later using real corpus statistics.
    """
    min_normalized_text_length: int = 200
    min_blocks_count: int = 3
    min_meaningful_blocks_count: int = 2

    max_empty_block_ratio: float = 0.30
    max_other_block_ratio: float = 0.60

    max_tables_without_rows_ratio: float = 0.20
    max_rows_without_table_binding_ratio: float = 0.10

    min_avg_block_text_length: int = 20

    require_title_or_uid: bool = True
    require_some_retrievable_content: bool = True

    fail_on_zero_tables_if_tables_declared: bool = True
    warn_on_zero_legal_facts: bool = False

    policy_version: str = "structural_qc_v1"


# ============================================================
# Service
# ============================================================

class StructuralQcService:
    """
    Deterministic structural QC for parsed documents.

    This service should be called after:
    - normalization
    - structural extraction
    - semantic enrichment

    and before publication into the active dataset.
    """

    def __init__(
        self,
        *,
        config: Optional[StructuralQcConfig] = None,
    ) -> None:
        self.config = config or StructuralQcConfig()

    async def run_checks(
        self,
        payload: QcInput,
    ) -> QcResult:
        """
        Run all structural QC checks and return final verdict.
        """
        self._validate_input(payload)

        warnings: list[str] = []
        errors: list[str] = []

        metrics = self._build_base_metrics(payload)

        # -----------------------------------------------------
        # Critical document-level checks
        # -----------------------------------------------------
        self._check_normalized_text(payload, metrics, errors, warnings)
        self._check_document_identity(payload, metrics, errors, warnings)

        # -----------------------------------------------------
        # Blocks
        # -----------------------------------------------------
        self._check_blocks_presence(payload, metrics, errors, warnings)
        self._check_block_quality(payload, metrics, errors, warnings)
        self._check_block_order_integrity(payload, metrics, errors, warnings)

        # -----------------------------------------------------
        # Tables / rows
        # -----------------------------------------------------
        self._check_tables_and_rows(payload, metrics, errors, warnings)

        # -----------------------------------------------------
        # Semantic enrichment consistency
        # -----------------------------------------------------
        self._check_semantic_enrichment(payload, metrics, errors, warnings)

        # -----------------------------------------------------
        # Final retrievability / usability check
        # -----------------------------------------------------
        self._check_retrievability(payload, metrics, errors, warnings)

        passed = len(errors) == 0
        error_code = None if passed else self._compose_error_code(errors)

        metrics["qc_policy_version"] = self.config.policy_version
        metrics["warnings_count"] = len(warnings)
        metrics["errors_count"] = len(errors)
        metrics["errors"] = errors

        return QcResult(
            passed=passed,
            error_code=error_code,
            warnings=warnings,
            metrics_json=metrics,
        )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _validate_input(
        self,
        payload: QcInput,
    ) -> None:
        if payload is None:
            raise ValueError("QcInput must not be None.")

        if payload.normalized_result is None:
            raise ValueError("normalized_result is required.")

        if payload.extraction_result is None:
            raise ValueError("extraction_result is required.")

        if payload.enrichment_result is None:
            raise ValueError("enrichment_result is required.")

    # ---------------------------------------------------------
    # Base metrics
    # ---------------------------------------------------------

    def _build_base_metrics(
        self,
        payload: QcInput,
    ) -> dict[str, Any]:
        blocks = payload.extraction_result.blocks or []
        tables = payload.extraction_result.tables or []
        rows = payload.extraction_result.table_rows or []
        legal_facts = payload.enrichment_result.legal_facts or []

        return {
            "normalized_text_length": len((payload.normalized_result.normalized_text or "").strip()),
            "blocks_count": len(blocks),
            "tables_count": len(tables),
            "table_rows_count": len(rows),
            "legal_facts_count": len(legal_facts),
            "document_title_present": bool((payload.extraction_result.document_title or "").strip()),
            "doc_uid_base_present": bool((payload.extraction_result.doc_uid_base or "").strip()),
            "revision_date_present": payload.extraction_result.revision_date is not None,
        }

    # ---------------------------------------------------------
    # Document-level checks
    # ---------------------------------------------------------

    def _check_normalized_text(
        self,
        payload: QcInput,
        metrics: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> None:
        text = (payload.normalized_result.normalized_text or "").strip()
        text_len = len(text)

        if text_len == 0:
            errors.append("normalized_text_empty")
            return

        if text_len < self.config.min_normalized_text_length:
            errors.append("normalized_text_too_short")

        if payload.normalized_result.detected_language_code is None:
            warnings.append("detected_language_code_missing")

    def _check_document_identity(
        self,
        payload: QcInput,
        metrics: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> None:
        title_present = bool((payload.extraction_result.document_title or "").strip())
        uid_present = bool((payload.extraction_result.doc_uid_base or "").strip())

        if self.config.require_title_or_uid and not (title_present or uid_present):
            errors.append("document_identity_missing")

        if not title_present:
            warnings.append("document_title_missing")

        if not uid_present:
            warnings.append("doc_uid_base_missing")

        if payload.extraction_result.revision_date is None:
            warnings.append("revision_date_missing")

    # ---------------------------------------------------------
    # Blocks checks
    # ---------------------------------------------------------

    def _check_blocks_presence(
        self,
        payload: QcInput,
        metrics: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> None:
        blocks = payload.extraction_result.blocks or []

        if len(blocks) == 0:
            errors.append("blocks_missing")
            return

        if len(blocks) < self.config.min_blocks_count:
            errors.append("blocks_count_too_low")

    def _check_block_quality(
        self,
        payload: QcInput,
        metrics: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> None:
        blocks = payload.extraction_result.blocks or []

        empty_blocks = 0
        meaningful_blocks = 0
        other_blocks = 0
        total_text_length = 0

        for block in blocks:
            block_text = self._extract_block_text(block)
            block_type = str(block.get("block_type") or "").strip().lower()

            if not block_text:
                empty_blocks += 1
            else:
                meaningful_blocks += 1
                total_text_length += len(block_text)

            if block_type in {"other", "unknown", ""}:
                other_blocks += 1

        total_blocks = len(blocks)
        empty_ratio = empty_blocks / total_blocks if total_blocks > 0 else 1.0
        other_ratio = other_blocks / total_blocks if total_blocks > 0 else 1.0
        avg_block_len = total_text_length / meaningful_blocks if meaningful_blocks > 0 else 0.0

        metrics["empty_blocks_count"] = empty_blocks
        metrics["meaningful_blocks_count"] = meaningful_blocks
        metrics["other_blocks_count"] = other_blocks
        metrics["empty_block_ratio"] = round(empty_ratio, 4)
        metrics["other_block_ratio"] = round(other_ratio, 4)
        metrics["avg_meaningful_block_text_length"] = round(avg_block_len, 2)

        if meaningful_blocks < self.config.min_meaningful_blocks_count:
            errors.append("meaningful_blocks_count_too_low")

        if empty_ratio > self.config.max_empty_block_ratio:
            errors.append("empty_block_ratio_too_high")

        if other_ratio > self.config.max_other_block_ratio:
            warnings.append("other_block_ratio_too_high")

        if meaningful_blocks > 0 and avg_block_len < self.config.min_avg_block_text_length:
            warnings.append("avg_block_text_length_too_low")

    def _check_block_order_integrity(
        self,
        payload: QcInput,
        metrics: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> None:
        blocks = payload.extraction_result.blocks or []

        order_values: list[int] = []
        missing_order_count = 0

        for block in blocks:
            order_value = block.get("block_order")
            if isinstance(order_value, int):
                order_values.append(order_value)
            else:
                missing_order_count += 1

        metrics["blocks_with_missing_order_count"] = missing_order_count

        if missing_order_count == len(blocks) and len(blocks) > 0:
            warnings.append("all_block_order_values_missing")
            return

        if order_values:
            sorted_values = sorted(order_values)
            has_duplicates = len(sorted_values) != len(set(sorted_values))
            is_monotonic = order_values == sorted_values

            metrics["block_order_has_duplicates"] = has_duplicates
            metrics["block_order_is_monotonic"] = is_monotonic

            if has_duplicates:
                errors.append("block_order_duplicates_detected")

            if not is_monotonic:
                warnings.append("block_order_not_monotonic")

    # ---------------------------------------------------------
    # Tables / rows checks
    # ---------------------------------------------------------

    def _check_tables_and_rows(
        self,
        payload: QcInput,
        metrics: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> None:
        tables = payload.extraction_result.tables or []
        rows = payload.extraction_result.table_rows or []
        extraction_payload = payload.extraction_result.extraction_payload_json or {}

        declared_table_count = extraction_payload.get("declared_table_count")

        if declared_table_count is not None:
            metrics["declared_table_count"] = declared_table_count
            if (
                self.config.fail_on_zero_tables_if_tables_declared
                and declared_table_count > 0
                and len(tables) == 0
            ):
                errors.append("declared_tables_not_extracted")

        if len(tables) == 0:
            if len(rows) > 0:
                errors.append("table_rows_present_without_tables")
            return

        table_ids = set()
        tables_without_rows = 0

        for table in tables:
            table_id = table.get("table_id")
            if table_id is not None:
                table_ids.add(str(table_id))

        rows_by_table_id: dict[str, int] = {}
        rows_without_table_binding = 0

        for row in rows:
            row_table_id = row.get("table_id")
            if row_table_id is None:
                rows_without_table_binding += 1
                continue

            key = str(row_table_id)
            rows_by_table_id[key] = rows_by_table_id.get(key, 0) + 1

        for table in tables:
            table_id = table.get("table_id")
            key = str(table_id) if table_id is not None else None

            if key is None or rows_by_table_id.get(key, 0) == 0:
                tables_without_rows += 1

        tables_without_rows_ratio = tables_without_rows / len(tables) if tables else 0.0
        rows_without_binding_ratio = rows_without_table_binding / len(rows) if rows else 0.0

        metrics["tables_without_rows_count"] = tables_without_rows
        metrics["tables_without_rows_ratio"] = round(tables_without_rows_ratio, 4)
        metrics["rows_without_table_binding_count"] = rows_without_table_binding
        metrics["rows_without_table_binding_ratio"] = round(rows_without_binding_ratio, 4)

        if tables_without_rows_ratio > self.config.max_tables_without_rows_ratio:
            errors.append("tables_without_rows_ratio_too_high")

        if rows_without_binding_ratio > self.config.max_rows_without_table_binding_ratio:
            errors.append("rows_without_table_binding_ratio_too_high")

        # Additional light checks
        tables_without_title = 0
        tables_with_unknown_type = 0

        for table in tables:
            title = str(table.get("table_title") or "").strip()
            table_type = str(table.get("table_type") or "").strip().lower()

            if not title:
                tables_without_title += 1

            if table_type in {"", "other", "unknown"}:
                tables_with_unknown_type += 1

        metrics["tables_without_title_count"] = tables_without_title
        metrics["tables_with_unknown_type_count"] = tables_with_unknown_type

        if tables_without_title > 0:
            warnings.append("some_tables_without_title")

        if tables_with_unknown_type > 0:
            warnings.append("some_tables_with_unknown_type")

        # Check row payload usefulness
        empty_row_payload_count = 0
        for row in rows:
            if not self._row_has_meaningful_payload(row):
                empty_row_payload_count += 1

        metrics["empty_row_payload_count"] = empty_row_payload_count
        metrics["empty_row_payload_ratio"] = round(
            (empty_row_payload_count / len(rows)) if rows else 0.0,
            4,
        )

        if rows and empty_row_payload_count == len(rows):
            errors.append("all_table_rows_empty")

    # ---------------------------------------------------------
    # Semantic enrichment checks
    # ---------------------------------------------------------

    def _check_semantic_enrichment(
        self,
        payload: QcInput,
        metrics: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> None:
        enrichment = payload.enrichment_result
        legal_facts = enrichment.legal_facts or []
        aliases = enrichment.aliases or []
        measure_codes = enrichment.measure_codes or []

        metrics["aliases_count"] = len(aliases)
        metrics["measure_codes_count"] = len(measure_codes)

        if not enrichment.document_type:
            warnings.append("document_type_missing")

        if not enrichment.source_authority:
            warnings.append("source_authority_missing")

        if self.config.warn_on_zero_legal_facts and len(legal_facts) == 0:
            warnings.append("legal_facts_missing")

        if len(measure_codes) == 0:
            warnings.append("measure_codes_missing")

    # ---------------------------------------------------------
    # Retrievability / usability
    # ---------------------------------------------------------

    def _check_retrievability(
        self,
        payload: QcInput,
        metrics: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> None:
        if not self.config.require_some_retrievable_content:
            return

        blocks = payload.extraction_result.blocks or []
        tables = payload.extraction_result.tables or []
        rows = payload.extraction_result.table_rows or []
        facts = payload.enrichment_result.legal_facts or []

        meaningful_blocks = 0
        for block in blocks:
            if self._extract_block_text(block):
                meaningful_blocks += 1

        retrievable_units_count = meaningful_blocks + len(tables) + len(rows) + len(facts)
        metrics["retrievable_units_count"] = retrievable_units_count

        if retrievable_units_count == 0:
            errors.append("no_retrievable_units")

        # Weak but useful signal:
        # document may technically parse, but still be poor for search/answering.
        if retrievable_units_count < 3:
            warnings.append("retrievable_units_count_low")

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _extract_block_text(
        self,
        block: dict[str, Any],
    ) -> str:
        candidates = [
            block.get("content_clean"),
            block.get("content"),
            block.get("text"),
            block.get("content_raw"),
        ]

        for value in candidates:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text

        return ""

    def _row_has_meaningful_payload(
        self,
        row: dict[str, Any],
    ) -> bool:
        candidates = [
            row.get("row_json"),
            row.get("normalized_row_json"),
            row.get("row_summary"),
            row.get("content"),
            row.get("text"),
        ]

        for value in candidates:
            if value is None:
                continue

            if isinstance(value, dict) and len(value) > 0:
                return True

            if isinstance(value, list) and len(value) > 0:
                return True

            text = str(value).strip()
            if text and text not in {"{}", "[]"}:
                return True

        return False

    def _compose_error_code(
        self,
        errors: list[str],
    ) -> str:
        """
        Produce one deterministic top-level error code.

        Priority matters: the earliest truly blocking issue should
        become the main QC error code for the ingestion job.
        """
        priority = [
            "normalized_text_empty",
            "normalized_text_too_short",
            "document_identity_missing",
            "blocks_missing",
            "blocks_count_too_low",
            "meaningful_blocks_count_too_low",
            "empty_block_ratio_too_high",
            "declared_tables_not_extracted",
            "table_rows_present_without_tables",
            "tables_without_rows_ratio_too_high",
            "rows_without_table_binding_ratio_too_high",
            "all_table_rows_empty",
            "no_retrievable_units",
            "block_order_duplicates_detected",
        ]

        for code in priority:
            if code in errors:
                return code

        return errors[0] if errors else "qc_failed"