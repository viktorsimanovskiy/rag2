# ============================================================
# File: app/services/answers/runtime_answer_service.py
# Purpose:
#   Runtime service that executes the core answer path:
#       question -> retrieval -> generation -> result
#
# Responsibilities:
#   - prepare retrieval input
#   - invoke retrieval orchestrator
#   - invoke generation pipeline
#   - apply safe runtime fallbacks
#   - return generation result with retrieval debug payload
# ============================================================

from __future__ import annotations

import logging
from time import perf_counter
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from app.db.models.enums import (
    AnswerModeEnum,
    EvidenceItemTypeEnum,
    QuestionIntentEnum,
    ValidationStatusEnum,
)
from app.services.generation.generation_pipeline import (
    GenerationPipeline,
    GenerationRequest,
    GenerationResult,
)
from app.services.retrieval.retrieval_orchestrator import (
    EvidencePackage,
    RetrievalInput,
    RetrievalOrchestrator,
)
from app.services.retrieval.service_resolver import (
    ServiceResolutionResult,
    ServiceResolver,
    ServiceResolverInput,
)
from app.services.feedback.feedback_service import EvidenceItemInput
from app.services.retrieval.service_discovery import (
    ServiceDiscovery,
    ServiceDiscoveryInput,
    ServiceDiscoveryResult,
)

logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================

class RuntimeAnswerServiceError(Exception):
    """Base runtime answer service error."""


class RuntimeAnswerValidationError(RuntimeAnswerServiceError):
    """Raised when runtime input is invalid."""


# ============================================================
# DTOs
# ============================================================

@dataclass(slots=True)
class RuntimeAnswerInput:
    """
    Input from upper application layer into runtime answer service.
    """
    session_id: UUID
    question_event_id: UUID
    channel_code: Any

    question_text_raw: str
    question_text_normalized: str
    language_code: str

    intent_type: QuestionIntentEnum
    subject_category_code: Optional[str] = None

    routing_payload_json: dict[str, Any] = field(default_factory=dict)
    query_constraints_json: dict[str, Any] = field(default_factory=dict)
    request_metadata_json: dict[str, Any] = field(default_factory=dict)

    query_terms: list[str] = field(default_factory=list)

    top_k_facts: int = 10
    top_k_tables: int = 10
    top_k_rows: int = 12
    top_k_blocks: int = 12
    final_top_k: int = 12


@dataclass(slots=True)
class RuntimeAnswerResult:
    """
    Final result of runtime answer path.
    """
    generation_result: GenerationResult
    evidence_package: EvidencePackage
    runtime_payload_json: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Service
# ============================================================

class RuntimeAnswerService:
    """
    Main runtime link between retrieval and generation.
    """

    def __init__(
        self,
        *,
        retrieval_orchestrator: RetrievalOrchestrator,
        generation_pipeline: GenerationPipeline,
        service_resolver: Optional[ServiceResolver] = None,
        service_discovery: Optional[ServiceDiscovery] = None,
    ) -> None:
        self.retrieval_orchestrator = retrieval_orchestrator
        self.generation_pipeline = generation_pipeline
        self.service_resolver = service_resolver
        self.service_discovery = service_discovery

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    async def build_answer(
        self,
        payload: RuntimeAnswerInput,
    ) -> RuntimeAnswerResult:
        total_started_at = perf_counter()

        validation_started_at = perf_counter()
        self._validate_input(payload)
        validation_elapsed = perf_counter() - validation_started_at

        terms_started_at = perf_counter()
        resolved_query_terms = self._deduplicate_terms(list(payload.query_terms or []))
        terms_elapsed = perf_counter() - terms_started_at

        if self._requires_service_discovery(payload):
            return await self._build_service_discovery_answer(
                payload=payload,
                total_started_at=total_started_at,
                validation_elapsed=validation_elapsed,
                terms_elapsed=terms_elapsed,
            )

        service_resolution_started_at = perf_counter()
        service_resolution = await self._resolve_service_context(payload)
        service_resolution_elapsed = perf_counter() - service_resolution_started_at

        retrieval_input_started_at = perf_counter()
        retrieval_input = self._build_retrieval_input(
            payload,
            resolved_query_terms=resolved_query_terms,
            service_resolution=service_resolution,
        )
        retrieval_input_elapsed = perf_counter() - retrieval_input_started_at

        retrieval_started_at = perf_counter()
        evidence_package = await self.retrieval_orchestrator.retrieve(
            retrieval_input
        )
        retrieval_elapsed = perf_counter() - retrieval_started_at

        generation_request_started_at = perf_counter()
        generation_request = self._build_generation_request(payload)
        generation_request_elapsed = perf_counter() - generation_request_started_at

        generation_started_at = perf_counter()
        generation_result = await self.generation_pipeline.generate_answer(
            payload=generation_request,
            evidence_package=evidence_package,
        )
        generation_elapsed = perf_counter() - generation_started_at

        enrich_started_at = perf_counter()
        enriched_generation_result = self._enrich_generation_result(
            generation_result=generation_result,
            evidence_package=evidence_package,
            payload=payload,
        )
        enrich_elapsed = perf_counter() - enrich_started_at

        total_elapsed = perf_counter() - total_started_at
        timings_json = {
            "validation_sec": round(validation_elapsed, 6),
            "query_terms_sec": round(terms_elapsed, 6),
            "service_resolution_sec": round(service_resolution_elapsed, 6),
            "build_retrieval_input_sec": round(retrieval_input_elapsed, 6),
            "retrieval_sec": round(retrieval_elapsed, 6),
            "build_generation_request_sec": round(generation_request_elapsed, 6),
            "generation_sec": round(generation_elapsed, 6),
            "enrich_generation_result_sec": round(enrich_elapsed, 6),
            "total_sec": round(total_elapsed, 6),
        }

        runtime_payload_json = {
            "question_event_id": str(payload.question_event_id),
            "strategy_code": evidence_package.strategy_code,
            "selected_candidates_count": len(evidence_package.selected_candidates),
            "selected_document_ids_count": len(evidence_package.selected_document_ids),
            "selected_fact_ids_count": len(evidence_package.selected_fact_ids),
            "selected_table_ids_count": len(evidence_package.selected_table_ids),
            "selected_row_ids_count": len(evidence_package.selected_row_ids),
            "selected_block_ids_count": len(evidence_package.selected_block_ids),
            "service_resolution": self._service_resolution_to_json(service_resolution),
            "timings_sec": timings_json,
        }

        logger.info(
            "Runtime answer built",
            extra={
                "question_event_id": str(payload.question_event_id),
                "intent_type": str(payload.intent_type),
                "strategy_code": evidence_package.strategy_code,
                "answer_mode": str(enriched_generation_result.answer_mode),
                "confidence_score": enriched_generation_result.confidence_score,
            },
        )

        return RuntimeAnswerResult(
            generation_result=enriched_generation_result,
            evidence_package=evidence_package,
            runtime_payload_json=runtime_payload_json,
        )


    # --------------------------------------------------------
    # Service discovery
    # --------------------------------------------------------

    def _requires_service_discovery(self, payload: RuntimeAnswerInput) -> bool:
        """
        Return True for broad entitlement questions that must not be forced
        into one randomly selected service.
        """
        sources = (
            payload.query_constraints_json or {},
            payload.routing_payload_json or {},
            payload.request_metadata_json or {},
        )
        for source in sources:
            if bool(source.get("requires_service_discovery")):
                return True
            if str(source.get("routing_mode") or "").strip().lower() == "service_discovery":
                return True
        return False

    async def _build_service_discovery_answer(
        self,
        *,
        payload: RuntimeAnswerInput,
        total_started_at: float,
        validation_elapsed: float,
        terms_elapsed: float,
    ) -> RuntimeAnswerResult:
        discovery_started_at = perf_counter()

        if self.service_discovery is None:
            evidence_package = EvidencePackage(
                question_event_id=payload.question_event_id,
                strategy_code="service_discovery_unavailable",
                metrics_json={
                    "evidence_quality": "insufficient",
                    "guard_reason": "service_discovery_not_configured",
                },
                debug_payload_json={
                    "service_resolution": {
                        "resolution_status": "service_discovery_unavailable",
                        "service_key": None,
                    },
                    "evidence_quality": "insufficient",
                    "guard_reason": "service_discovery_not_configured",
                },
            )
            generation_result = self._build_service_discovery_unavailable_result(payload)
            enriched_generation_result = self._enrich_generation_result(
                generation_result=generation_result,
                evidence_package=evidence_package,
                payload=payload,
            )
            return RuntimeAnswerResult(
                generation_result=enriched_generation_result,
                evidence_package=evidence_package,
                runtime_payload_json={
                    "question_event_id": str(payload.question_event_id),
                    "strategy_code": evidence_package.strategy_code,
                    "selected_candidates_count": 0,
                    "selected_document_ids_count": 0,
                    "selected_fact_ids_count": 0,
                    "selected_table_ids_count": 0,
                    "selected_row_ids_count": 0,
                    "selected_block_ids_count": 0,
                    "service_resolution": evidence_package.debug_payload_json.get("service_resolution"),
                    "timings_sec": {
                        "validation_sec": round(validation_elapsed, 6),
                        "query_terms_sec": round(terms_elapsed, 6),
                        "service_discovery_sec": 0.0,
                        "total_sec": round(perf_counter() - total_started_at, 6),
                    },
                },
            )

        discovery_result = await self.service_discovery.discover(
            ServiceDiscoveryInput(
                question_text_raw=payload.question_text_raw,
                question_text_normalized=payload.question_text_normalized,
            )
        )
        discovery_elapsed = perf_counter() - discovery_started_at

        evidence_package = self._build_service_discovery_evidence_package(
            payload=payload,
            discovery_result=discovery_result,
        )
        generation_result = self._build_service_discovery_generation_result(
            discovery_result=discovery_result,
        )
        enriched_generation_result = self._enrich_generation_result(
            generation_result=generation_result,
            evidence_package=evidence_package,
            payload=payload,
        )

        total_elapsed = perf_counter() - total_started_at
        timings_json = {
            "validation_sec": round(validation_elapsed, 6),
            "query_terms_sec": round(terms_elapsed, 6),
            "service_discovery_sec": round(discovery_elapsed, 6),
            "total_sec": round(total_elapsed, 6),
        }

        runtime_payload_json = {
            "question_event_id": str(payload.question_event_id),
            "strategy_code": evidence_package.strategy_code,
            "selected_candidates_count": len(evidence_package.selected_candidates),
            "selected_document_ids_count": len(evidence_package.selected_document_ids),
            "selected_fact_ids_count": len(evidence_package.selected_fact_ids),
            "selected_table_ids_count": len(evidence_package.selected_table_ids),
            "selected_row_ids_count": len(evidence_package.selected_row_ids),
            "selected_block_ids_count": len(evidence_package.selected_block_ids),
            "service_resolution": evidence_package.debug_payload_json.get("service_resolution"),
            "timings_sec": timings_json,
        }

        logger.info(
            "Service discovery answer built",
            extra={
                "question_event_id": str(payload.question_event_id),
                "answer_mode": str(enriched_generation_result.answer_mode),
                "selected_services_count": len(discovery_result.candidates),
            },
        )

        return RuntimeAnswerResult(
            generation_result=enriched_generation_result,
            evidence_package=evidence_package,
            runtime_payload_json=runtime_payload_json,
        )

    def _build_service_discovery_evidence_package(
        self,
        *,
        payload: RuntimeAnswerInput,
        discovery_result: ServiceDiscoveryResult,
    ) -> EvidencePackage:
        from app.services.retrieval.retrieval_orchestrator import RetrievedCandidate

        candidates: list[RetrievedCandidate] = []
        for service_candidate in discovery_result.candidates:
            for row in service_candidate.matched_rows[:2]:
                candidates.append(
                    RetrievedCandidate(
                        source_type="table_row",
                        source_id=row.row_id,
                        document_id=row.document_id,
                        score=row.score,
                        rerank_score=None,
                        document_name=row.document_name,
                        doc_uid_base=None,
                        revision_date=None,
                        subject_category=None,
                        title=row.table_title,
                        snippet=row.applicant_category_name or row.row_summary,
                        citation_json=row.citation_json,
                        metadata_json={
                            "service_key": row.service_key,
                            "service_name_short": row.service_name_short,
                            "service_discovery_score": row.score,
                            "matched_signal_codes": row.matched_signal_codes,
                            "matched_terms": row.matched_terms,
                        },
                    )
                )

        row_ids = [candidate.source_id for candidate in candidates]
        document_ids = self._unique_uuid(candidate.document_id for candidate in candidates)
        table_ids = self._unique_uuid(
            row.table_id
            for service_candidate in discovery_result.candidates
            for row in service_candidate.matched_rows[:2]
        )

        service_resolution = {
            "resolution_status": "service_discovery",
            "service_key": None,
            "service_name_short": None,
            "service_name_full": None,
            "candidates": [
                {
                    "service_key": candidate.service_key,
                    "service_name_short": candidate.service_name_short,
                    "score": candidate.score,
                    "matched_signal_codes": candidate.matched_signal_codes,
                    "matched_signal_labels": candidate.matched_signal_labels,
                }
                for candidate in discovery_result.candidates[:7]
            ],
            "debug_payload_json": discovery_result.debug_payload_json,
        }

        evidence_quality = "strong" if discovery_result.can_answer else "insufficient"
        guard_reason = None if discovery_result.can_answer else "service_discovery_no_candidates"

        return EvidencePackage(
            question_event_id=payload.question_event_id,
            strategy_code="service_discovery",
            selected_candidates=candidates,
            selected_fact_ids=[],
            selected_table_ids=table_ids,
            selected_row_ids=row_ids,
            selected_block_ids=[],
            selected_document_ids=document_ids,
            metrics_json={
                "service_filter_applied": False,
                "service_filter_key": None,
                "selected_services_count": len(discovery_result.candidates),
                "selected_document_ids_count": len(document_ids),
                "selected_table_ids_count": len(table_ids),
                "selected_row_ids_count": len(row_ids),
                "selected_fact_ids_count": 0,
                "selected_block_ids_count": 0,
                "final_candidates_count": len(candidates),
                "evidence_quality": evidence_quality,
                "guard_reason": guard_reason,
            },
            debug_payload_json={
                "strategy_code": "service_discovery",
                "service_resolution": service_resolution,
                "service_filter_applied": False,
                "service_filter_key": None,
                "evidence_quality": evidence_quality,
                "guard_reason": guard_reason,
                "service_discovery": discovery_result.debug_payload_json,
            },
        )

    def _build_service_discovery_generation_result(
        self,
        *,
        discovery_result: ServiceDiscoveryResult,
    ) -> GenerationResult:
        if discovery_result.can_answer:
            answer_mode = AnswerModeEnum.GROUNDED_NARRATIVE
            confidence_score = 0.72
            trust_score = 0.70
            reason_code = None
        else:
            answer_mode = AnswerModeEnum.SAFE_NO_ANSWER
            confidence_score = 0.40
            trust_score = 0.35
            reason_code = "service_discovery_no_candidates"

        evidence_items: list[EvidenceItemInput] = []
        seen_rows: set[Any] = set()
        for service_candidate in discovery_result.candidates:
            for row in service_candidate.matched_rows[:2]:
                if row.row_id in seen_rows:
                    continue
                seen_rows.add(row.row_id)
                evidence_items.append(
                    EvidenceItemInput(
                        evidence_item_type=EvidenceItemTypeEnum.TABLE_ROW,
                        role_code="primary_evidence" if len(evidence_items) < 5 else "supporting_evidence",
                        citation_json=row.citation_json,
                        document_id=row.document_id,
                        table_row_id=row.row_id,
                    )
                )

        return GenerationResult(
            answer_mode=answer_mode,
            answer_text=discovery_result.answer_text,
            answer_text_short=discovery_result.answer_text_short,
            confidence_score=confidence_score,
            trust_score_at_generation=trust_score,
            validation_status=ValidationStatusEnum.PASSED,
            deterministic_validation_passed=True,
            semantic_validation_passed=True,
            reuse_allowed=False,
            reuse_policy_version="reuse_gate_v1",
            citations_json=discovery_result.citations_json,
            answer_payload_json={
                "strategy_code": "service_discovery",
                "reason_code": reason_code,
                "service_discovery": discovery_result.debug_payload_json,
                "warnings": discovery_result.warnings,
            },
            reuse_decision_payload_json={
                "reuse_allowed": False,
                "reason_code": "service_discovery_answer",
            },
            evidence_items=evidence_items,
            generation_model_name=None,
            generation_prompt_version="service_discovery_template_v1",
            pipeline_version="runtime_service_discovery_v1",
        )

    @staticmethod
    def _build_service_discovery_unavailable_result(payload: RuntimeAnswerInput) -> GenerationResult:
        answer_text = (
            "Для такого вопроса нужен режим подбора возможных мер, но он сейчас не подключён. "
            "Я не буду выбирать одну случайную услугу, потому что это может привести к неверному ответу."
        )
        return GenerationResult(
            answer_mode=AnswerModeEnum.SAFE_NO_ANSWER,
            answer_text=answer_text,
            answer_text_short=answer_text,
            confidence_score=0.30,
            trust_score_at_generation=0.30,
            validation_status=ValidationStatusEnum.PASSED,
            deterministic_validation_passed=True,
            semantic_validation_passed=True,
            reuse_allowed=False,
            reuse_policy_version="reuse_gate_v1",
            citations_json=[],
            answer_payload_json={
                "strategy_code": "service_discovery_unavailable",
                "reason_code": "service_discovery_not_configured",
                "question_event_id": str(payload.question_event_id),
            },
            reuse_decision_payload_json={
                "reuse_allowed": False,
                "reason_code": "service_discovery_not_configured",
            },
            evidence_items=[],
            generation_model_name=None,
            generation_prompt_version="service_discovery_template_v1",
            pipeline_version="runtime_service_discovery_v1",
        )

    @staticmethod
    def _unique_uuid(values: Any) -> list[Any]:
        result: list[Any] = []
        seen: set[Any] = set()
        for value in values:
            if value is None or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    # --------------------------------------------------------
    # Service resolution
    # --------------------------------------------------------

    async def _resolve_service_context(
        self,
        payload: RuntimeAnswerInput,
    ) -> Optional[ServiceResolutionResult]:
        """
        Try to resolve the user's question to a concrete service.

        The result is passed to retrieval as metadata. Retrieval may apply a
        strict service_key filter only when the resolver status is "resolved".
        Ambiguous and not_found results are intentionally not hard-filtered.
        """
        if self.service_resolver is None:
            return None

        question_text = payload.question_text_normalized or payload.question_text_raw
        result = await self.service_resolver.resolve(
            ServiceResolverInput(question_text=question_text)
        )

        logger.info(
            "Service resolution completed",
            extra={
                "question_event_id": str(payload.question_event_id),
                "resolution_status": result.resolution_status,
                "service_key": result.service_key,
                "candidates_count": len(result.candidates),
            },
        )

        return result

    def _service_resolution_to_json(
        self,
        service_resolution: Optional[ServiceResolutionResult],
    ) -> dict[str, Any]:
        if service_resolution is None:
            return {}

        selected = service_resolution.selected_service

        return {
            "resolution_status": service_resolution.resolution_status,
            "service_key": selected.service_key if selected is not None else None,
            "service_name_short": selected.service_name_short if selected is not None else None,
            "service_name_full": selected.service_name_full if selected is not None else None,
            "frgu_1": selected.frgu_1 if selected is not None else None,
            "frgu_3": selected.frgu_3 if selected is not None else None,
            "score": selected.score if selected is not None else None,
            "confidence": selected.confidence if selected is not None else None,
            "candidates": [
                {
                    "service_key": candidate.service_key,
                    "service_name_short": candidate.service_name_short,
                    "score": candidate.score,
                    "confidence": candidate.confidence,
                    "matched_terms": list(candidate.matched_terms),
                    "matched_aliases": list(candidate.matched_aliases),
                }
                for candidate in service_resolution.candidates[:5]
            ],
            "debug_payload_json": service_resolution.debug_payload_json,
        }

    # --------------------------------------------------------
    # Builders
    # --------------------------------------------------------

    def _build_retrieval_input(
        self,
        payload: RuntimeAnswerInput,
        *,
        resolved_query_terms: list[str],
        service_resolution: Optional[ServiceResolutionResult],
    ) -> RetrievalInput:
        constraints_json = dict(payload.query_constraints_json or {})
        service_resolution_json = self._service_resolution_to_json(service_resolution)
        if service_resolution_json:
            constraints_json["service_resolution"] = service_resolution_json

        return RetrievalInput(
            question_event_id=payload.question_event_id,
            question_text_raw=payload.question_text_raw,
            question_text_normalized=payload.question_text_normalized,
            intent_type=payload.intent_type,
            subject_category_code=payload.subject_category_code,
            query_terms=resolved_query_terms,
            constraints_json=constraints_json,
            top_k_facts=payload.top_k_facts,
            top_k_tables=payload.top_k_tables,
            top_k_rows=payload.top_k_rows,
            top_k_blocks=payload.top_k_blocks,
            final_top_k=payload.final_top_k,
        )

    def _build_generation_request(
        self,
        payload: RuntimeAnswerInput,
    ) -> GenerationRequest:
        routing_payload_json = dict(payload.routing_payload_json or {})
        query_constraints_json = dict(payload.query_constraints_json or {})
        request_metadata_json = dict(payload.request_metadata_json or {})

        return GenerationRequest(
            session_id=payload.session_id,
            question_event_id=payload.question_event_id,
            channel_code=payload.channel_code,
            question_text_raw=payload.question_text_raw,
            question_text_normalized=payload.question_text_normalized,
            language_code=payload.language_code,
            intent_type=payload.intent_type,
            subject_category_code=payload.subject_category_code,
            routing_payload_json=routing_payload_json,
            query_constraints_json=query_constraints_json,
            request_metadata_json=request_metadata_json,
        )

    def _enrich_generation_result(
        self,
        *,
        generation_result: GenerationResult,
        evidence_package: EvidencePackage,
        payload: RuntimeAnswerInput,
    ) -> GenerationResult:
        answer_payload_json = dict(generation_result.answer_payload_json or {})
        existing_runtime_debug = dict(answer_payload_json.get("runtime_answer_service") or {})
        answer_payload_json["runtime_answer_service"] = {
            **existing_runtime_debug,
            "question_event_id": str(payload.question_event_id),
            "strategy_code": evidence_package.strategy_code,
            "evidence_metrics": evidence_package.metrics_json,
            "selected_document_ids": [str(x) for x in evidence_package.selected_document_ids],
            "selected_fact_ids": [str(x) for x in evidence_package.selected_fact_ids],
            "selected_table_ids": [str(x) for x in evidence_package.selected_table_ids],
            "selected_row_ids": [str(x) for x in evidence_package.selected_row_ids],
            "selected_block_ids": [str(x) for x in evidence_package.selected_block_ids],
            "debug_payload_json": evidence_package.debug_payload_json,
        }

        return GenerationResult(
            answer_mode=generation_result.answer_mode,
            answer_text=generation_result.answer_text,
            answer_text_short=generation_result.answer_text_short,
            confidence_score=generation_result.confidence_score,
            trust_score_at_generation=generation_result.trust_score_at_generation,
            validation_status=generation_result.validation_status,
            deterministic_validation_passed=generation_result.deterministic_validation_passed,
            semantic_validation_passed=generation_result.semantic_validation_passed,
            reuse_allowed=generation_result.reuse_allowed,
            reuse_policy_version=generation_result.reuse_policy_version,
            citations_json=generation_result.citations_json,
            answer_payload_json=answer_payload_json,
            reuse_decision_payload_json=generation_result.reuse_decision_payload_json,
            evidence_items=generation_result.evidence_items,
            generation_model_name=generation_result.generation_model_name,
            generation_prompt_version=generation_result.generation_prompt_version,
            pipeline_version=generation_result.pipeline_version,
        )


    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    @staticmethod
    def _deduplicate_terms(terms: list[str]) -> list[str]:
        """Deduplicate query terms while preserving their original order."""
        result: list[str] = []
        seen: set[str] = set()

        for term in terms:
            cleaned = " ".join(str(term).split()).strip()
            if not cleaned:
                continue
            key = cleaned.lower().replace("ё", "е")
            if key in seen:
                continue
            seen.add(key)
            result.append(cleaned)

        return result

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def _validate_input(
        self,
        payload: RuntimeAnswerInput,
    ) -> None:
        if not payload.question_text_raw or not payload.question_text_raw.strip():
            raise RuntimeAnswerValidationError("question_text_raw must not be empty.")

        if not payload.question_text_normalized or not payload.question_text_normalized.strip():
            raise RuntimeAnswerValidationError("question_text_normalized must not be empty.")

        if not payload.language_code or not payload.language_code.strip():
            raise RuntimeAnswerValidationError("language_code must not be empty.")

        if payload.final_top_k < 1:
            raise RuntimeAnswerValidationError("final_top_k must be >= 1.")