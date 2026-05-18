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
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from app.db.models.enums import QuestionIntentEnum
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
    ) -> None:
        self.retrieval_orchestrator = retrieval_orchestrator
        self.generation_pipeline = generation_pipeline
        self.service_resolver = service_resolver

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    async def build_answer(
        self,
        payload: RuntimeAnswerInput,
    ) -> RuntimeAnswerResult:
        self._validate_input(payload)

        resolved_query_terms = self._deduplicate_terms(list(payload.query_terms or []))

        service_resolution = await self._resolve_service_context(payload)

        retrieval_input = self._build_retrieval_input(
            payload,
            resolved_query_terms=resolved_query_terms,
            service_resolution=service_resolution,
        )

        evidence_package = await self.retrieval_orchestrator.retrieve(
            retrieval_input
        )

        generation_request = self._build_generation_request(payload)

        generation_result = await self.generation_pipeline.generate_answer(
            payload=generation_request,
            evidence_package=evidence_package,
        )

        enriched_generation_result = self._enrich_generation_result(
            generation_result=generation_result,
            evidence_package=evidence_package,
            payload=payload,
        )

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
        answer_payload_json["runtime_answer_service"] = {
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