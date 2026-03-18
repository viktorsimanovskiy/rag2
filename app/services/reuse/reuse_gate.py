# ============================================================
# File: app/services/reuse/reuse_gate.py
# Purpose:
#   Safe controlled answer reuse for RAG system.
#
# Responsibilities:
#   - find similar historical answers
#   - filter by question signature
#   - validate freshness of evidence
#   - validate evidence consistency
#   - choose best reusable candidate
#   - return a deterministic reuse decision
#
# Design principles:
#   - conservative by default
#   - reuse only when evidence is still valid
#   - block reuse on ambiguity or stale documents
#   - prefer full regeneration over risky reuse
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import Select, and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.feedback import (
    AnswerEvent,
    AnswerEvidenceItem,
    AnswerReuseCandidate,
    QuestionEvent,
)
from app.db.models.enums import (
    AnswerModeEnum,
    QuestionIntentEnum,
    ReuseStatusEnum,
    ValidationStatusEnum,
)

logger = logging.getLogger(__name__)


# ============================================================
# IMPORTANT PLACEHOLDER IMPORT
# ============================================================
# This model should represent your active document registry.
# It is referenced here because reuse must verify current document
# freshness against the active knowledge base.
#
# Expected fields:
#   - document_id: UUID
#   - status: str or enum (must contain "active")
#   - file_hash: Optional[str]
#   - content_hash: Optional[str]
#
# Replace this import with your actual model when ready.
# ============================================================
try:
    from app.db.models.documents import DocumentRegistry  # pragma: no cover
except Exception:  # pragma: no cover
    DocumentRegistry = None  # type: ignore


# ============================================================
# Exceptions
# ============================================================

class ReuseGateError(Exception):
    """Base error for reuse gate."""


class ReuseValidationError(ReuseGateError):
    """Raised when reuse input is invalid."""


class ReuseDependencyError(ReuseGateError):
    """Raised when required DB dependencies or models are unavailable."""


# ============================================================
# Input / Output DTOs
# ============================================================

@dataclass(slots=True)
class ReuseQueryInput:
    """
    Input description of current question for reuse lookup.
    """
    question_event_id: UUID
    similarity_threshold: float = 0.90
    max_candidates: int = 20
    allow_measure_mismatch: bool = False
    allow_subject_category_mismatch: bool = False


@dataclass(slots=True)
class QuestionSignature:
    """
    Normalized routing signature of a question.
    """
    question_event_id: UUID
    intent_type: QuestionIntentEnum
    measure_code: Optional[str]
    subject_category_code: Optional[str]
    question_text_normalized: Optional[str]


@dataclass(slots=True)
class CandidateFreshnessResult:
    """
    Freshness validation result for one candidate.
    """
    answer_event_id: UUID
    is_fresh: bool
    checked_items_count: int
    mismatched_items_count: int
    block_reason_code: Optional[str]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateEvidenceValidationResult:
    """
    Evidence validation result for one candidate.
    """
    answer_event_id: UUID
    is_valid: bool
    checked_items_count: int
    has_evidence_hash: bool
    block_reason_code: Optional[str]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReuseCandidateMatch:
    """
    One candidate considered for reuse.
    """
    answer_event: AnswerEvent
    question_event: QuestionEvent
    reuse_candidate: AnswerReuseCandidate
    similarity_score: float
    signature_match_score: float
    freshness_result: Optional[CandidateFreshnessResult] = None
    evidence_result: Optional[CandidateEvidenceValidationResult] = None


@dataclass(slots=True)
class ReuseDecision:
    """
    Final output of reuse gate.
    """
    should_reuse: bool
    source_answer_event_id: Optional[UUID]
    decision_code: str
    confidence_score: float
    reason: str
    payload: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Service
# ============================================================

class ReuseGate:
    """
    Conservative reuse gate for previously validated answers.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    async def build_reuse_decision(
        self,
        payload: ReuseQueryInput,
    ) -> ReuseDecision:
        """
        Main orchestration method.

        Flow:
        1. Load current question and signature
        2. Find historical reusable candidates
        3. Filter by question signature
        4. Validate freshness
        5. Validate evidence
        6. Select best candidate
        7. Return deterministic reuse decision
        """
        self._validate_reuse_query_input(payload)

        current_question = await self._get_question_event_or_raise(payload.question_event_id)
        current_signature = self._build_question_signature(current_question)

        raw_candidates = await self.find_reuse_candidates(
            current_question=current_question,
            payload=payload,
        )

        if not raw_candidates:
            return ReuseDecision(
                should_reuse=False,
                source_answer_event_id=None,
                decision_code="no_candidates_found",
                confidence_score=0.0,
                reason="Не найдено ни одного безопасного кандидата для reuse.",
                payload={
                    "question_event_id": str(payload.question_event_id),
                    "stage": "find_reuse_candidates",
                },
            )

        signature_filtered = self.filter_candidates_by_question_signature(
            current_signature=current_signature,
            candidates=raw_candidates,
            allow_measure_mismatch=payload.allow_measure_mismatch,
            allow_subject_category_mismatch=payload.allow_subject_category_mismatch,
        )

        if not signature_filtered:
            return ReuseDecision(
                should_reuse=False,
                source_answer_event_id=None,
                decision_code="signature_mismatch",
                confidence_score=0.0,
                reason="Исторические ответы найдены, но их сигнатура не совпадает с текущим вопросом.",
                payload={
                    "question_event_id": str(payload.question_event_id),
                    "stage": "filter_candidates_by_question_signature",
                    "raw_candidates_count": len(raw_candidates),
                },
            )

        freshness_checked: list[ReuseCandidateMatch] = []
        for candidate in signature_filtered:
            candidate.freshness_result = await self.validate_candidate_freshness(candidate)
            if candidate.freshness_result.is_fresh:
                freshness_checked.append(candidate)

        if not freshness_checked:
            return ReuseDecision(
                should_reuse=False,
                source_answer_event_id=None,
                decision_code="all_candidates_stale",
                confidence_score=0.0,
                reason="Подходящие исторические ответы найдены, но их источники устарели или больше не совпадают с активной базой.",
                payload={
                    "question_event_id": str(payload.question_event_id),
                    "stage": "validate_candidate_freshness",
                    "signature_candidates_count": len(signature_filtered),
                },
            )

        evidence_checked: list[ReuseCandidateMatch] = []
        for candidate in freshness_checked:
            candidate.evidence_result = await self.validate_candidate_evidence(candidate)
            if candidate.evidence_result.is_valid:
                evidence_checked.append(candidate)

        if not evidence_checked:
            return ReuseDecision(
                should_reuse=False,
                source_answer_event_id=None,
                decision_code="evidence_mismatch",
                confidence_score=0.0,
                reason="Источники формально актуальны, но доказательная база ответа больше не совпадает с текущим состоянием системы.",
                payload={
                    "question_event_id": str(payload.question_event_id),
                    "stage": "validate_candidate_evidence",
                    "fresh_candidates_count": len(freshness_checked),
                },
            )

        best_candidate = self.select_best_candidate(evidence_checked)
        if best_candidate is None:
            return ReuseDecision(
                should_reuse=False,
                source_answer_event_id=None,
                decision_code="no_best_candidate",
                confidence_score=0.0,
                reason="Не удалось выбрать безопасный лучший кандидат для reuse.",
                payload={
                    "question_event_id": str(payload.question_event_id),
                    "stage": "select_best_candidate",
                    "validated_candidates_count": len(evidence_checked),
                },
            )

        decision_score = self._calculate_decision_confidence(best_candidate)

        return ReuseDecision(
            should_reuse=True,
            source_answer_event_id=best_candidate.answer_event.answer_event_id,
            decision_code="reuse_approved",
            confidence_score=decision_score,
            reason="Найден исторический ответ с высокой оценкой, совпадающей сигнатурой вопроса и подтверждённой актуальностью evidence.",
            payload={
                "question_event_id": str(payload.question_event_id),
                "source_answer_event_id": str(best_candidate.answer_event.answer_event_id),
                "similarity_score": best_candidate.similarity_score,
                "signature_match_score": best_candidate.signature_match_score,
                "reuse_score": float(best_candidate.reuse_candidate.reuse_score or 0),
                "avg_feedback_score": float(best_candidate.reuse_candidate.avg_feedback_score or 0),
                "feedback_count": best_candidate.reuse_candidate.feedback_count,
                "freshness_result": best_candidate.freshness_result.details if best_candidate.freshness_result else {},
                "evidence_result": best_candidate.evidence_result.details if best_candidate.evidence_result else {},
            },
        )

    async def find_reuse_candidates(
        self,
        *,
        current_question: QuestionEvent,
        payload: ReuseQueryInput,
    ) -> list[ReuseCandidateMatch]:
        """
        Find potentially reusable historical answers.

        Current conservative strategy:
        - only answers marked as reuse-eligible
        - only reuse candidates with effective allow flag
        - only passed validation
        - exclude reused answers as reuse source
        - exclude answers to the same current question_event
        - rank by similarity proxy + reuse quality

        Similarity:
        - exact normalized question text => 1.0
        - otherwise a conservative fallback heuristic is used

        Later this method should be extended to:
        - vector search over question embeddings
        - reranking by semantic similarity model
        """
        current_question_norm = self._normalize_text(current_question.question_text_normalized or current_question.question_text_raw)

        stmt: Select[Any] = (
            select(
                AnswerEvent,
                QuestionEvent,
                AnswerReuseCandidate,
            )
            .join(QuestionEvent, QuestionEvent.question_event_id == AnswerEvent.question_event_id)
            .join(
                AnswerReuseCandidate,
                AnswerReuseCandidate.source_answer_event_id == AnswerEvent.answer_event_id,
            )
            .where(AnswerEvent.question_event_id != current_question.question_event_id)
            .where(AnswerEvent.reuse_allowed.is_(True))
            .where(AnswerEvent.answer_mode != AnswerModeEnum.REUSED_ANSWER)
            .where(AnswerEvent.answer_mode != AnswerModeEnum.SAFE_NO_ANSWER)
            .where(AnswerEvent.validation_status == ValidationStatusEnum.PASSED)
            .where(AnswerEvent.deterministic_validation_passed.is_(True))
            .where(AnswerEvent.semantic_validation_passed.is_(True))
            .where(AnswerReuseCandidate.reuse_status == ReuseStatusEnum.ELIGIBLE)
            .where(AnswerReuseCandidate.reuse_allowed_effective.is_(True))
            .order_by(
                desc(AnswerReuseCandidate.reuse_score),
                desc(AnswerReuseCandidate.avg_feedback_score),
                desc(AnswerReuseCandidate.feedback_count),
                desc(AnswerEvent.created_at),
            )
            .limit(payload.max_candidates * 3)
        )

        result = await self.db.execute(stmt)
        rows = result.all()

        candidates: list[ReuseCandidateMatch] = []
        for answer_event, question_event, reuse_candidate in rows:
            historical_question_norm = self._normalize_text(
                question_event.question_text_normalized or question_event.question_text_raw
            )

            similarity_score = self._calculate_similarity_proxy(
                current_question_norm=current_question_norm,
                historical_question_norm=historical_question_norm,
            )

            if similarity_score < payload.similarity_threshold:
                continue

            signature_score = self._calculate_signature_match_score(
                current_question=current_question,
                historical_question=question_event,
            )

            candidates.append(
                ReuseCandidateMatch(
                    answer_event=answer_event,
                    question_event=question_event,
                    reuse_candidate=reuse_candidate,
                    similarity_score=similarity_score,
                    signature_match_score=signature_score,
                )
            )

        candidates.sort(
            key=lambda c: (
                c.similarity_score,
                c.signature_match_score,
                float(c.reuse_candidate.reuse_score or 0),
                float(c.reuse_candidate.avg_feedback_score or 0),
                c.reuse_candidate.feedback_count,
                c.answer_event.created_at,
            ),
            reverse=True,
        )

        return candidates[: payload.max_candidates]

    def filter_candidates_by_question_signature(
        self,
        *,
        current_signature: QuestionSignature,
        candidates: list[ReuseCandidateMatch],
        allow_measure_mismatch: bool = False,
        allow_subject_category_mismatch: bool = False,
    ) -> list[ReuseCandidateMatch]:
        """
        Filter candidates using deterministic signature rules.

        Must match:
        - intent_type (always)
        - measure_code (unless explicitly relaxed)
        - subject_category_code (unless explicitly relaxed)

        Conservative rule:
        if current question has a measure_code, candidate must match it.
        """
        filtered: list[ReuseCandidateMatch] = []

        for candidate in candidates:
            historical_signature = self._build_question_signature(candidate.question_event)

            if historical_signature.intent_type != current_signature.intent_type:
                continue

            if current_signature.measure_code:
                if not allow_measure_mismatch and historical_signature.measure_code != current_signature.measure_code:
                    continue

            if current_signature.subject_category_code:
                if (
                    not allow_subject_category_mismatch
                    and historical_signature.subject_category_code != current_signature.subject_category_code
                ):
                    continue

            candidate.signature_match_score = self._calculate_signature_pair_score(
                current=current_signature,
                historical=historical_signature,
            )
            filtered.append(candidate)

        filtered.sort(
            key=lambda c: (
                c.signature_match_score,
                c.similarity_score,
                float(c.reuse_candidate.reuse_score or 0),
            ),
            reverse=True,
        )
        return filtered

    async def validate_candidate_freshness(
        self,
        candidate: ReuseCandidateMatch,
    ) -> CandidateFreshnessResult:
        """
        Validate that all evidence documents are still active and unchanged.

        Checks:
        - evidence exists
        - each evidence item with document_id still points to active document
        - document_content_hash (preferred) still matches
        - fallback to document_file_hash if content hash unavailable

        If DocumentRegistry model is not available yet, this method raises
        ReuseDependencyError because safe reuse cannot be guaranteed.
        """
        if DocumentRegistry is None:
            raise ReuseDependencyError(
                "DocumentRegistry model is required for freshness validation."
            )

        evidence_items = await self._get_evidence_items(candidate.answer_event.answer_event_id)
        if not evidence_items:
            return CandidateFreshnessResult(
                answer_event_id=candidate.answer_event.answer_event_id,
                is_fresh=False,
                checked_items_count=0,
                mismatched_items_count=0,
                block_reason_code="no_evidence_items",
                details={"message": "Candidate has no evidence items."},
            )

        checked_items_count = 0
        mismatched_items_count = 0
        mismatch_details: list[dict[str, Any]] = []

        for item in evidence_items:
            if item.document_id is None:
                continue

            checked_items_count += 1

            stmt: Select[Any] = select(DocumentRegistry).where(
                DocumentRegistry.document_id == item.document_id
            )
            result = await self.db.execute(stmt)
            document = result.scalar_one_or_none()

            if document is None:
                mismatched_items_count += 1
                mismatch_details.append({
                    "document_id": str(item.document_id),
                    "reason": "document_not_found",
                })
                continue

            doc_status = getattr(document, "status", None)
            if str(doc_status).lower() != "active":
                mismatched_items_count += 1
                mismatch_details.append({
                    "document_id": str(item.document_id),
                    "reason": "document_not_active",
                    "actual_status": str(doc_status),
                })
                continue

            current_content_hash = getattr(document, "content_hash", None)
            current_file_hash = getattr(document, "file_hash", None)

            if item.document_content_hash:
                if current_content_hash != item.document_content_hash:
                    mismatched_items_count += 1
                    mismatch_details.append({
                        "document_id": str(item.document_id),
                        "reason": "document_content_hash_mismatch",
                        "expected": item.document_content_hash,
                        "actual": current_content_hash,
                    })
                    continue
            elif item.document_file_hash:
                if current_file_hash != item.document_file_hash:
                    mismatched_items_count += 1
                    mismatch_details.append({
                        "document_id": str(item.document_id),
                        "reason": "document_file_hash_mismatch",
                        "expected": item.document_file_hash,
                        "actual": current_file_hash,
                    })
                    continue
            else:
                mismatched_items_count += 1
                mismatch_details.append({
                    "document_id": str(item.document_id),
                    "reason": "missing_stored_hashes_in_evidence",
                })
                continue

        if checked_items_count == 0:
            return CandidateFreshnessResult(
                answer_event_id=candidate.answer_event.answer_event_id,
                is_fresh=False,
                checked_items_count=0,
                mismatched_items_count=0,
                block_reason_code="no_document_bound_evidence",
                details={
                    "message": "Candidate evidence contains no document-bound items.",
                },
            )

        is_fresh = mismatched_items_count == 0
        return CandidateFreshnessResult(
            answer_event_id=candidate.answer_event.answer_event_id,
            is_fresh=is_fresh,
            checked_items_count=checked_items_count,
            mismatched_items_count=mismatched_items_count,
            block_reason_code=None if is_fresh else "freshness_check_failed",
            details={
                "checked_items_count": checked_items_count,
                "mismatched_items_count": mismatched_items_count,
                "mismatches": mismatch_details,
            },
        )

    async def validate_candidate_evidence(
        self,
        candidate: ReuseCandidateMatch,
    ) -> CandidateEvidenceValidationResult:
        """
        Validate that candidate evidence package is still coherent.

        Checks:
        - answer has evidence_hash
        - answer has at least one evidence item
        - evidence items are structurally complete
        - every evidence item points to exactly one object
        - at least one strong evidence object exists:
          document, block, table, table_row, or legal_fact

        Note:
        This method does not re-run full retrieval.
        It validates stored evidence package integrity.
        """
        answer_event = candidate.answer_event
        evidence_items = await self._get_evidence_items(answer_event.answer_event_id)

        if not evidence_items:
            return CandidateEvidenceValidationResult(
                answer_event_id=answer_event.answer_event_id,
                is_valid=False,
                checked_items_count=0,
                has_evidence_hash=bool(answer_event.evidence_hash),
                block_reason_code="no_evidence_items",
                details={"message": "Candidate has no evidence items."},
            )

        invalid_items: list[dict[str, Any]] = []
        strong_item_count = 0

        for item in evidence_items:
            pointer_count = sum(
                1 for value in [
                    item.document_id,
                    item.block_id,
                    item.table_id,
                    item.table_row_id,
                    item.legal_fact_id,
                ] if value is not None
            )

            if pointer_count != 1:
                invalid_items.append({
                    "answer_evidence_item_id": str(item.answer_evidence_item_id),
                    "reason": "invalid_pointer_count",
                    "pointer_count": pointer_count,
                })
                continue

            if item.document_id or item.block_id or item.table_id or item.table_row_id or item.legal_fact_id:
                strong_item_count += 1

        if not answer_event.evidence_hash:
            return CandidateEvidenceValidationResult(
                answer_event_id=answer_event.answer_event_id,
                is_valid=False,
                checked_items_count=len(evidence_items),
                has_evidence_hash=False,
                block_reason_code="missing_evidence_hash",
                details={
                    "invalid_items": invalid_items,
                    "strong_item_count": strong_item_count,
                },
            )

        if invalid_items:
            return CandidateEvidenceValidationResult(
                answer_event_id=answer_event.answer_event_id,
                is_valid=False,
                checked_items_count=len(evidence_items),
                has_evidence_hash=True,
                block_reason_code="invalid_evidence_items",
                details={
                    "invalid_items": invalid_items,
                    "strong_item_count": strong_item_count,
                },
            )

        if strong_item_count == 0:
            return CandidateEvidenceValidationResult(
                answer_event_id=answer_event.answer_event_id,
                is_valid=False,
                checked_items_count=len(evidence_items),
                has_evidence_hash=True,
                block_reason_code="no_strong_evidence_items",
                details={
                    "invalid_items": invalid_items,
                    "strong_item_count": strong_item_count,
                },
            )

        return CandidateEvidenceValidationResult(
            answer_event_id=answer_event.answer_event_id,
            is_valid=True,
            checked_items_count=len(evidence_items),
            has_evidence_hash=True,
            block_reason_code=None,
            details={
                "strong_item_count": strong_item_count,
                "checked_items_count": len(evidence_items),
            },
        )

    def select_best_candidate(
        self,
        candidates: list[ReuseCandidateMatch],
    ) -> Optional[ReuseCandidateMatch]:
        """
        Select best candidate using conservative weighted order.

        Priority:
        1. similarity_score
        2. signature_match_score
        3. reuse_score
        4. avg_feedback_score
        5. feedback_count
        6. recency
        """
        if not candidates:
            return None

        def sort_key(candidate: ReuseCandidateMatch) -> tuple[float, float, float, float, int, datetime]:
            return (
                candidate.similarity_score,
                candidate.signature_match_score,
                float(candidate.reuse_candidate.reuse_score or 0),
                float(candidate.reuse_candidate.avg_feedback_score or 0),
                int(candidate.reuse_candidate.feedback_count or 0),
                candidate.answer_event.created_at,
            )

        return sorted(candidates, key=sort_key, reverse=True)[0]

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _validate_reuse_query_input(self, payload: ReuseQueryInput) -> None:
        if not (0.0 <= payload.similarity_threshold <= 1.0):
            raise ReuseValidationError("similarity_threshold must be between 0 and 1.")

        if payload.max_candidates < 1 or payload.max_candidates > 100:
            raise ReuseValidationError("max_candidates must be between 1 and 100.")

    def _build_question_signature(self, question_event: QuestionEvent) -> QuestionSignature:
        return QuestionSignature(
            question_event_id=question_event.question_event_id,
            intent_type=question_event.intent_type,
            measure_code=question_event.measure_code,
            subject_category_code=question_event.subject_category_code,
            question_text_normalized=self._normalize_text(
                question_event.question_text_normalized or question_event.question_text_raw
            ),
        )

    def _calculate_similarity_proxy(
        self,
        *,
        current_question_norm: str,
        historical_question_norm: str,
    ) -> float:
        """
        Conservative similarity proxy until vector search is connected.

        Rules:
        - exact match => 1.0
        - strict containment => 0.97
        - high token overlap => 0.90..0.96
        - else => 0.0..0.89
        """
        if not current_question_norm or not historical_question_norm:
            return 0.0

        if current_question_norm == historical_question_norm:
            return 1.0

        if (
            current_question_norm in historical_question_norm
            or historical_question_norm in current_question_norm
        ):
            return 0.97

        current_tokens = set(current_question_norm.split())
        historical_tokens = set(historical_question_norm.split())

        if not current_tokens or not historical_tokens:
            return 0.0

        intersection = current_tokens.intersection(historical_tokens)
        union = current_tokens.union(historical_tokens)
        overlap = len(intersection) / len(union)

        if overlap >= 0.80:
            return 0.95
        if overlap >= 0.70:
            return 0.92
        if overlap >= 0.60:
            return 0.90

        return round(overlap, 4)

    def _calculate_signature_match_score(
        self,
        *,
        current_question: QuestionEvent,
        historical_question: QuestionEvent,
    ) -> float:
        current_signature = self._build_question_signature(current_question)
        historical_signature = self._build_question_signature(historical_question)
        return self._calculate_signature_pair_score(
            current=current_signature,
            historical=historical_signature,
        )

    def _calculate_signature_pair_score(
        self,
        *,
        current: QuestionSignature,
        historical: QuestionSignature,
    ) -> float:
        score = 0.0

        if current.intent_type == historical.intent_type:
            score += 0.50

        if current.measure_code and historical.measure_code and current.measure_code == historical.measure_code:
            score += 0.30
        elif current.measure_code is None and historical.measure_code is None:
            score += 0.10

        if (
            current.subject_category_code
            and historical.subject_category_code
            and current.subject_category_code == historical.subject_category_code
        ):
            score += 0.20
        elif current.subject_category_code is None and historical.subject_category_code is None:
            score += 0.05

        return round(min(score, 1.0), 4)

    def _calculate_decision_confidence(
        self,
        candidate: ReuseCandidateMatch,
    ) -> float:
        reuse_score = float(candidate.reuse_candidate.reuse_score or 0)
        avg_feedback_score = float(candidate.reuse_candidate.avg_feedback_score or 0)
        feedback_score_normalized = min(max((avg_feedback_score - 1.0) / 4.0, 0.0), 1.0)

        freshness_component = 1.0 if candidate.freshness_result and candidate.freshness_result.is_fresh else 0.0
        evidence_component = 1.0 if candidate.evidence_result and candidate.evidence_result.is_valid else 0.0

        value = (
            0.35 * candidate.similarity_score
            + 0.20 * candidate.signature_match_score
            + 0.20 * reuse_score
            + 0.10 * feedback_score_normalized
            + 0.10 * freshness_component
            + 0.05 * evidence_component
        )
        return round(min(max(value, 0.0), 1.0), 4)

    async def _get_question_event_or_raise(self, question_event_id: UUID) -> QuestionEvent:
        stmt: Select[Any] = select(QuestionEvent).where(
            QuestionEvent.question_event_id == question_event_id
        )
        result = await self.db.execute(stmt)
        obj = result.scalar_one_or_none()
        if obj is None:
            raise ReuseValidationError(f"QuestionEvent not found: {question_event_id}")
        return obj

    async def _get_evidence_items(self, answer_event_id: UUID) -> list[AnswerEvidenceItem]:
        stmt: Select[Any] = (
            select(AnswerEvidenceItem)
            .where(AnswerEvidenceItem.answer_event_id == answer_event_id)
            .order_by(AnswerEvidenceItem.evidence_order.asc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    def _normalize_text(self, value: Optional[str]) -> str:
        if not value:
            return ""
        return " ".join(value.strip().lower().split())

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)