# ============================================================
# File: app/services/feedback/feedback_service.py
# Purpose:
#   Service layer for:
#   - answer event creation
#   - evidence persistence
#   - user feedback recording
#   - reuse candidate recomputation
#   - daily quality aggregate generation
#
# Notes:
#   This file is designed for production-oriented architecture:
#   - async SQLAlchemy 2.x
#   - transactional writes
#   - deterministic hashing
#   - safe validation
#   - future reuse gate compatibility
# ============================================================

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Iterable, Optional
from uuid import UUID, uuid4

from sqlalchemy import Select, case, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.feedback import (
    AnswerEvent,
    AnswerEvidenceItem,
    AnswerFeedback,
    AnswerReuseCandidate,
    Channel,
    ConversationSession,
    QualityAggregateDaily,
    QuestionEvent,
)
from app.db.models.enums import (
    AnswerModeEnum,
    ChannelTypeEnum,
    EvidenceItemTypeEnum,
    FeedbackReasonCodeEnum,
    QuestionIntentEnum,
    ReuseStatusEnum,
    ValidationStatusEnum,
)

logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================

class FeedbackServiceError(Exception):
    """Base error for feedback service."""


class ValidationError(FeedbackServiceError):
    """Raised when input validation fails."""


class NotFoundError(FeedbackServiceError):
    """Raised when an expected DB object does not exist."""


class ConflictError(FeedbackServiceError):
    """Raised when uniqueness or state conflicts happen."""


# ============================================================
# Input DTOs
# ============================================================

@dataclass(slots=True)
class EvidenceItemInput:
    """
    Represents one evidence object used to produce an answer.

    Exactly one of:
    - document_id
    - block_id
    - table_id
    - table_row_id
    - legal_fact_id
    must be present.
    """
    evidence_item_type: EvidenceItemTypeEnum
    role_code: str
    citation_json: dict[str, Any] = field(default_factory=dict)

    document_id: Optional[UUID] = None
    block_id: Optional[UUID] = None
    table_id: Optional[UUID] = None
    table_row_id: Optional[UUID] = None
    legal_fact_id: Optional[UUID] = None

    document_file_hash: Optional[str] = None
    document_content_hash: Optional[str] = None


@dataclass(slots=True)
class AnswerEventCreateInput:
    """
    Input for answer event creation.
    """
    question_event_id: UUID
    answer_mode: AnswerModeEnum
    answer_text: str
    answer_text_short: Optional[str] = None
    answer_language_code: str = "ru"

    confidence_score: Optional[float] = None
    trust_score_at_generation: Optional[float] = None

    validation_status: ValidationStatusEnum = ValidationStatusEnum.NOT_RUN
    deterministic_validation_passed: bool = False
    semantic_validation_passed: bool = False

    reuse_allowed: bool = False
    reused_from_answer_event_id: Optional[UUID] = None
    reuse_policy_version: Optional[str] = None
    reuse_decision_payload_json: dict[str, Any] = field(default_factory=dict)

    citations_json: list[dict[str, Any]] = field(default_factory=list)
    answer_payload_json: dict[str, Any] = field(default_factory=dict)

    generation_model_name: Optional[str] = None
    generation_prompt_version: Optional[str] = None
    pipeline_version: Optional[str] = None

    evidence_items: list[EvidenceItemInput] = field(default_factory=list)


@dataclass(slots=True)
class FeedbackInput:
    """
    Input for user feedback recording.
    """
    answer_event_id: UUID
    session_id: UUID
    score: int
    reason_code: Optional[FeedbackReasonCodeEnum] = None
    comment_text: Optional[str] = None

    feedback_channel_code: Optional[ChannelTypeEnum] = None
    is_sampled_request: bool = False
    sampling_policy_version: Optional[str] = None
    metadata_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReuseRecomputeResult:
    """
    Result of reuse candidate recomputation.
    """
    source_answer_event_id: UUID
    feedback_count: int
    avg_feedback_score: Optional[float]
    negative_feedback_count: int
    reuse_score: Optional[float]
    reuse_status: ReuseStatusEnum
    reuse_allowed_effective: bool
    block_reason_code: Optional[str]


@dataclass(slots=True)
class QualityAggregateResult:
    """
    Result of one daily quality aggregate recalculation.
    """
    aggregate_date: date
    channel_code: ChannelTypeEnum
    intent_type: QuestionIntentEnum
    measure_code: Optional[str]
    total_answers: int
    total_feedback: int
    avg_feedback_score: Optional[float]
    reused_answers_count: int
    low_rated_answers_count: int
    failed_validation_count: int


# ============================================================
# Service
# ============================================================

class FeedbackService:
    """
    Production-oriented service for answer history, feedback, and reuse eligibility.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    async def create_answer_event(
        self,
        payload: AnswerEventCreateInput,
    ) -> AnswerEvent:
        """
        Create an answer event and persist all evidence items transactionally.

        Guarantees:
        - question_event must exist
        - evidence ordering is deterministic
        - hashes are derived from normalized content
        - answer_event and evidence_items are written atomically
        """
        self._validate_answer_event_input(payload)

        question_event = await self._get_question_event_or_raise(payload.question_event_id)

        document_set_hash = self._build_document_set_hash(payload.evidence_items)
        evidence_hash = self._build_evidence_hash(payload.evidence_items)
        answer_payload_hash = self._hash_json(payload.answer_payload_json)
        answer_text_hash = self._hash_text(payload.answer_text)

        answer_event = AnswerEvent(
            answer_event_id=uuid4(),
            question_event_id=payload.question_event_id,
            answer_mode=payload.answer_mode,
            answer_text=payload.answer_text.strip(),
            answer_text_short=(payload.answer_text_short or None),
            answer_language_code=payload.answer_language_code,

            confidence_score=self._to_decimal_or_none(payload.confidence_score),
            trust_score_at_generation=self._to_decimal_or_none(payload.trust_score_at_generation),

            validation_status=payload.validation_status,
            deterministic_validation_passed=payload.deterministic_validation_passed,
            semantic_validation_passed=payload.semantic_validation_passed,

            reuse_allowed=payload.reuse_allowed,
            reused_from_answer_event_id=payload.reused_from_answer_event_id,
            reuse_policy_version=payload.reuse_policy_version,
            reuse_decision_payload_json=payload.reuse_decision_payload_json,

            citations_json=payload.citations_json,
            answer_payload_json=payload.answer_payload_json,

            document_set_hash=document_set_hash,
            evidence_hash=evidence_hash,
            answer_payload_hash=answer_payload_hash,
            answer_text_hash=answer_text_hash,

            generation_model_name=payload.generation_model_name,
            generation_prompt_version=payload.generation_prompt_version,
            pipeline_version=payload.pipeline_version,
        )

        try:
            self.db.add(answer_event)
            await self.db.flush()

            evidence_rows = self._build_evidence_rows(
                answer_event_id=answer_event.answer_event_id,
                evidence_items=payload.evidence_items,
            )
            for row in evidence_rows:
                self.db.add(row)

            # Если answer reuse допустим теоретически, сразу создаём/обновляем запись кандидата.
            if payload.reuse_allowed:
                await self._upsert_empty_reuse_candidate(answer_event.answer_event_id)

            await self.db.commit()
        except Exception:
            await self.db.rollback()
            logger.exception("Failed to create answer_event", extra={
                "question_event_id": str(payload.question_event_id),
            })
            raise

        await self.db.refresh(answer_event)

        logger.info(
            "Created answer_event",
            extra={
                "answer_event_id": str(answer_event.answer_event_id),
                "question_event_id": str(question_event.question_event_id),
                "evidence_count": len(payload.evidence_items),
            },
        )
        return answer_event

    async def record_feedback(
        self,
        payload: FeedbackInput,
    ) -> AnswerFeedback:
        """
        Record user feedback for one answer event.

        Rules:
        - answer_event must exist
        - session must exist
        - only one vote per (session_id, answer_event_id)
        - after recording feedback, recompute reuse candidate
        """
        self._validate_feedback_input(payload)

        answer_event = await self._get_answer_event_or_raise(payload.answer_event_id)
        await self._get_session_or_raise(payload.session_id)

        feedback_channel_id: Optional[int] = None
        if payload.feedback_channel_code is not None:
            feedback_channel_id = await self._get_channel_id_by_code(payload.feedback_channel_code)

        feedback = AnswerFeedback(
            feedback_id=uuid4(),
            answer_event_id=payload.answer_event_id,
            session_id=payload.session_id,
            score=payload.score,
            reason_code=payload.reason_code,
            comment_text=(payload.comment_text.strip() if payload.comment_text else None),
            feedback_channel_id=feedback_channel_id,
            is_sampled_request=payload.is_sampled_request,
            sampling_policy_version=payload.sampling_policy_version,
            metadata_json=payload.metadata_json,
        )

        try:
            self.db.add(feedback)
            await self.db.flush()
            await self.db.commit()
        except IntegrityError as exc:
            await self.db.rollback()
            raise ConflictError(
                "Feedback for this answer_event from this session already exists."
            ) from exc
        except Exception:
            await self.db.rollback()
            logger.exception("Failed to record feedback", extra={
                "answer_event_id": str(payload.answer_event_id),
                "session_id": str(payload.session_id),
            })
            raise

        await self.recompute_reuse_candidate(answer_event.answer_event_id)

        logger.info(
            "Recorded feedback",
            extra={
                "feedback_id": str(feedback.feedback_id),
                "answer_event_id": str(answer_event.answer_event_id),
                "score": payload.score,
            },
        )
        return feedback

    async def recompute_reuse_candidate(
        self,
        source_answer_event_id: UUID,
    ) -> ReuseRecomputeResult:
        """
        Recompute whether an answer is eligible for safe reuse.

        Current policy (conservative):
        - answer_event must exist
        - answer_event.reuse_allowed must be True
        - validation_status must be PASSED
        - deterministic and semantic validation must both be True
        - safe_no_answer is never reusable
        - minimum feedback count required: 3
        - average score must be >= 4.2
        - negative feedback (<=2) must be <= 20%
        """
        answer_event = await self._get_answer_event_or_raise(source_answer_event_id)

        summary = await self._get_feedback_summary(source_answer_event_id)

        feedback_count = summary["feedback_count"]
        avg_feedback_score = summary["avg_score"]
        negative_feedback_count = summary["negative_feedback_count"]

        reuse_score = self._calculate_reuse_score(
            answer_event=answer_event,
            feedback_count=feedback_count,
            avg_feedback_score=avg_feedback_score,
            negative_feedback_count=negative_feedback_count,
        )

        reuse_status, reuse_allowed_effective, block_reason_code = self._decide_reuse_status(
            answer_event=answer_event,
            feedback_count=feedback_count,
            avg_feedback_score=avg_feedback_score,
            negative_feedback_count=negative_feedback_count,
            reuse_score=reuse_score,
        )

        candidate = await self._get_reuse_candidate_by_source_answer_event_id(source_answer_event_id)
        if candidate is None:
            candidate = AnswerReuseCandidate(
                reuse_candidate_id=uuid4(),
                source_answer_event_id=source_answer_event_id,
            )
            self.db.add(candidate)

        candidate.reuse_status = reuse_status
        candidate.reuse_allowed_effective = reuse_allowed_effective
        candidate.avg_feedback_score = self._to_decimal_or_none(avg_feedback_score, digits=4)
        candidate.feedback_count = feedback_count
        candidate.negative_feedback_count = negative_feedback_count
        candidate.reuse_score = self._to_decimal_or_none(reuse_score, digits=4)
        candidate.last_revalidated_at = self._utcnow()
        candidate.revalidation_payload_json = {
            "policy": "feedback_reuse_v1",
            "feedback_count": feedback_count,
            "avg_feedback_score": avg_feedback_score,
            "negative_feedback_count": negative_feedback_count,
            "reuse_score": reuse_score,
            "validation_status": str(answer_event.validation_status),
            "deterministic_validation_passed": answer_event.deterministic_validation_passed,
            "semantic_validation_passed": answer_event.semantic_validation_passed,
            "answer_mode": str(answer_event.answer_mode),
        }
        candidate.block_reason_code = block_reason_code

        try:
            await self.db.commit()
        except Exception:
            await self.db.rollback()
            logger.exception("Failed to recompute reuse candidate", extra={
                "source_answer_event_id": str(source_answer_event_id),
            })
            raise

        result = ReuseRecomputeResult(
            source_answer_event_id=source_answer_event_id,
            feedback_count=feedback_count,
            avg_feedback_score=avg_feedback_score,
            negative_feedback_count=negative_feedback_count,
            reuse_score=reuse_score,
            reuse_status=reuse_status,
            reuse_allowed_effective=reuse_allowed_effective,
            block_reason_code=block_reason_code,
        )

        logger.info(
            "Recomputed reuse candidate",
            extra={
                "source_answer_event_id": str(source_answer_event_id),
                "reuse_status": str(reuse_status),
                "reuse_allowed_effective": reuse_allowed_effective,
                "reuse_score": reuse_score,
            },
        )
        return result

    async def build_quality_aggregate(
        self,
        aggregate_date: date,
    ) -> list[QualityAggregateResult]:
        """
        Rebuild daily quality aggregates for one calendar day.

        Strategy:
        - delete existing aggregates for this date
        - rebuild from answer_events/question_events/feedback
        - aggregate by:
            date + channel + intent + measure_code

        Note:
        channel is derived through question -> session -> channel.
        """
        if not isinstance(aggregate_date, date):
            raise ValidationError("aggregate_date must be a date instance.")

        day_start = datetime.combine(aggregate_date, datetime.min.time(), tzinfo=timezone.utc)
        day_end = datetime.combine(aggregate_date, datetime.max.time(), tzinfo=timezone.utc)

        # Удаляем старые агрегаты за день, затем строим заново.
        await self.db.execute(
            delete(QualityAggregateDaily).where(
                QualityAggregateDaily.aggregate_date == aggregate_date
            )
        )

        rows = await self._fetch_quality_aggregate_source_rows(day_start, day_end)

        grouped: dict[tuple[ChannelTypeEnum, QuestionIntentEnum, Optional[str]], dict[str, Any]] = {}

        for row in rows:
            key = (
                row["channel_code"],
                row["intent_type"],
                row["measure_code"],
            )
            bucket = grouped.setdefault(
                key,
                {
                    "total_answers": 0,
                    "total_feedback": 0,
                    "feedback_scores": [],
                    "reused_answers_count": 0,
                    "low_rated_answers_count": 0,
                    "failed_validation_count": 0,
                },
            )

            bucket["total_answers"] += 1

            if row["is_reused"]:
                bucket["reused_answers_count"] += 1

            if row["validation_status"] != ValidationStatusEnum.PASSED:
                bucket["failed_validation_count"] += 1

            if row["feedback_score"] is not None:
                bucket["total_feedback"] += 1
                bucket["feedback_scores"].append(row["feedback_score"])
                if row["feedback_score"] <= 2:
                    bucket["low_rated_answers_count"] += 1

        results: list[QualityAggregateResult] = []

        for (channel_code, intent_type, measure_code), bucket in grouped.items():
            avg_feedback_score = (
                round(sum(bucket["feedback_scores"]) / len(bucket["feedback_scores"]), 4)
                if bucket["feedback_scores"]
                else None
            )

            agg = QualityAggregateDaily(
                aggregate_date=aggregate_date,
                channel_code=channel_code,
                intent_type=intent_type,
                measure_code=measure_code,
                total_answers=bucket["total_answers"],
                total_feedback=bucket["total_feedback"],
                avg_feedback_score=self._to_decimal_or_none(avg_feedback_score, digits=4),
                reused_answers_count=bucket["reused_answers_count"],
                low_rated_answers_count=bucket["low_rated_answers_count"],
                failed_validation_count=bucket["failed_validation_count"],
            )
            self.db.add(agg)

            results.append(
                QualityAggregateResult(
                    aggregate_date=aggregate_date,
                    channel_code=channel_code,
                    intent_type=intent_type,
                    measure_code=measure_code,
                    total_answers=bucket["total_answers"],
                    total_feedback=bucket["total_feedback"],
                    avg_feedback_score=avg_feedback_score,
                    reused_answers_count=bucket["reused_answers_count"],
                    low_rated_answers_count=bucket["low_rated_answers_count"],
                    failed_validation_count=bucket["failed_validation_count"],
                )
            )

        try:
            await self.db.commit()
        except Exception:
            await self.db.rollback()
            logger.exception("Failed to build quality aggregate", extra={
                "aggregate_date": aggregate_date.isoformat(),
            })
            raise

        logger.info(
            "Built daily quality aggregates",
            extra={
                "aggregate_date": aggregate_date.isoformat(),
                "groups_count": len(results),
            },
        )
        return results

    # --------------------------------------------------------
    # Internal validation
    # --------------------------------------------------------

    def _validate_answer_event_input(self, payload: AnswerEventCreateInput) -> None:
        if not payload.answer_text or not payload.answer_text.strip():
            raise ValidationError("answer_text must not be empty.")

        if payload.confidence_score is not None and not (0.0 <= payload.confidence_score <= 1.0):
            raise ValidationError("confidence_score must be between 0 and 1.")

        if payload.trust_score_at_generation is not None and not (0.0 <= payload.trust_score_at_generation <= 1.0):
            raise ValidationError("trust_score_at_generation must be between 0 and 1.")

        if payload.answer_mode == AnswerModeEnum.REUSED_ANSWER and payload.reused_from_answer_event_id is None:
            raise ValidationError(
                "reused_from_answer_event_id is required when answer_mode == REUSED_ANSWER."
            )

        for idx, item in enumerate(payload.evidence_items):
            self._validate_evidence_item_input(item, index=idx)

    def _validate_evidence_item_input(self, item: EvidenceItemInput, index: int) -> None:
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
            raise ValidationError(
                f"Evidence item at index={index} must contain exactly one pointer id."
            )

        if not item.role_code or not item.role_code.strip():
            raise ValidationError(
                f"Evidence item at index={index} must have non-empty role_code."
            )

    def _validate_feedback_input(self, payload: FeedbackInput) -> None:
        if payload.score < 1 or payload.score > 5:
            raise ValidationError("score must be between 1 and 5.")

        if payload.comment_text is not None and len(payload.comment_text) > 5000:
            raise ValidationError("comment_text is too long.")

    # --------------------------------------------------------
    # Internal persistence helpers
    # --------------------------------------------------------

    def _build_evidence_rows(
        self,
        answer_event_id: UUID,
        evidence_items: list[EvidenceItemInput],
    ) -> list[AnswerEvidenceItem]:
        rows: list[AnswerEvidenceItem] = []
        for order_index, item in enumerate(evidence_items, start=1):
            rows.append(
                AnswerEvidenceItem(
                    answer_evidence_item_id=uuid4(),
                    answer_event_id=answer_event_id,
                    evidence_order=order_index,
                    evidence_item_type=item.evidence_item_type,
                    document_id=item.document_id,
                    block_id=item.block_id,
                    table_id=item.table_id,
                    table_row_id=item.table_row_id,
                    legal_fact_id=item.legal_fact_id,
                    citation_json=item.citation_json,
                    document_file_hash=item.document_file_hash,
                    document_content_hash=item.document_content_hash,
                    role_code=item.role_code,
                )
            )
        return rows

    async def _upsert_empty_reuse_candidate(self, source_answer_event_id: UUID) -> None:
        candidate = await self._get_reuse_candidate_by_source_answer_event_id(source_answer_event_id)
        if candidate is not None:
            return

        candidate = AnswerReuseCandidate(
            reuse_candidate_id=uuid4(),
            source_answer_event_id=source_answer_event_id,
            reuse_status=ReuseStatusEnum.NEEDS_REVALIDATION,
            reuse_allowed_effective=False,
            feedback_count=0,
            negative_feedback_count=0,
            revalidation_payload_json={"policy": "feedback_reuse_v1", "state": "initialized"},
        )
        self.db.add(candidate)
        await self.db.flush()

    # --------------------------------------------------------
    # Reuse policy
    # --------------------------------------------------------

    def _calculate_reuse_score(
        self,
        *,
        answer_event: AnswerEvent,
        feedback_count: int,
        avg_feedback_score: Optional[float],
        negative_feedback_count: int,
    ) -> Optional[float]:
        """
        Conservative heuristic reuse score in range [0, 1].

        This is intentionally strict for early production.
        """
        if not answer_event.reuse_allowed:
            return 0.0

        if answer_event.answer_mode == AnswerModeEnum.SAFE_NO_ANSWER:
            return 0.0

        if avg_feedback_score is None:
            return 0.0

        negative_ratio = (negative_feedback_count / feedback_count) if feedback_count > 0 else 1.0

        validation_component = 1.0 if (
            answer_event.validation_status == ValidationStatusEnum.PASSED
            and answer_event.deterministic_validation_passed
            and answer_event.semantic_validation_passed
        ) else 0.0

        feedback_volume_component = min(feedback_count / 5.0, 1.0)
        avg_score_component = max(min((avg_feedback_score - 1.0) / 4.0, 1.0), 0.0)
        negative_penalty_component = max(1.0 - negative_ratio, 0.0)

        raw_score = (
            0.40 * validation_component
            + 0.20 * feedback_volume_component
            + 0.30 * avg_score_component
            + 0.10 * negative_penalty_component
        )

        return round(max(min(raw_score, 1.0), 0.0), 4)

    def _decide_reuse_status(
        self,
        *,
        answer_event: AnswerEvent,
        feedback_count: int,
        avg_feedback_score: Optional[float],
        negative_feedback_count: int,
        reuse_score: Optional[float],
    ) -> tuple[ReuseStatusEnum, bool, Optional[str]]:
        """
        Decides whether answer can be considered a safe reuse candidate.
        """
        if not answer_event.reuse_allowed:
            return ReuseStatusEnum.BLOCKED, False, "answer_event_reuse_not_allowed"

        if answer_event.answer_mode == AnswerModeEnum.SAFE_NO_ANSWER:
            return ReuseStatusEnum.BLOCKED, False, "safe_no_answer_not_reusable"

        if answer_event.validation_status != ValidationStatusEnum.PASSED:
            return ReuseStatusEnum.BLOCKED, False, "validation_status_not_passed"

        if not answer_event.deterministic_validation_passed:
            return ReuseStatusEnum.BLOCKED, False, "deterministic_validation_failed"

        if not answer_event.semantic_validation_passed:
            return ReuseStatusEnum.BLOCKED, False, "semantic_validation_failed"

        if feedback_count < 3:
            return ReuseStatusEnum.NEEDS_REVALIDATION, False, "insufficient_feedback_count"

        if avg_feedback_score is None:
            return ReuseStatusEnum.NEEDS_REVALIDATION, False, "avg_feedback_unavailable"

        negative_ratio = negative_feedback_count / feedback_count if feedback_count > 0 else 1.0
        if avg_feedback_score < 4.2:
            return ReuseStatusEnum.BLOCKED, False, "avg_feedback_below_threshold"

        if negative_ratio > 0.20:
            return ReuseStatusEnum.BLOCKED, False, "negative_feedback_ratio_too_high"

        if reuse_score is None or reuse_score < 0.80:
            return ReuseStatusEnum.NEEDS_REVALIDATION, False, "reuse_score_below_threshold"

        return ReuseStatusEnum.APPROVED, True, None

    # --------------------------------------------------------
    # Aggregate source fetch
    # --------------------------------------------------------

    async def _fetch_quality_aggregate_source_rows(
        self,
        day_start: datetime,
        day_end: datetime,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(
                Channel.channel_code.label("channel_code"),
                QuestionEvent.intent_type.label("intent_type"),
                QuestionEvent.measure_code.label("measure_code"),
                AnswerEvent.answer_event_id.label("answer_event_id"),
                AnswerEvent.answer_mode.label("answer_mode"),
                AnswerEvent.validation_status.label("validation_status"),
                AnswerFeedback.score.label("feedback_score"),
            )
            .join(ConversationSession, ConversationSession.channel_id == Channel.channel_id)
            .join(QuestionEvent, QuestionEvent.session_id == ConversationSession.session_id)
            .join(AnswerEvent, AnswerEvent.question_event_id == QuestionEvent.question_event_id)
            .outerjoin(AnswerFeedback, AnswerFeedback.answer_event_id == AnswerEvent.answer_event_id)
            .where(AnswerEvent.created_at >= day_start)
            .where(AnswerEvent.created_at <= day_end)
        )

        result = await self.db.execute(stmt)
        rows = result.mappings().all()

        flattened: list[dict[str, Any]] = []
        seen_answers: set[UUID] = set()

        for row in rows:
            answer_id = row["answer_event_id"]
            is_reused = row["answer_mode"] == AnswerModeEnum.REUSED_ANSWER

            flattened.append({
                "channel_code": row["channel_code"],
                "intent_type": row["intent_type"],
                "measure_code": row["measure_code"],
                "answer_event_id": answer_id,
                "is_reused": is_reused,
                "validation_status": row["validation_status"],
                "feedback_score": row["feedback_score"],
                "count_answer_once": answer_id not in seen_answers,
            })
            seen_answers.add(answer_id)

        normalized: list[dict[str, Any]] = []
        already_counted_answers: set[UUID] = set()

        for row in flattened:
            answer_id = row["answer_event_id"]
            if answer_id in already_counted_answers:
                total_answer_increment = 0
                reused_increment = 0
                failed_validation_increment = 0
            else:
                total_answer_increment = 1
                reused_increment = 1 if row["is_reused"] else 0
                failed_validation_increment = 1 if row["validation_status"] != ValidationStatusEnum.PASSED else 0
                already_counted_answers.add(answer_id)

            normalized.append({
                "channel_code": row["channel_code"],
                "intent_type": row["intent_type"],
                "measure_code": row["measure_code"],
                "feedback_score": row["feedback_score"],
                "total_answer_increment": total_answer_increment,
                "reused_increment": reused_increment,
                "failed_validation_increment": failed_validation_increment,
                "validation_status": row["validation_status"],
                "is_reused": row["is_reused"],
            })

        return normalized

    # --------------------------------------------------------
    # Query helpers
    # --------------------------------------------------------

    async def _get_question_event_or_raise(self, question_event_id: UUID) -> QuestionEvent:
        stmt: Select[Any] = select(QuestionEvent).where(
            QuestionEvent.question_event_id == question_event_id
        )
        result = await self.db.execute(stmt)
        obj = result.scalar_one_or_none()
        if obj is None:
            raise NotFoundError(f"QuestionEvent not found: {question_event_id}")
        return obj

    async def _get_answer_event_or_raise(self, answer_event_id: UUID) -> AnswerEvent:
        stmt: Select[Any] = select(AnswerEvent).where(
            AnswerEvent.answer_event_id == answer_event_id
        )
        result = await self.db.execute(stmt)
        obj = result.scalar_one_or_none()
        if obj is None:
            raise NotFoundError(f"AnswerEvent not found: {answer_event_id}")
        return obj

    async def _get_session_or_raise(self, session_id: UUID) -> ConversationSession:
        stmt: Select[Any] = select(ConversationSession).where(
            ConversationSession.session_id == session_id
        )
        result = await self.db.execute(stmt)
        obj = result.scalar_one_or_none()
        if obj is None:
            raise NotFoundError(f"ConversationSession not found: {session_id}")
        return obj

    async def _get_channel_id_by_code(self, channel_code: ChannelTypeEnum) -> int:
        stmt: Select[Any] = select(Channel.channel_id).where(Channel.channel_code == channel_code)
        result = await self.db.execute(stmt)
        channel_id = result.scalar_one_or_none()
        if channel_id is None:
            raise NotFoundError(f"Channel not found: {channel_code}")
        return int(channel_id)

    async def _get_reuse_candidate_by_source_answer_event_id(
        self,
        source_answer_event_id: UUID,
    ) -> Optional[AnswerReuseCandidate]:
        stmt: Select[Any] = select(AnswerReuseCandidate).where(
            AnswerReuseCandidate.source_answer_event_id == source_answer_event_id
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_feedback_summary(self, answer_event_id: UUID) -> dict[str, Any]:
        stmt = (
            select(
                func.count(AnswerFeedback.feedback_id).label("feedback_count"),
                func.avg(AnswerFeedback.score).label("avg_score"),
                func.sum(
                    case(
                        (AnswerFeedback.score <= 2, 1),
                        else_=0,
                    )
                ).label("negative_feedback_count"),
            )
            .where(AnswerFeedback.answer_event_id == answer_event_id)
        )

        result = await self.db.execute(stmt)
        row = result.mappings().one()

        feedback_count = int(row["feedback_count"] or 0)
        avg_score_raw = row["avg_score"]
        negative_feedback_count = int(row["negative_feedback_count"] or 0)

        return {
            "feedback_count": feedback_count,
            "avg_score": float(avg_score_raw) if avg_score_raw is not None else None,
            "negative_feedback_count": negative_feedback_count,
        }

    # --------------------------------------------------------
    # Hashing helpers
    # --------------------------------------------------------

    def _build_document_set_hash(self, evidence_items: Iterable[EvidenceItemInput]) -> str:
        """
        Hash only the distinct document-level hashes used in evidence.
        """
        documents = sorted({
            item.document_content_hash or item.document_file_hash or ""
            for item in evidence_items
            if (item.document_content_hash or item.document_file_hash)
        })
        return self._hash_json({"documents": documents})

    def _build_evidence_hash(self, evidence_items: Iterable[EvidenceItemInput]) -> str:
        """
        Stable hash of exact evidence composition and ordering.
        """
        normalized_items: list[dict[str, Any]] = []
        for idx, item in enumerate(evidence_items, start=1):
            normalized_items.append({
                "order": idx,
                "evidence_item_type": str(item.evidence_item_type),
                "role_code": item.role_code,
                "document_id": str(item.document_id) if item.document_id else None,
                "block_id": str(item.block_id) if item.block_id else None,
                "table_id": str(item.table_id) if item.table_id else None,
                "table_row_id": str(item.table_row_id) if item.table_row_id else None,
                "legal_fact_id": str(item.legal_fact_id) if item.legal_fact_id else None,
                "document_file_hash": item.document_file_hash,
                "document_content_hash": item.document_content_hash,
                "citation_json": self._normalize_json(item.citation_json),
            })
        return self._hash_json(normalized_items)

    def _hash_text(self, value: str) -> str:
        normalized = " ".join(value.strip().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _hash_json(self, payload: Any) -> str:
        normalized = self._normalize_json(payload)
        raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _normalize_json(self, payload: Any) -> Any:
        """
        Deterministic JSON normalization for hashing.
        """
        if isinstance(payload, dict):
            return {str(k): self._normalize_json(v) for k, v in sorted(payload.items(), key=lambda x: str(x[0]))}
        if isinstance(payload, list):
            return [self._normalize_json(v) for v in payload]
        if isinstance(payload, tuple):
            return [self._normalize_json(v) for v in payload]
        if isinstance(payload, UUID):
            return str(payload)
        if isinstance(payload, datetime):
            return payload.astimezone(timezone.utc).isoformat()
        return payload

    # --------------------------------------------------------
    # Primitive helpers
    # --------------------------------------------------------

    def _to_decimal_or_none(
        self,
        value: Optional[float],
        *,
        digits: int = 4,
    ) -> Optional[Decimal]:
        if value is None:
            return None
        quantized = round(float(value), digits)
        return Decimal(str(quantized))

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)