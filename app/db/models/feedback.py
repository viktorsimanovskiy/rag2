# ============================================================
# File: app/db/models/feedback.py
# Purpose:
#   ORM models for:
#   - channels
#   - conversation sessions
#   - question events
#   - answer events
#   - answer evidence items
#   - answer feedback
#   - answer reuse candidates
#   - daily quality aggregates
#
# Notes:
#   - SQLAlchemy 2.x style
#   - PostgreSQL-oriented
#   - pgvector support included
#   - JSONB used for extensibility
#
# Important:
#   This file assumes:
#   - PostgreSQL
#   - pgvector extension installed
#   - a shared DeclarativeBase exists at app.db.base
# ============================================================

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
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
    Numeric,
    SmallInteger,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ENUM as PGEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover
    Vector = None  # type: ignore

from app.db.base import Base
from app.db.models.enums import (
    AnswerModeEnum,
    ChannelTypeEnum,
    EvidenceItemTypeEnum,
    FeedbackReasonCodeEnum,
    QuestionIntentEnum,
    ReuseStatusEnum,
    ValidationStatusEnum,
)


# ============================================================
# Shared enum declarations
# ============================================================

channel_type_enum_pg = PGEnum(
    ChannelTypeEnum,
    name="channel_type_enum",
    create_type=False,
)

question_intent_enum_pg = PGEnum(
    QuestionIntentEnum,
    name="question_intent_enum",
    create_type=False,
)

answer_mode_enum_pg = PGEnum(
    AnswerModeEnum,
    name="answer_mode_enum",
    create_type=False,
)

validation_status_enum_pg = PGEnum(
    ValidationStatusEnum,
    name="validation_status_enum",
    create_type=False,
)

evidence_item_type_enum_pg = PGEnum(
    EvidenceItemTypeEnum,
    name="evidence_item_type_enum",
    create_type=False,
)

feedback_reason_code_enum_pg = PGEnum(
    FeedbackReasonCodeEnum,
    name="feedback_reason_code_enum",
    create_type=False,
)

reuse_status_enum_pg = PGEnum(
    ReuseStatusEnum,
    name="reuse_status_enum",
    create_type=False,
)


# ============================================================
# Helpers
# ============================================================

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# ORM Models
# ============================================================

class Channel(Base):
    __tablename__ = "channels"

    channel_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    channel_code: Mapped[ChannelTypeEnum] = mapped_column(
        channel_type_enum_pg,
        nullable=False,
        unique=True,
    )

    channel_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("TRUE"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("NOW()"),
    )

    sessions: Mapped[list["ConversationSession"]] = relationship(
        "ConversationSession",
        back_populates="channel",
        cascade="save-update, merge",
        lazy="selectin",
    )

    feedback_items: Mapped[list["AnswerFeedback"]] = relationship(
        "AnswerFeedback",
        back_populates="feedback_channel",
        cascade="save-update, merge",
        lazy="selectin",
    )


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"
    __table_args__ = (
        UniqueConstraint(
            "channel_id",
            "external_session_id",
            name="uq_conversation_sessions_unique",
        ),
        Index("idx_conversation_sessions_channel", "channel_id"),
        Index("idx_conversation_sessions_last_activity", "session_last_activity_at"),
    )

    session_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    channel_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("channels.channel_id", ondelete="RESTRICT"),
        nullable=False,
    )

    external_session_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    external_user_id: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    external_chat_id: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    user_platform_name: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    session_started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("NOW()"),
    )

    session_last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("NOW()"),
    )

    metadata_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    channel: Mapped["Channel"] = relationship(
        "Channel",
        back_populates="sessions",
        lazy="joined",
    )

    question_events: Mapped[list["QuestionEvent"]] = relationship(
        "QuestionEvent",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )

    feedback_items: Mapped[list["AnswerFeedback"]] = relationship(
        "AnswerFeedback",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )


class QuestionEvent(Base):
    __tablename__ = "question_events"
    __table_args__ = (
        Index("idx_question_events_session", "session_id", "created_at"),
        Index("idx_question_events_intent", "intent_type"),
        Index("idx_question_events_measure_code", "measure_code"),
        Index("idx_question_events_created_at", "created_at"),
    )

    question_event_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    session_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("conversation_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )

    question_text_raw: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    question_text_normalized: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    question_language_code: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="ru",
        server_default=text("'ru'"),
    )

    intent_type: Mapped[QuestionIntentEnum] = mapped_column(
        question_intent_enum_pg,
        nullable=False,
        default=QuestionIntentEnum.OTHER,
        server_default=text("'other'"),
    )

    measure_code: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    subject_category_code: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    query_constraints_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    routing_payload_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    classifier_version: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    embedding_model_name: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    if Vector is not None:
        question_embedding: Mapped[Optional[list[float]]] = mapped_column(
            Vector(1536),
            nullable=True,
        )
    else:  # fallback for environments without pgvector package
        question_embedding: Mapped[Optional[str]] = mapped_column(
            Text,
            nullable=True,
        )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("NOW()"),
    )

    session: Mapped["ConversationSession"] = relationship(
        "ConversationSession",
        back_populates="question_events",
        lazy="joined",
    )

    answer_events: Mapped[list["AnswerEvent"]] = relationship(
        "AnswerEvent",
        back_populates="question_event",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )


class AnswerEvent(Base):
    __tablename__ = "answer_events"
    __table_args__ = (
        CheckConstraint(
            "confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)",
            name="chk_answer_events_confidence_range",
        ),
        CheckConstraint(
            "trust_score_at_generation IS NULL OR (trust_score_at_generation >= 0 AND trust_score_at_generation <= 1)",
            name="chk_answer_events_trust_range",
        ),
        Index("idx_answer_events_question", "question_event_id"),
        Index("idx_answer_events_created_at", "created_at"),
        Index("idx_answer_events_reuse_allowed", "reuse_allowed"),
        Index("idx_answer_events_reused_from", "reused_from_answer_event_id"),
        Index("idx_answer_events_validation_status", "validation_status"),
        Index("idx_answer_events_document_set_hash", "document_set_hash"),
        Index("idx_answer_events_evidence_hash", "evidence_hash"),
    )

    answer_event_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    question_event_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("question_events.question_event_id", ondelete="CASCADE"),
        nullable=False,
    )

    answer_mode: Mapped[AnswerModeEnum] = mapped_column(
        answer_mode_enum_pg,
        nullable=False,
    )

    answer_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    answer_text_short: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    answer_language_code: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="ru",
        server_default=text("'ru'"),
    )

    confidence_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )

    trust_score_at_generation: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )

    validation_status: Mapped[ValidationStatusEnum] = mapped_column(
        validation_status_enum_pg,
        nullable=False,
        default=ValidationStatusEnum.NOT_RUN,
        server_default=text("'not_run'"),
    )

    deterministic_validation_passed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("FALSE"),
    )

    semantic_validation_passed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("FALSE"),
    )

    reuse_allowed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("FALSE"),
    )

    reused_from_answer_event_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("answer_events.answer_event_id", ondelete="SET NULL"),
        nullable=True,
    )

    reuse_policy_version: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    reuse_decision_payload_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    citations_json: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )

    answer_payload_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    document_set_hash: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    evidence_hash: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    answer_payload_hash: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    answer_text_hash: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    generation_model_name: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    generation_prompt_version: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    pipeline_version: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("NOW()"),
    )

    question_event: Mapped["QuestionEvent"] = relationship(
        "QuestionEvent",
        back_populates="answer_events",
        lazy="joined",
    )

    reused_from_answer_event: Mapped[Optional["AnswerEvent"]] = relationship(
        "AnswerEvent",
        remote_side="AnswerEvent.answer_event_id",
        lazy="joined",
    )

    evidence_items: Mapped[list["AnswerEvidenceItem"]] = relationship(
        "AnswerEvidenceItem",
        back_populates="answer_event",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
        order_by="AnswerEvidenceItem.evidence_order",
    )

    feedback_items: Mapped[list["AnswerFeedback"]] = relationship(
        "AnswerFeedback",
        back_populates="answer_event",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )

    reuse_candidate: Mapped[Optional["AnswerReuseCandidate"]] = relationship(
        "AnswerReuseCandidate",
        back_populates="source_answer_event",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )


class AnswerEvidenceItem(Base):
    __tablename__ = "answer_evidence_items"
    __table_args__ = (
        UniqueConstraint(
            "answer_event_id",
            "evidence_order",
            name="uq_answer_evidence_items_order",
        ),
        CheckConstraint(
            """
            ((CASE WHEN document_id IS NOT NULL THEN 1 ELSE 0 END) +
             (CASE WHEN block_id IS NOT NULL THEN 1 ELSE 0 END) +
             (CASE WHEN table_id IS NOT NULL THEN 1 ELSE 0 END) +
             (CASE WHEN table_row_id IS NOT NULL THEN 1 ELSE 0 END) +
             (CASE WHEN legal_fact_id IS NOT NULL THEN 1 ELSE 0 END)) = 1
            """,
            name="chk_answer_evidence_exactly_one_pointer",
        ),
        Index("idx_answer_evidence_items_answer", "answer_event_id"),
        Index("idx_answer_evidence_items_document", "document_id"),
        Index("idx_answer_evidence_items_table", "table_id"),
        Index("idx_answer_evidence_items_fact", "legal_fact_id"),
        Index("idx_answer_evidence_items_doc_content_hash", "document_content_hash"),
    )

    answer_evidence_item_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    answer_event_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("answer_events.answer_event_id", ondelete="CASCADE"),
        nullable=False,
    )

    evidence_order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    evidence_item_type: Mapped[EvidenceItemTypeEnum] = mapped_column(
        evidence_item_type_enum_pg,
        nullable=False,
    )

    document_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    block_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    table_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    table_row_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    legal_fact_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    citation_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    document_file_hash: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    document_content_hash: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    role_code: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("NOW()"),
    )

    answer_event: Mapped["AnswerEvent"] = relationship(
        "AnswerEvent",
        back_populates="evidence_items",
        lazy="joined",
    )


class AnswerFeedback(Base):
    __tablename__ = "answer_feedback"
    __table_args__ = (
        CheckConstraint(
            "score >= 1 AND score <= 5",
            name="chk_answer_feedback_score_range",
        ),
        Index("idx_answer_feedback_answer", "answer_event_id"),
        Index("idx_answer_feedback_session", "session_id"),
        Index("idx_answer_feedback_submitted_at", "submitted_at"),
        Index("idx_answer_feedback_score", "score"),
        UniqueConstraint(
            "session_id",
            "answer_event_id",
            name="uq_answer_feedback_one_vote_per_session_answer",
        ),
    )

    feedback_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    answer_event_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("answer_events.answer_event_id", ondelete="CASCADE"),
        nullable=False,
    )

    session_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("conversation_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )

    score: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
    )

    reason_code: Mapped[Optional[FeedbackReasonCodeEnum]] = mapped_column(
        feedback_reason_code_enum_pg,
        nullable=True,
    )

    comment_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    feedback_channel_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("channels.channel_id", ondelete="SET NULL"),
        nullable=True,
    )

    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("NOW()"),
    )

    is_sampled_request: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("FALSE"),
    )

    sampling_policy_version: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    is_resolved_for_analytics: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("TRUE"),
    )

    metadata_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    answer_event: Mapped["AnswerEvent"] = relationship(
        "AnswerEvent",
        back_populates="feedback_items",
        lazy="joined",
    )

    session: Mapped["ConversationSession"] = relationship(
        "ConversationSession",
        back_populates="feedback_items",
        lazy="joined",
    )

    feedback_channel: Mapped[Optional["Channel"]] = relationship(
        "Channel",
        back_populates="feedback_items",
        lazy="joined",
    )


class AnswerReuseCandidate(Base):
    __tablename__ = "answer_reuse_candidates"
    __table_args__ = (
        CheckConstraint(
            "avg_feedback_score IS NULL OR (avg_feedback_score >= 1 AND avg_feedback_score <= 5)",
            name="chk_answer_reuse_candidates_avg_feedback_range",
        ),
        CheckConstraint(
            "reuse_score IS NULL OR (reuse_score >= 0 AND reuse_score <= 1)",
            name="chk_answer_reuse_candidates_reuse_score_range",
        ),
        Index("idx_answer_reuse_candidates_status", "reuse_status"),
        Index("idx_answer_reuse_candidates_allowed", "reuse_allowed_effective"),
        Index("idx_answer_reuse_candidates_reuse_score", "reuse_score"),
    )

    reuse_candidate_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    source_answer_event_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("answer_events.answer_event_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    reuse_status: Mapped[ReuseStatusEnum] = mapped_column(
        reuse_status_enum_pg,
        nullable=False,
        default=ReuseStatusEnum.NEEDS_REVALIDATION,
        server_default=text("'needs_revalidation'"),
    )

    reuse_allowed_effective: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("FALSE"),
    )

    avg_feedback_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )

    feedback_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    negative_feedback_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    reuse_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )

    last_revalidated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    revalidation_payload_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    block_reason_code: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
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

    source_answer_event: Mapped["AnswerEvent"] = relationship(
        "AnswerEvent",
        back_populates="reuse_candidate",
        lazy="joined",
    )


class QualityAggregateDaily(Base):
    __tablename__ = "quality_aggregates_daily"
    __table_args__ = (
        UniqueConstraint(
            "aggregate_date",
            "channel_code",
            "intent_type",
            "measure_code",
            name="uq_quality_aggregates_daily",
        ),
        Index("idx_quality_aggregates_daily_date", "aggregate_date"),
        Index("idx_quality_aggregates_daily_measure", "measure_code"),
    )

    quality_aggregate_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    aggregate_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    channel_code: Mapped[ChannelTypeEnum] = mapped_column(
        channel_type_enum_pg,
        nullable=False,
        default=ChannelTypeEnum.UNKNOWN,
        server_default=text("'unknown'"),
    )

    intent_type: Mapped[QuestionIntentEnum] = mapped_column(
        question_intent_enum_pg,
        nullable=False,
        default=QuestionIntentEnum.OTHER,
        server_default=text("'other'"),
    )

    measure_code: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    total_answers: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    total_feedback: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    avg_feedback_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )

    reused_answers_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    low_rated_answers_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    failed_validation_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("NOW()"),
    )