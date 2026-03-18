# ============================================================
# File: app/services/answers/answer_orchestrator.py
# Purpose:
#   Central orchestration service for processing incoming user questions.
#
# Responsibilities:
#   - resolve/create conversation session
#   - create question_event
#   - attempt safe answer reuse
#   - fallback to full RAG generation
#   - persist answer_event
#   - prepare response payload for messenger adapters
#   - decide whether feedback request should be shown
#
# Design principles:
#   - orchestrator coordinates, but does not implement retrieval/generation
#   - conservative reuse
#   - production-oriented traceability
#   - transport-agnostic core logic
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.feedback import (
    AnswerEvent,
    Channel,
    ConversationSession,
    QuestionEvent,
)
from app.db.models.enums import (
    AnswerModeEnum,
    ChannelTypeEnum,
    QuestionIntentEnum,
    ValidationStatusEnum,
)
from app.services.answers.runtime_answer_service import RuntimeAnswerInput
from app.services.feedback.feedback_service import (
    AnswerEventCreateInput,
    EvidenceItemInput,
    FeedbackService,
)
from app.services.generation.generation_pipeline import GenerationResult
from app.services.reuse.reuse_gate import (
    ReuseDecision,
    ReuseGate,
    ReuseQueryInput,
)

logger = logging.getLogger(__name__)


# ============================================================
# PLACEHOLDER IMPORTS / INTERFACES
# Replace these with actual implementations when you add them.
# ============================================================

class IntentClassifierProtocol:
    async def classify(self, question_text: str) -> dict[str, Any]:
        raise NotImplementedError


class QuestionNormalizerProtocol:
    async def normalize(self, question_text: str) -> str:
        raise NotImplementedError


class QuestionEmbeddingProtocol:
    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class RuntimeAnswerServiceProtocol:
    async def build_answer(self, payload: RuntimeAnswerInput) -> Any:
        raise NotImplementedError


class SamplingPolicyProtocol:
    async def should_request_feedback(self, payload: "SamplingDecisionInput") -> bool:
        raise NotImplementedError


# ============================================================
# Exceptions
# ============================================================

class AnswerOrchestratorError(Exception):
    """Base orchestrator error."""


class OrchestratorValidationError(AnswerOrchestratorError):
    """Raised when input validation fails."""


class OrchestratorNotFoundError(AnswerOrchestratorError):
    """Raised when required entities do not exist."""


# ============================================================
# DTOs
# ============================================================

@dataclass(slots=True)
class UserQuestionInput:
    """
    Raw user question coming from a messenger adapter or API.
    """
    channel_code: ChannelTypeEnum
    external_session_id: str
    external_user_id: Optional[str]
    external_chat_id: Optional[str]
    user_platform_name: Optional[str]

    question_text: str
    language_code: str = "ru"

    request_metadata_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QuestionRoutingResult:
    """
    Output of lightweight question understanding.
    """
    question_text_normalized: str
    intent_type: QuestionIntentEnum
    measure_code: Optional[str]
    subject_category_code: Optional[str]
    classifier_version: Optional[str]
    embedding_model_name: Optional[str]
    routing_payload_json: dict[str, Any] = field(default_factory=dict)
    query_constraints_json: dict[str, Any] = field(default_factory=dict)
    question_embedding: Optional[list[float]] = None


@dataclass(slots=True)
class SamplingDecisionInput:
    """
    Input for feedback sampling policy.
    """
    channel_code: ChannelTypeEnum
    session_id: UUID
    question_event_id: UUID
    answer_event_id: UUID
    answer_mode: AnswerModeEnum
    intent_type: QuestionIntentEnum
    measure_code: Optional[str]
    confidence_score: Optional[float]
    request_metadata_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OutgoingAnswerPayload:
    """
    Transport-agnostic result for messenger adapters.
    """
    answer_event_id: UUID
    session_id: UUID
    question_event_id: UUID

    answer_text: str
    answer_text_short: Optional[str]
    citations_json: list[dict[str, Any]]

    answer_mode: AnswerModeEnum
    was_reused: bool
    reused_from_answer_event_id: Optional[UUID]

    should_request_feedback: bool
    feedback_payload_json: dict[str, Any]

    delivery_payload_json: dict[str, Any] = field(default_factory=dict)
    debug_payload_json: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Orchestrator
# ============================================================

class AnswerOrchestrator:
    """
    Central service that coordinates question processing end-to-end.
    """

    def __init__(
        self,
        db: AsyncSession,
        *,
        feedback_service: FeedbackService,
        reuse_gate: ReuseGate,
        intent_classifier: IntentClassifierProtocol,
        question_normalizer: QuestionNormalizerProtocol,
        question_embedding_service: Optional[QuestionEmbeddingProtocol],
        runtime_answer_service: RuntimeAnswerServiceProtocol,
        sampling_policy: SamplingPolicyProtocol,
    ) -> None:
        self.db = db
        self.feedback_service = feedback_service
        self.reuse_gate = reuse_gate
        self.intent_classifier = intent_classifier
        self.question_normalizer = question_normalizer
        self.question_embedding_service = question_embedding_service
        self.runtime_answer_service = runtime_answer_service
        self.sampling_policy = sampling_policy

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    async def handle_user_question(
        self,
        payload: UserQuestionInput,
    ) -> OutgoingAnswerPayload:
        """
        Main entrypoint for processing a user question.

        Flow:
        1. Validate input
        2. Resolve or create conversation session
        3. Build routing result
        4. Create question_event
        5. Try safe reuse
        6. If reuse approved -> persist reused answer_event
        7. Else -> run full generation pipeline and persist answer_event
        8. Decide whether to ask for feedback
        9. Return transport-ready payload
        """
        self._validate_user_question_input(payload)

        session = await self._resolve_or_create_session(payload)
        routing = await self._build_question_routing(payload.question_text)

        question_event = await self._create_question_event(
            session_id=session.session_id,
            question_text_raw=payload.question_text,
            language_code=payload.language_code,
            routing=routing,
        )

        reuse_decision = await self.reuse_gate.build_reuse_decision(
            ReuseQueryInput(
                question_event_id=question_event.question_event_id,
                similarity_threshold=0.90,
                max_candidates=20,
                allow_measure_mismatch=False,
                allow_subject_category_mismatch=False,
            )
        )

        if reuse_decision.should_reuse and reuse_decision.source_answer_event_id:
            answer_event = await self._persist_reused_answer_event(
                question_event=question_event,
                reuse_decision=reuse_decision,
            )
        else:
            generation_result = await self._run_full_generation(
                payload=payload,
                question_event=question_event,
                routing=routing,
                session=session,
                reuse_decision=reuse_decision,
            )
            answer_event = await self._persist_generated_answer_event(
                question_event=question_event,
                generation_result=generation_result,
            )

        should_request_feedback = await self.sampling_policy.should_request_feedback(
            SamplingDecisionInput(
                channel_code=payload.channel_code,
                session_id=session.session_id,
                question_event_id=question_event.question_event_id,
                answer_event_id=answer_event.answer_event_id,
                answer_mode=answer_event.answer_mode,
                intent_type=question_event.intent_type,
                measure_code=question_event.measure_code,
                confidence_score=float(answer_event.confidence_score) if answer_event.confidence_score is not None else None,
                request_metadata_json=payload.request_metadata_json,
            )
        )

        result = self._build_outgoing_payload(
            session=session,
            question_event=question_event,
            answer_event=answer_event,
            should_request_feedback=should_request_feedback,
            reuse_decision=reuse_decision,
        )

        logger.info(
            "Processed user question",
            extra={
                "session_id": str(session.session_id),
                "question_event_id": str(question_event.question_event_id),
                "answer_event_id": str(answer_event.answer_event_id),
                "answer_mode": str(answer_event.answer_mode),
                "reuse_approved": reuse_decision.should_reuse,
                "feedback_requested": should_request_feedback,
            },
        )
        return result

    # --------------------------------------------------------
    # Session handling
    # --------------------------------------------------------

    async def _resolve_or_create_session(
        self,
        payload: UserQuestionInput,
    ) -> ConversationSession:
        channel = await self._get_channel_or_raise(payload.channel_code)

        stmt: Select[Any] = select(ConversationSession).where(
            ConversationSession.channel_id == channel.channel_id,
            ConversationSession.external_session_id == payload.external_session_id,
        )
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            session = ConversationSession(
                session_id=uuid4(),
                channel_id=channel.channel_id,
                external_session_id=payload.external_session_id,
                external_user_id=payload.external_user_id,
                external_chat_id=payload.external_chat_id,
                user_platform_name=payload.user_platform_name,
                metadata_json={
                    "created_by": "answer_orchestrator",
                    "initial_request_metadata": payload.request_metadata_json,
                },
            )
            self.db.add(session)
            await self.db.commit()
            await self.db.refresh(session)
            return session

        session.external_user_id = payload.external_user_id
        session.external_chat_id = payload.external_chat_id
        session.user_platform_name = payload.user_platform_name
        session.session_last_activity_at = self._utcnow()

        await self.db.commit()
        await self.db.refresh(session)
        return session

    # --------------------------------------------------------
    # Question routing / understanding
    # --------------------------------------------------------

    async def _build_question_routing(
        self,
        question_text: str,
    ) -> QuestionRoutingResult:
        normalized_text = await self.question_normalizer.normalize(question_text)
        classification = await self.intent_classifier.classify(normalized_text)

        intent_value = classification.get("intent_type", QuestionIntentEnum.OTHER)
        intent_type = (
            intent_value
            if isinstance(intent_value, QuestionIntentEnum)
            else QuestionIntentEnum(intent_value)
        )

        question_embedding: Optional[list[float]] = None
        embedding_model_name: Optional[str] = None

        if self.question_embedding_service is not None:
            question_embedding = await self.question_embedding_service.embed(normalized_text)
            embedding_model_name = classification.get("embedding_model_name")

        return QuestionRoutingResult(
            question_text_normalized=normalized_text,
            intent_type=intent_type,
            measure_code=classification.get("measure_code"),
            subject_category_code=classification.get("subject_category_code"),
            classifier_version=classification.get("classifier_version"),
            embedding_model_name=embedding_model_name,
            routing_payload_json=classification.get("routing_payload_json", {}),
            query_constraints_json=classification.get("query_constraints_json", {}),
            question_embedding=question_embedding,
        )

    async def _create_question_event(
        self,
        *,
        session_id: UUID,
        question_text_raw: str,
        language_code: str,
        routing: QuestionRoutingResult,
    ) -> QuestionEvent:
        question_event = QuestionEvent(
            question_event_id=uuid4(),
            session_id=session_id,
            question_text_raw=question_text_raw.strip(),
            question_text_normalized=routing.question_text_normalized,
            question_language_code=language_code,
            intent_type=routing.intent_type,
            measure_code=routing.measure_code,
            subject_category_code=routing.subject_category_code,
            query_constraints_json=routing.query_constraints_json,
            routing_payload_json=routing.routing_payload_json,
            classifier_version=routing.classifier_version,
            embedding_model_name=routing.embedding_model_name,
            question_embedding=routing.question_embedding,
        )

        self.db.add(question_event)
        await self.db.commit()
        await self.db.refresh(question_event)
        return question_event

    # --------------------------------------------------------
    # Reused answer flow
    # --------------------------------------------------------

    async def _persist_reused_answer_event(
        self,
        *,
        question_event: QuestionEvent,
        reuse_decision: ReuseDecision,
    ) -> AnswerEvent:
        if not reuse_decision.source_answer_event_id:
            raise OrchestratorValidationError(
                "reuse_decision.source_answer_event_id is required for reused answer flow."
            )

        source_answer = await self._get_answer_event_or_raise(reuse_decision.source_answer_event_id)

        answer_event = await self.feedback_service.create_answer_event(
            AnswerEventCreateInput(
                question_event_id=question_event.question_event_id,
                answer_mode=AnswerModeEnum.REUSED_ANSWER,
                answer_text=source_answer.answer_text,
                answer_text_short=source_answer.answer_text_short,
                answer_language_code=source_answer.answer_language_code,
                confidence_score=reuse_decision.confidence_score,
                trust_score_at_generation=source_answer.trust_score_at_generation and float(source_answer.trust_score_at_generation),
                validation_status=ValidationStatusEnum.PASSED,
                deterministic_validation_passed=True,
                semantic_validation_passed=True,
                reuse_allowed=False,
                reused_from_answer_event_id=source_answer.answer_event_id,
                reuse_policy_version="reuse_gate_v1",
                reuse_decision_payload_json=reuse_decision.payload,
                citations_json=source_answer.citations_json or [],
                answer_payload_json={
                    "source": "reuse",
                    "source_answer_event_id": str(source_answer.answer_event_id),
                    "decision_code": reuse_decision.decision_code,
                    "decision_reason": reuse_decision.reason,
                },
                generation_model_name=None,
                generation_prompt_version=None,
                pipeline_version="answer_orchestrator_reuse_v1",
                evidence_items=await self._clone_evidence_items_from_answer(source_answer.answer_event_id),
            )
        )
        return answer_event

    async def _clone_evidence_items_from_answer(
        self,
        source_answer_event_id: UUID,
    ) -> list[EvidenceItemInput]:
        source_answer = await self._get_answer_event_or_raise(source_answer_event_id)
        _ = source_answer  # explicit: ensures source answer exists

        # Lazy import to avoid circular dependency on ORM usage patterns.
        from app.db.models.feedback import AnswerEvidenceItem

        stmt: Select[Any] = (
            select(AnswerEvidenceItem)
            .where(AnswerEvidenceItem.answer_event_id == source_answer_event_id)
            .order_by(AnswerEvidenceItem.evidence_order.asc())
        )
        result = await self.db.execute(stmt)
        source_items = list(result.scalars().all())

        cloned: list[EvidenceItemInput] = []
        for item in source_items:
            cloned.append(
                EvidenceItemInput(
                    evidence_item_type=item.evidence_item_type,
                    role_code=item.role_code or "supporting_evidence",
                    citation_json=item.citation_json or {},
                    document_id=item.document_id,
                    block_id=item.block_id,
                    table_id=item.table_id,
                    table_row_id=item.table_row_id,
                    legal_fact_id=item.legal_fact_id,
                    document_file_hash=item.document_file_hash,
                    document_content_hash=item.document_content_hash,
                )
            )
        return cloned

    # --------------------------------------------------------
    # Full generation flow
    # --------------------------------------------------------

    async def _run_full_generation(
        self,
        *,
        payload: UserQuestionInput,
        question_event: QuestionEvent,
        routing: QuestionRoutingResult,
        session: ConversationSession,
        reuse_decision: ReuseDecision,
    ) -> GenerationResult:
        runtime_input = RuntimeAnswerInput(
            session_id=session.session_id,
            question_event_id=question_event.question_event_id,
            channel_code=payload.channel_code,
            question_text_raw=question_event.question_text_raw,
            question_text_normalized=routing.question_text_normalized,
            language_code=payload.language_code,
            intent_type=routing.intent_type,
            measure_code=routing.measure_code,
            subject_category_code=routing.subject_category_code,
            routing_payload_json=routing.routing_payload_json,
            query_constraints_json=routing.query_constraints_json,
            request_metadata_json={
                **payload.request_metadata_json,
                "reuse_gate_result": {
                    "should_reuse": reuse_decision.should_reuse,
                    "decision_code": reuse_decision.decision_code,
                    "reason": reuse_decision.reason,
                    "payload": reuse_decision.payload,
                },
            },
            query_terms=[
                question_event.question_text_raw,
                routing.question_text_normalized,
                *(
                    [routing.measure_code]
                    if routing.measure_code
                    else []
                ),
                *(
                    [routing.subject_category_code]
                    if routing.subject_category_code
                    else []
                ),
            ],
        )

        runtime_result = await self.runtime_answer_service.build_answer(runtime_input)
        return runtime_result.generation_result

    async def _persist_generated_answer_event(
        self,
        *,
        question_event: QuestionEvent,
        generation_result: GenerationResult,
    ) -> AnswerEvent:
        answer_event = await self.feedback_service.create_answer_event(
            AnswerEventCreateInput(
                question_event_id=question_event.question_event_id,
                answer_mode=generation_result.answer_mode,
                answer_text=generation_result.answer_text,
                answer_text_short=generation_result.answer_text_short,
                answer_language_code="ru",
                confidence_score=generation_result.confidence_score,
                trust_score_at_generation=generation_result.trust_score_at_generation,
                validation_status=generation_result.validation_status,
                deterministic_validation_passed=generation_result.deterministic_validation_passed,
                semantic_validation_passed=generation_result.semantic_validation_passed,
                reuse_allowed=generation_result.reuse_allowed,
                reused_from_answer_event_id=None,
                reuse_policy_version=generation_result.reuse_policy_version,
                reuse_decision_payload_json=generation_result.reuse_decision_payload_json,
                citations_json=generation_result.citations_json,
                answer_payload_json=generation_result.answer_payload_json,
                generation_model_name=generation_result.generation_model_name,
                generation_prompt_version=generation_result.generation_prompt_version,
                pipeline_version=generation_result.pipeline_version,
                evidence_items=generation_result.evidence_items,
            )
        )
        return answer_event

    # --------------------------------------------------------
    # Outgoing payload
    # --------------------------------------------------------

    def _build_outgoing_payload(
        self,
        *,
        session: ConversationSession,
        question_event: QuestionEvent,
        answer_event: AnswerEvent,
        should_request_feedback: bool,
        reuse_decision: ReuseDecision,
    ) -> OutgoingAnswerPayload:
        was_reused = answer_event.answer_mode == AnswerModeEnum.REUSED_ANSWER

        feedback_payload_json = {
            "enabled": should_request_feedback,
            "answer_event_id": str(answer_event.answer_event_id),
            "question_event_id": str(question_event.question_event_id),
            "session_id": str(session.session_id),
            "type": "rating_1_to_5",
            "allow_comment": True,
        }

        delivery_payload_json = {
            "citations": answer_event.citations_json or [],
            "messenger_format": {
                "supports_inline_buttons": True,
                "supports_markdown": True,
            },
        }

        debug_payload_json = {
            "reuse_gate": {
                "should_reuse": reuse_decision.should_reuse,
                "decision_code": reuse_decision.decision_code,
                "confidence_score": reuse_decision.confidence_score,
            },
            "answer_event": {
                "validation_status": str(answer_event.validation_status),
                "answer_mode": str(answer_event.answer_mode),
            },
        }

        return OutgoingAnswerPayload(
            answer_event_id=answer_event.answer_event_id,
            session_id=session.session_id,
            question_event_id=question_event.question_event_id,
            answer_text=answer_event.answer_text,
            answer_text_short=answer_event.answer_text_short,
            citations_json=answer_event.citations_json or [],
            answer_mode=answer_event.answer_mode,
            was_reused=was_reused,
            reused_from_answer_event_id=answer_event.reused_from_answer_event_id,
            should_request_feedback=should_request_feedback,
            feedback_payload_json=feedback_payload_json,
            delivery_payload_json=delivery_payload_json,
            debug_payload_json=debug_payload_json,
        )

    # --------------------------------------------------------
    # Validation / lookups
    # --------------------------------------------------------

    def _validate_user_question_input(
        self,
        payload: UserQuestionInput,
    ) -> None:
        if not payload.external_session_id or not payload.external_session_id.strip():
            raise OrchestratorValidationError("external_session_id must not be empty.")

        if not payload.question_text or not payload.question_text.strip():
            raise OrchestratorValidationError("question_text must not be empty.")

        if len(payload.question_text) > 10000:
            raise OrchestratorValidationError("question_text is too long.")

    async def _get_channel_or_raise(
        self,
        channel_code: ChannelTypeEnum,
    ) -> Channel:
        stmt: Select[Any] = select(Channel).where(Channel.channel_code == channel_code)
        result = await self.db.execute(stmt)
        channel = result.scalar_one_or_none()
        if channel is None:
            raise OrchestratorNotFoundError(f"Channel not found: {channel_code}")
        return channel

    async def _get_answer_event_or_raise(
        self,
        answer_event_id: UUID,
    ) -> AnswerEvent:
        stmt: Select[Any] = select(AnswerEvent).where(
            AnswerEvent.answer_event_id == answer_event_id
        )
        result = await self.db.execute(stmt)
        answer_event = result.scalar_one_or_none()
        if answer_event is None:
            raise OrchestratorNotFoundError(f"AnswerEvent not found: {answer_event_id}")
        return answer_event

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)