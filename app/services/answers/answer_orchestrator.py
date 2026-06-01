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
import time
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
from app.services.answers.message_guard import MessageGuardResult
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


class MessageGuardProtocol:
    async def check(self, message_text: str, *, channel_code: str | None = None) -> MessageGuardResult:
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

    _CHANNEL_ID_CACHE: dict[str, UUID] = {}

    def __init__(
        self,
        db: AsyncSession,
        *,
        feedback_service: FeedbackService,
        reuse_gate: ReuseGate,
        intent_classifier: IntentClassifierProtocol,
        question_normalizer: QuestionNormalizerProtocol,
        message_guard: Optional[MessageGuardProtocol],
        question_embedding_service: Optional[QuestionEmbeddingProtocol],
        runtime_answer_service: RuntimeAnswerServiceProtocol,
        sampling_policy: SamplingPolicyProtocol,
    ) -> None:
        self.db = db
        self.feedback_service = feedback_service
        self.reuse_gate = reuse_gate
        self.intent_classifier = intent_classifier
        self.question_normalizer = question_normalizer
        self.message_guard = message_guard
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
        timings_started_at = time.perf_counter()
        timings: dict[str, Any] = {
            "version": "second_step_01_message_guard_v1",
        }

        def measure_start() -> float:
            return time.perf_counter()

        def measure_stop(key: str, started_at: float) -> None:
            timings[key] = round(time.perf_counter() - started_at, 6)

        started = measure_start()
        self._validate_user_question_input(payload)
        measure_stop("validate_input_sec", started)

        started = measure_start()
        session = await self._resolve_or_create_session(payload)
        measure_stop("resolve_or_create_session_sec", started)
        session_resolution_timings = getattr(self, "_last_session_resolution_timings", None)
        if isinstance(session_resolution_timings, dict):
            timings["session_resolution_details"] = session_resolution_timings

        started = measure_start()
        guard_result = await self._run_message_guard(payload)
        measure_stop("message_guard_sec", started)

        if guard_result is not None and not guard_result.should_run_rag:
            return await self._handle_message_guard_blocked(
                payload=payload,
                session=session,
                guard_result=guard_result,
                timings=timings,
                timings_started_at=timings_started_at,
                measure_start=measure_start,
                measure_stop=measure_stop,
            )

        started = measure_start()
        routing = await self._build_question_routing(
            payload.question_text,
            request_metadata_json=payload.request_metadata_json,
            guard_result=guard_result,
        )
        measure_stop("build_question_routing_sec", started)

        timings["question_embedding_skipped"] = bool(
            (routing.routing_payload_json or {})
            .get("question_embedding_skipped", {})
            .get("enabled")
        )
        timings["intent_type"] = routing.intent_type.value

        started = measure_start()
        question_event = await self._create_question_event(
            session_id=session.session_id,
            question_text_raw=payload.question_text,
            language_code=payload.language_code,
            routing=routing,
        )
        measure_stop("create_question_event_sec", started)

        if self._should_skip_reuse_for_routing(routing):
            reuse_decision = ReuseDecision(
                should_reuse=False,
                source_answer_event_id=None,
                decision_code="reuse_skipped_for_unstable_answer_type",
                confidence_score=0.0,
                reason="Reuse отключён для вопросов по документам: формат таких ответов активно зависит от текущей версии builder'а.",
                payload={
                    "question_event_id": str(question_event.question_event_id),
                    "intent_type": routing.intent_type.value,
                    "reason_code": "documents_builder_version_sensitive",
                },
            )
            timings["reuse_gate_sec"] = 0.0
            timings["reuse_skipped"] = True
        else:
            started = measure_start()
            reuse_decision = await self.reuse_gate.build_reuse_decision(
                ReuseQueryInput(
                    question_event_id=question_event.question_event_id,
                    similarity_threshold=0.90,
                    max_candidates=20,
                    allow_subject_category_mismatch=False,
                )
            )
            measure_stop("reuse_gate_sec", started)
            timings["reuse_skipped"] = False

        if reuse_decision.should_reuse and reuse_decision.source_answer_event_id:
            started = measure_start()
            answer_event = await self._persist_reused_answer_event(
                question_event=question_event,
                reuse_decision=reuse_decision,
            )
            measure_stop("persist_reused_answer_event_sec", started)
        else:
            started = measure_start()
            generation_result = await self._run_full_generation(
                payload=payload,
                question_event=question_event,
                routing=routing,
                session=session,
                reuse_decision=reuse_decision,
            )
            measure_stop("run_full_generation_sec", started)

            started = measure_start()
            answer_event = await self._persist_generated_answer_event(
                question_event=question_event,
                generation_result=generation_result,
            )
            measure_stop("persist_generated_answer_event_sec", started)

        started = measure_start()
        should_request_feedback = await self.sampling_policy.should_request_feedback(
            SamplingDecisionInput(
                channel_code=payload.channel_code,
                session_id=session.session_id,
                question_event_id=question_event.question_event_id,
                answer_event_id=answer_event.answer_event_id,
                answer_mode=answer_event.answer_mode,
                intent_type=question_event.intent_type,
                confidence_score=float(answer_event.confidence_score) if answer_event.confidence_score is not None else None,
                request_metadata_json=payload.request_metadata_json,
            )
        )
        measure_stop("sampling_policy_sec", started)

        started = measure_start()
        result = self._build_outgoing_payload(
            session=session,
            question_event=question_event,
            answer_event=answer_event,
            should_request_feedback=should_request_feedback,
            reuse_decision=reuse_decision,
        )
        measure_stop("build_outgoing_payload_sec", started)

        timings["total_sec"] = round(time.perf_counter() - timings_started_at, 6)
        known_keys = [
            "validate_input_sec",
            "resolve_or_create_session_sec",
            "message_guard_sec",
            "build_question_routing_sec",
            "create_question_event_sec",
            "reuse_gate_sec",
            "run_full_generation_sec",
            "persist_generated_answer_event_sec",
            "persist_reused_answer_event_sec",
            "sampling_policy_sec",
            "build_outgoing_payload_sec",
        ]
        timings["unaccounted_sec"] = round(
            max(
                timings["total_sec"] - sum(float(timings.get(key) or 0.0) for key in known_keys),
                0.0,
            ),
            6,
        )

        result.debug_payload_json = dict(result.debug_payload_json or {})
        result.debug_payload_json["orchestrator_timings_sec"] = timings

        logger.info(
            "Processed user question",
            extra={
                "session_id": str(session.session_id),
                "question_event_id": str(question_event.question_event_id),
                "answer_event_id": str(answer_event.answer_event_id),
                "answer_mode": str(answer_event.answer_mode),
                "reuse_approved": reuse_decision.should_reuse,
                "feedback_requested": should_request_feedback,
                "orchestrator_total_sec": timings["total_sec"],
                "question_embedding_skipped": timings["question_embedding_skipped"],
            },
        )
        return result

    # --------------------------------------------------------
    # Message guard
    # --------------------------------------------------------

    async def _run_message_guard(
        self,
        payload: UserQuestionInput,
    ) -> Optional[MessageGuardResult]:
        """Run cheap pre-retrieval guard if it is configured."""
        if self.message_guard is None:
            return None
        return await self.message_guard.check(
            payload.question_text,
            channel_code=(
                payload.channel_code.value
                if isinstance(payload.channel_code, ChannelTypeEnum)
                else str(payload.channel_code)
            ),
        )

    async def _handle_message_guard_blocked(
        self,
        *,
        payload: UserQuestionInput,
        session: ConversationSession,
        guard_result: MessageGuardResult,
        timings: dict[str, Any],
        timings_started_at: float,
        measure_start: Any,
        measure_stop: Any,
    ) -> OutgoingAnswerPayload:
        """Persist and return a service answer without retrieval/generation."""
        routing = QuestionRoutingResult(
            question_text_normalized=guard_result.normalized_text or payload.question_text.strip(),
            intent_type=QuestionIntentEnum.OTHER,
            subject_category_code=None,
            classifier_version=guard_result.guard_version,
            embedding_model_name=None,
            routing_payload_json={
                "message_guard": guard_result.to_payload(),
                "routing_mode": "no_retrieval",
                "should_run_rag": False,
            },
            query_constraints_json={
                "message_guard_blocked": True,
                "message_guard_reason_code": guard_result.reason_code,
            },
            question_embedding=None,
        )

        started = measure_start()
        question_event = await self._create_question_event(
            session_id=session.session_id,
            question_text_raw=payload.question_text,
            language_code=payload.language_code,
            routing=routing,
        )
        measure_stop("create_question_event_sec", started)

        reuse_decision = ReuseDecision(
            should_reuse=False,
            source_answer_event_id=None,
            decision_code="message_guard_no_retrieval",
            confidence_score=guard_result.confidence_score,
            reason="MessageGuard остановил сообщение до retrieval.",
            payload={
                "question_event_id": str(question_event.question_event_id),
                "message_guard": guard_result.to_payload(),
            },
        )
        timings["reuse_gate_sec"] = 0.0
        timings["reuse_skipped"] = True
        timings["run_full_generation_sec"] = 0.0

        generation_result = GenerationResult(
            answer_mode=AnswerModeEnum.SAFE_NO_ANSWER,
            answer_text=(guard_result.answer_text or "Сообщение не передано в поиск по нормативным актам."),
            answer_text_short=guard_result.answer_text,
            confidence_score=guard_result.confidence_score,
            trust_score_at_generation=1.0,
            validation_status=ValidationStatusEnum.PASSED,
            deterministic_validation_passed=True,
            semantic_validation_passed=True,
            reuse_allowed=False,
            reuse_policy_version="reuse_gate_v1",
            citations_json=[],
            answer_payload_json={
                "strategy_code": "message_guard_no_retrieval",
                "pipeline_version": "second_step_01_message_guard_v1",
                "message_guard": guard_result.to_payload(),
                "runtime_answer_service_timings": {},
                "runtime_answer_service_runtime_payload": {},
            },
            reuse_decision_payload_json={
                "reuse_allowed": False,
                "reuse_policy_version": "reuse_gate_v1",
                "message_guard": guard_result.to_payload(),
            },
            evidence_items=[],
            generation_model_name=None,
            generation_prompt_version=None,
            pipeline_version="second_step_01_message_guard_v1",
        )

        started = measure_start()
        answer_event = await self._persist_generated_answer_event(
            question_event=question_event,
            generation_result=generation_result,
        )
        measure_stop("persist_generated_answer_event_sec", started)

        timings["sampling_policy_sec"] = 0.0
        started = measure_start()
        result = self._build_outgoing_payload(
            session=session,
            question_event=question_event,
            answer_event=answer_event,
            should_request_feedback=False,
            reuse_decision=reuse_decision,
        )
        measure_stop("build_outgoing_payload_sec", started)

        timings["question_embedding_skipped"] = True
        timings["intent_type"] = QuestionIntentEnum.OTHER.value
        timings["message_guard_blocked"] = True
        timings["message_guard_reason_code"] = guard_result.reason_code
        timings["total_sec"] = round(time.perf_counter() - timings_started_at, 6)
        known_keys = [
            "validate_input_sec",
            "resolve_or_create_session_sec",
            "message_guard_sec",
            "create_question_event_sec",
            "reuse_gate_sec",
            "run_full_generation_sec",
            "persist_generated_answer_event_sec",
            "sampling_policy_sec",
            "build_outgoing_payload_sec",
        ]
        timings["unaccounted_sec"] = round(
            max(
                timings["total_sec"] - sum(float(timings.get(key) or 0.0) for key in known_keys),
                0.0,
            ),
            6,
        )

        result.debug_payload_json = dict(result.debug_payload_json or {})
        result.debug_payload_json["orchestrator_timings_sec"] = timings
        result.debug_payload_json["message_guard"] = guard_result.to_payload()

        logger.info(
            "Message stopped by MessageGuard before retrieval",
            extra={
                "session_id": str(session.session_id),
                "question_event_id": str(question_event.question_event_id),
                "answer_event_id": str(answer_event.answer_event_id),
                "message_guard_reason_code": guard_result.reason_code,
                "orchestrator_total_sec": timings["total_sec"],
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
        """Resolve or create conversation session.

        Hot-path note:
        Previously every existing session was updated, committed and refreshed
        before the actual answer was built. On the VPS this added roughly 5-7
        seconds per Telegram/n8n request. For the live bot path we only need a
        stable session id, so existing sessions now use a read-only fast path.

        New sessions are still committed and refreshed normally.
        """
        total_started_at = time.perf_counter()
        details: dict[str, Any] = {
            "version": "step66_deferred_session_commit_v1",
            "session_created": False,
            "existing_session_fast_path": False,
        }

        started_at = time.perf_counter()
        channel_id = await self._get_channel_id_or_raise(payload.channel_code)
        details["channel_lookup_sec"] = round(time.perf_counter() - started_at, 6)

        started_at = time.perf_counter()
        stmt: Select[Any] = select(ConversationSession).where(
            ConversationSession.channel_id == channel_id,
            ConversationSession.external_session_id == payload.external_session_id,
        )
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()
        details["session_lookup_sec"] = round(time.perf_counter() - started_at, 6)

        if session is not None:
            details["existing_session_fast_path"] = True
            details["commit_refresh_sec"] = 0.0
            details["total_sec"] = round(time.perf_counter() - total_started_at, 6)
            self._last_session_resolution_timings = details
            return session

        started_at = time.perf_counter()
        session = ConversationSession(
            session_id=uuid4(),
            channel_id=channel_id,
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

        # Do not commit/refresh the newly created session here.
        #
        # In the question-bank run every API call uses a fresh external_session_id.
        # On the VPS the old commit+refresh path for a new ConversationSession
        # sometimes took 30-50 seconds, while the actual RAG path stayed below
        # one second. The session_id is generated application-side, so the next
        # step can create QuestionEvent in the same transaction and commit both
        # rows together. This keeps the hot path safe and avoids an unnecessary
        # round trip before the answer is built.
        await self.db.flush()

        details["flush_sec"] = round(time.perf_counter() - started_at, 6)
        details["commit_refresh_sec"] = 0.0
        details["session_created"] = True
        details["session_persist_mode"] = "flush_only_deferred_commit"
        details["total_sec"] = round(time.perf_counter() - total_started_at, 6)
        self._last_session_resolution_timings = details
        return session

    # --------------------------------------------------------
    # Question routing / understanding
    # --------------------------------------------------------

    def _should_skip_reuse_for_intent(self, intent_type: QuestionIntentEnum) -> bool:
        """Return True when answer reuse should be skipped for the intent.

        Document answers are version-sensitive: the rendered text depends on the
        current documents builder, channel wording and full-list/form-detail mode.
        When reuse is skipped, the question embedding is also unnecessary because
        it is only used by ReuseGate in the current runtime path.
        """
        return intent_type == QuestionIntentEnum.DOCUMENTS_QUESTION

    def _should_skip_reuse_for_routing(self, routing: QuestionRoutingResult) -> bool:
        """Skip reuse for answer types whose rendering is version-sensitive."""
        return self._should_skip_reuse_for_intent(routing.intent_type)

    async def _build_question_routing(
        self,
        question_text: str,
        *,
        request_metadata_json: Optional[dict[str, Any]] = None,
        guard_result: Optional[MessageGuardResult] = None,
    ) -> QuestionRoutingResult:
        """
        Build routing metadata for a user question.

        request_metadata_json may contain forced_intent_type from the HTTP API/n8n.
        Even when intent is forced, the classifier is still executed because it may
        add useful query constraints, for example requires_service_discovery.
        """
        request_metadata_json = dict(request_metadata_json or {})

        normalized_text = await self.question_normalizer.normalize(question_text)
        classification = await self.intent_classifier.classify(normalized_text)

        intent_value = classification.get("intent_type", QuestionIntentEnum.OTHER)
        intent_type = (
            intent_value
            if isinstance(intent_value, QuestionIntentEnum)
            else QuestionIntentEnum(intent_value)
        )

        routing_payload_json = dict(classification.get("routing_payload_json") or {})
        query_constraints_json = dict(classification.get("query_constraints_json") or {})

        if guard_result is not None:
            routing_payload_json["message_guard"] = guard_result.to_payload()

        forced_intent_value = request_metadata_json.get("forced_intent_type")
        if forced_intent_value:
            forced_intent = (
                forced_intent_value
                if isinstance(forced_intent_value, QuestionIntentEnum)
                else QuestionIntentEnum(str(forced_intent_value))
            )
            routing_payload_json["forced_intent"] = {
                "enabled": True,
                "source": request_metadata_json.get("forced_intent_source") or "request_metadata",
                "original_intent_type": intent_type.value,
                "forced_intent_type": forced_intent.value,
            }
            intent_type = forced_intent

        question_embedding: Optional[list[float]] = None
        embedding_model_name: Optional[str] = None

        should_skip_reuse = self._should_skip_reuse_for_intent(intent_type)
        if should_skip_reuse:
            routing_payload_json["question_embedding_skipped"] = {
                "enabled": True,
                "reason": "reuse_skipped_for_intent",
                "intent_type": intent_type.value,
            }
        elif self.question_embedding_service is not None:
            question_embedding = await self.question_embedding_service.embed(normalized_text)
            embedding_model_name = classification.get("embedding_model_name")

        return QuestionRoutingResult(
            question_text_normalized=normalized_text,
            intent_type=intent_type,
            subject_category_code=classification.get("subject_category_code"),
            classifier_version=classification.get("classifier_version"),
            embedding_model_name=embedding_model_name,
            routing_payload_json=routing_payload_json,
            query_constraints_json=query_constraints_json,
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
                    [routing.subject_category_code]
                    if routing.subject_category_code
                    else []
                ),
            ],
        )

        runtime_result = await self.runtime_answer_service.build_answer(runtime_input)

        # Keep server-side runtime timings in the persisted answer payload.
        # This is safe diagnostic metadata, not reasoning text. It helps
        # distinguish slow service resolution, retrieval, generation and
        # persistence when the API is called from n8n/Telegram.
        generation_result = runtime_result.generation_result
        answer_payload_json = dict(generation_result.answer_payload_json or {})
        runtime_payload_json = dict(runtime_result.runtime_payload_json or {})
        answer_payload_json["runtime_answer_service_timings"] = dict(
            runtime_payload_json.get("timings_sec") or {}
        )
        answer_payload_json["runtime_answer_service_runtime_payload"] = runtime_payload_json
        generation_result.answer_payload_json = answer_payload_json

        return generation_result

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

    async def _get_channel_id_or_raise(
        self,
        channel_code: ChannelTypeEnum,
    ) -> UUID:
        cache_key = channel_code.value if isinstance(channel_code, ChannelTypeEnum) else str(channel_code)
        cached = self._CHANNEL_ID_CACHE.get(cache_key)
        if cached is not None:
            return cached

        stmt: Select[Any] = select(Channel.channel_id).where(Channel.channel_code == channel_code)
        result = await self.db.execute(stmt)
        channel_id = result.scalar_one_or_none()
        if channel_id is None:
            raise OrchestratorNotFoundError(f"Channel not found: {channel_code}")

        self._CHANNEL_ID_CACHE[cache_key] = channel_id
        return channel_id

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