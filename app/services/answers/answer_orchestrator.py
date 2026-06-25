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
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 4c04853102b91a0c6e1fdd3692b3fb98688e2646
from app.services.generation.llm_answer_composer import (
    LLMAnswerComposerInput,
    LLMAnswerComposerResult,
    input_from_generation_result,
)
<<<<<<< HEAD
=======
=======
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
>>>>>>> 4c04853102b91a0c6e1fdd3692b3fb98688e2646
from app.services.answers.message_guard import MessageGuardResult
from app.services.answers.message_understanding import MessageUnderstandingResult
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


class MessageUnderstandingProtocol:
    async def understand(
        self,
        question_text: str,
        *,
        deterministic_classification: Optional[dict[str, Any]] = None,
        channel_code: Optional[str] = None,
    ) -> MessageUnderstandingResult:
        raise NotImplementedError


<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 4c04853102b91a0c6e1fdd3692b3fb98688e2646
class LLMAnswerComposerProtocol:
    async def compose(self, payload: LLMAnswerComposerInput) -> LLMAnswerComposerResult:
        raise NotImplementedError


<<<<<<< HEAD
=======
=======
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
>>>>>>> 4c04853102b91a0c6e1fdd3692b3fb98688e2646
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
class ResolvedConversationSession:
    """Lightweight session reference used on the hot path.

    For existing sessions the orchestrator only needs session_id. Selecting the
    full ConversationSession ORM object may trigger relationship loaders and
    become slow when the session has many question_events.
    """
    session_id: UUID


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
        message_understanding_service: Optional[MessageUnderstandingProtocol],
<<<<<<< HEAD
        llm_answer_composer_service: Optional[LLMAnswerComposerProtocol],
=======
<<<<<<< HEAD
        llm_answer_composer_service: Optional[LLMAnswerComposerProtocol],
=======
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
>>>>>>> 4c04853102b91a0c6e1fdd3692b3fb98688e2646
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
        self.message_understanding_service = message_understanding_service
<<<<<<< HEAD
        self.llm_answer_composer_service = llm_answer_composer_service
=======
<<<<<<< HEAD
        self.llm_answer_composer_service = llm_answer_composer_service
=======
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
>>>>>>> 4c04853102b91a0c6e1fdd3692b3fb98688e2646
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
            "version": "second_step_17_assist_call_policy_narrow_broad_help_v1",
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
    ) -> ConversationSession | ResolvedConversationSession:
        """Resolve or create conversation session.

        Hot-path note:
        Existing sessions are looked up by session_id only. Selecting the full
        ConversationSession ORM object can trigger relationship loaders
        (question_events / feedback_items) and becomes slow when a test or a
        Telegram chat accumulates many messages in one session. The rest of the
        orchestrator only needs session_id, so a lightweight reference is safer
        and faster for the hot path.

        New sessions still use the ORM object and are flushed only; the next
        QuestionEvent commit persists both rows together.
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
        stmt: Select[Any] = select(ConversationSession.session_id).where(
            ConversationSession.channel_id == channel_id,
            ConversationSession.external_session_id == payload.external_session_id,
        )
        result = await self.db.execute(stmt)
        existing_session_id = result.scalar_one_or_none()
        details["session_lookup_sec"] = round(time.perf_counter() - started_at, 6)

        if existing_session_id is not None:
            details["existing_session_fast_path"] = True
            details["existing_session_lookup_mode"] = "session_id_only"
            details["commit_refresh_sec"] = 0.0
            details["total_sec"] = round(time.perf_counter() - total_started_at, 6)
            self._last_session_resolution_timings = details
            return ResolvedConversationSession(session_id=existing_session_id)

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

    async def _run_message_understanding(
        self,
        normalized_text: str,
        *,
        deterministic_classification: dict[str, Any],
    ) -> Optional[MessageUnderstandingResult]:
        """Run optional LLM understanding layer.

        The service is intentionally optional. If it is not configured or fails,
        the deterministic classifier remains the source of routing.
        """
        if self.message_understanding_service is None:
            return None

        return await self.message_understanding_service.understand(
            normalized_text,
            deterministic_classification=deterministic_classification,
        )

    def _apply_message_understanding_to_routing(
        self,
        *,
        understanding_result: MessageUnderstandingResult,
        rule_intent_type: QuestionIntentEnum,
        routing_payload_json: dict[str, Any],
        query_constraints_json: dict[str, Any],
    ) -> QuestionIntentEnum:
        """Merge optional LLM understanding into routing metadata.

        Modes:
        - shadow: store diagnostics only;
        - assist: apply intent only when the rule route is weak/conflicting,
          and use neutral semantic slots as resolver hints;
        - enforce: apply if the model is confident enough.

        The LLM never becomes a legal evidence source. It may only route the
        question and add neutral search hints such as topic, applicant facts,
        user_needs and territory.
        """
        payload = understanding_result.to_payload()
        routing_payload_json["message_understanding"] = payload

        mode = (understanding_result.mode or "shadow").strip().lower()
        mapped_intent = understanding_result.mapped_intent_type
        min_confidence = _message_understanding_min_confidence(payload)

        intent_apply_reason = "shadow_mode"
        should_apply_intent = False
        should_apply_hints = False
        hints_apply_reason = "shadow_mode"

        provider_ok = understanding_result.provider_status == "ok"
        supported_ok = bool(understanding_result.is_supported_domain)
        confidence_ok = understanding_result.confidence >= min_confidence
        medium_confidence_ok = understanding_result.confidence >= 0.65
        hint_confidence_ok = understanding_result.confidence >= 0.55
        mapped_ok = mapped_intent is not None and mapped_intent != QuestionIntentEnum.OTHER
        rule_is_weak_or_conflicting = self._is_weak_or_conflicting_rule_route(
            rule_intent_type=rule_intent_type,
            routing_payload_json=routing_payload_json,
        )

        if not provider_ok:
            intent_apply_reason = "provider_not_ok"
            hints_apply_reason = "provider_not_ok"
        elif not supported_ok:
            intent_apply_reason = "unsupported_domain"
            hints_apply_reason = "unsupported_domain"
        elif mapped_intent is None:
            intent_apply_reason = "unknown_intent"
            hints_apply_reason = "unknown_intent"
        elif mode == "enforce":
            if confidence_ok:
                should_apply_intent = mapped_intent is not None
                should_apply_hints = self._has_message_understanding_expansion_terms(understanding_result)
                intent_apply_reason = "enforce_mode_high_confidence"
                hints_apply_reason = "enforce_mode_high_confidence"
            else:
                intent_apply_reason = "low_confidence"
                hints_apply_reason = "low_confidence"
        elif mode == "assist":
            if confidence_ok:
                should_apply_intent = self._should_apply_message_understanding_assist(
                    rule_intent_type=rule_intent_type,
                    understanding_result=understanding_result,
                    routing_payload_json=routing_payload_json,
                )
                intent_apply_reason = (
                    "assist_mode_rule_is_weak_or_conflicting"
                    if should_apply_intent
                    else "assist_mode_rule_is_strong"
                )
            elif medium_confidence_ok and rule_is_weak_or_conflicting and mapped_ok:
                # second_step_17_assist_call_policy_narrow_broad_help_v1
                # If the deterministic route is essentially empty/ambiguous,
                # medium-confidence LLM intent is useful enough to route the
                # question to the correct deterministic builder. This still does
                # not create evidence and is limited to weak rule cases.
                should_apply_intent = self._should_apply_message_understanding_assist_medium(
                    rule_intent_type=rule_intent_type,
                    understanding_result=understanding_result,
                    routing_payload_json=routing_payload_json,
                )
                intent_apply_reason = (
                    "assist_mode_medium_confidence_weak_rule_intent"
                    if should_apply_intent
                    else "assist_mode_medium_confidence_intent_not_safe"
                )
            else:
                intent_apply_reason = "low_confidence"

            should_apply_hints = bool(
                mapped_ok
                and self._has_message_understanding_expansion_terms(understanding_result)
                and (confidence_ok or (hint_confidence_ok and rule_is_weak_or_conflicting))
            )
            hints_apply_reason = (
                "assist_mode_high_confidence_resolver_hints"
                if should_apply_hints and confidence_ok
                else "assist_mode_medium_confidence_weak_rule_resolver_hints"
                if should_apply_hints
                else "assist_mode_no_safe_hints"
            )
        else:
            intent_apply_reason = "shadow_mode"
            hints_apply_reason = "shadow_mode"

        routing_payload_json["message_understanding_application"] = {
            "version": "second_step_17_assist_call_policy_narrow_broad_help_v1",
            "mode": mode,
            "applied": bool(should_apply_intent or should_apply_hints),
            "intent_applied": should_apply_intent,
            "intent_reason": intent_apply_reason,
            "resolver_hints_applied": should_apply_hints,
            "resolver_hints_reason": hints_apply_reason,
            "rule_intent_type": rule_intent_type.value,
            "llm_intent_type": mapped_intent.value if mapped_intent is not None else None,
            "confidence": understanding_result.confidence,
            "min_confidence": min_confidence,
            "medium_intent_confidence": 0.65,
            "medium_hint_confidence": 0.55,
            "rule_is_weak_or_conflicting": rule_is_weak_or_conflicting,
        }

        expansion_terms = self._message_understanding_expansion_terms(understanding_result)
        if should_apply_hints and expansion_terms:
            query_constraints_json["message_understanding_hints_applied"] = True
            query_constraints_json["resolver_query_expansion_terms"] = expansion_terms

        if not should_apply_intent or mapped_intent is None:
            return rule_intent_type

        query_constraints_json["message_understanding_applied"] = True
        query_constraints_json["message_understanding_intent"] = mapped_intent.value

        if self._should_apply_message_understanding_service_discovery(
            understanding_result=understanding_result,
            rule_intent_type=rule_intent_type,
            routing_payload_json=routing_payload_json,
        ):
            query_constraints_json["requires_service_discovery"] = True
            query_constraints_json["avoid_single_service_resolution"] = True
            query_constraints_json["routing_mode"] = "service_discovery"

        if understanding_result.needs_clarification and understanding_result.clarification_question:
            query_constraints_json["needs_clarification"] = True
            query_constraints_json["clarification_question"] = understanding_result.clarification_question

        return mapped_intent

    @staticmethod
    def _is_weak_or_conflicting_rule_route(
        *,
        rule_intent_type: QuestionIntentEnum,
        routing_payload_json: dict[str, Any],
    ) -> bool:
        if rule_intent_type in {QuestionIntentEnum.OTHER, QuestionIntentEnum.AMBIGUOUS_QUESTION}:
            return True

        rule_confidence = _to_float(
            (routing_payload_json or {}).get("confidence"),
            default=0.0,
        )
        if rule_confidence and rule_confidence < 0.72:
            return True

        query_constraints = (routing_payload_json or {}).get("query_constraints_json") or {}
        if query_constraints.get("requires_service_discovery"):
            return True

        chosen_rules = (routing_payload_json or {}).get("chosen_rules") or []
        if isinstance(chosen_rules, list) and len(chosen_rules) > 1:
            return True

        return False

    @staticmethod
    def _should_apply_message_understanding_assist(
        *,
        rule_intent_type: QuestionIntentEnum,
        understanding_result: MessageUnderstandingResult,
        routing_payload_json: dict[str, Any],
    ) -> bool:
        mapped_intent = understanding_result.mapped_intent_type
        if mapped_intent is None or mapped_intent == QuestionIntentEnum.OTHER:
            return False

        if rule_intent_type in {QuestionIntentEnum.OTHER, QuestionIntentEnum.AMBIGUOUS_QUESTION}:
            return True

        rule_confidence = _to_float(
            (routing_payload_json or {}).get("confidence"),
            default=0.0,
        )
        if rule_confidence and rule_confidence < 0.72:
            return True

        # In assist mode the LLM should change the intent only when it
        # genuinely disagrees with a weak/conflicting rule result. If it returns
        # the same intent as the rule layer, keep the rule intent and use only
        # neutral resolver hints. This makes diagnostics truthful: intent_applied
        # means that routing actually changed.
        if mapped_intent == rule_intent_type:
            return False

        # Generic precedence rule: an explicit document intent may override a
        # broad/weak rule result, but not a strong non-document route.
        if mapped_intent == QuestionIntentEnum.DOCUMENTS_QUESTION:
            return True

        return False

    @staticmethod
    def _should_apply_message_understanding_assist_medium(
        *,
        rule_intent_type: QuestionIntentEnum,
        understanding_result: MessageUnderstandingResult,
        routing_payload_json: dict[str, Any],
    ) -> bool:
        """Allow medium-confidence LLM intent only for weak rule routes.

        This is intentionally narrower than the high-confidence assist path.
        It is meant for cases where the rule layer returned OTHER/AMBIGUOUS or
        another weak result, while the model extracted a supported domain intent
        plus semantic slots. The model still does not produce legal content.
        """
        mapped_intent = understanding_result.mapped_intent_type
        if mapped_intent is None or mapped_intent == QuestionIntentEnum.OTHER:
            return False
        if mapped_intent == rule_intent_type:
            return False
        if not AnswerOrchestrator._is_weak_or_conflicting_rule_route(
            rule_intent_type=rule_intent_type,
            routing_payload_json=routing_payload_json,
        ):
            return False
        if not AnswerOrchestrator._has_message_understanding_expansion_terms(understanding_result):
            return False
        return mapped_intent in {
            QuestionIntentEnum.DOCUMENTS_QUESTION,
            QuestionIntentEnum.ELIGIBILITY_QUESTION,
            QuestionIntentEnum.DEADLINE_QUESTION,
            QuestionIntentEnum.PAYMENT_TIMING_QUESTION,
            QuestionIntentEnum.AMOUNT_QUESTION,
            QuestionIntentEnum.REJECTION_QUESTION,
            QuestionIntentEnum.PROCEDURE_QUESTION,
            QuestionIntentEnum.APPEAL_QUESTION,
            QuestionIntentEnum.FORM_QUESTION,
            QuestionIntentEnum.MIXED_QUESTION,
        }

    @staticmethod
    def _should_apply_message_understanding_service_discovery(
        *,
        understanding_result: MessageUnderstandingResult,
        rule_intent_type: QuestionIntentEnum,
        routing_payload_json: dict[str, Any],
    ) -> bool:
        """Decide whether LLM may force service_discovery.

        LLMs often mark broad eligibility as needs_service_discovery=true. That
        is correct for questions like "what am I entitled to?". It is not safe
        to force discovery when the user already gave a specific semantic bundle
        such as applicant category + territory + concrete need. In those cases
        resolver hints are better: let deterministic service_resolver choose one
        concrete service.
        """
        if not understanding_result.needs_service_discovery:
            return False

        constraints = (routing_payload_json or {}).get("query_constraints_json") or {}
        if constraints.get("requires_service_discovery"):
            return True
        if str(constraints.get("routing_mode") or "").strip().lower() == "service_discovery":
            return True

        if understanding_result.service_hint:
            return False
        if understanding_result.territory and understanding_result.user_needs:
            return False
        if understanding_result.requested_channel and understanding_result.user_needs:
            return False

        # A broad weak eligibility route with applicant facts but no concrete
        # need should remain service_discovery: this is the usual "what am I
        # entitled to?" scenario.
        return True


    @staticmethod
    def _is_broad_eligibility_help_question(
        *,
        rule_intent_type: QuestionIntentEnum,
        routing_payload_json: dict[str, Any],
    ) -> bool:
        """True for broad natural help questions where LLM semantic slots help resolver.

        second_step_17_assist_call_policy_narrow_broad_help_v1

        The previous policy was too broad: it called the LLM for virtually every
        eligibility question in the live-question bank, including explicit
        questions like "могу ли подать заявление на ...".  That increased latency
        while adding little value, because such questions already contain enough
        service-specific terms for the deterministic resolver.

        This predicate is intentionally narrower.  It calls the LLM for broad
        real-life help messages where the user describes a situation/need, but
        avoids doing so for explicit application questions that already name the
        measure or service.
        """
        if rule_intent_type != QuestionIntentEnum.ELIGIBILITY_QUESTION:
            return False

        payload = routing_payload_json or {}
        normalized = str(payload.get("normalized_for_classification") or "").lower()
        if not normalized:
            return False

        # Do not call the LLM for explicit deterministic sub-routes.
        blocked_markers = (
            "какие документы",
            "список документов",
            "перечень документов",
            "полный перечень",
            "срок",
            "когда",
            "причин",
            "основан",
            "отказ",
            "обжал",
        )
        if any(marker in normalized for marker in blocked_markers):
            return False

        # Explicit "apply for X" questions usually already contain service terms.
        # Let the deterministic resolver handle them first; LLM fallback should
        # be added later after resolver_result=ambiguous/not_found, not before.
        explicit_application_markers = (
            "могу ли подать заявление",
            "можно ли подать заявление",
            "могу подать заявление",
            "подать заявление на",
            "оформить заявление на",
        )
        if any(marker in normalized for marker in explicit_application_markers):
            return False

        broad_help_markers = (
            "можно ли получить помощь",
            "можно ли получить поддержку",
            "помощь от соцзащит",
            "нужна помощь",
            "помогите",
            "не хватает денег",
            "денег не хватает",
            "денег нет",
            "случилась беда",
            "трудная ситуация",
            "сгорел",
            "нечего есть",
            "дров",
            "собрать детей в школу",
        )
        if any(marker in normalized for marker in broad_help_markers):
            return True

        chosen_rules = payload.get("chosen_rules") or []
        if isinstance(chosen_rules, list) and any(
            str(rule) in {
                "service_discovery_crisis_or_material_help",
                "service_discovery_broad_entitlement",
            }
            for rule in chosen_rules
        ):
            return len(normalized.split()) >= 7

        return False

    def _message_understanding_mode(self) -> str:
        """Return configured message-understanding mode without calling the LLM."""
        service = self.message_understanding_service
        if service is None:
            return "disabled"
        config = getattr(service, "config", None)
        mode = getattr(config, "mode", "shadow")
        return str(mode or "shadow").strip().lower()

    def _should_call_message_understanding(
        self,
        *,
        rule_intent_type: QuestionIntentEnum,
        routing_payload_json: dict[str, Any],
        query_constraints_json: dict[str, Any],
        request_metadata_json: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, str, str]:
        """Decide whether the optional LLM understanding layer should be called.

        The expensive call is useful for diagnostics in shadow mode and for
        weak/conflicting routes in assist mode. Strong deterministic routes,
        especially ordinary document/deadline/refusal questions, should not pay
        the LLM latency on every request.
        """
        if self.message_understanding_service is None:
            return False, "disabled", "service_not_configured"

        mode = self._message_understanding_mode()
        request_metadata_json = dict(request_metadata_json or {})

        forced = request_metadata_json.get("force_message_understanding")
        if forced is True or str(forced).strip().lower() in {"1", "true", "yes", "да"}:
            return True, mode, "forced_by_request_metadata"

        if mode == "shadow":
            return True, mode, "shadow_mode_observe_all"

        if mode == "enforce":
            return True, mode, "enforce_mode_evaluate_all"

        if mode != "assist":
            return False, mode, "unsupported_mode"

        if self._is_weak_or_conflicting_rule_route(
            rule_intent_type=rule_intent_type,
            routing_payload_json=routing_payload_json,
        ):
            return True, mode, "assist_mode_weak_or_conflicting_rule_route"

        if query_constraints_json.get("requires_service_discovery"):
            return True, mode, "assist_mode_service_discovery_route"

        if str(query_constraints_json.get("routing_mode") or "").strip().lower() == "service_discovery":
            return True, mode, "assist_mode_service_discovery_route"

        # second_step_20_post_resolver_llm_policy_v1
        # Do not call the LLM upfront for every broad eligibility/help question.
        # The broad-question detector remains useful as a cheap signal, but the
        # expensive LLM call should be moved closer to the resolver fallback path.
        # This prevents the pattern observed in the 220-question bank where every
        # Qxx_01 triggered the model before we even knew whether the deterministic
        # resolver could handle the question.
        if self._is_broad_eligibility_help_question(
            rule_intent_type=rule_intent_type,
            routing_payload_json=routing_payload_json,
        ):
            return False, mode, "assist_mode_broad_help_deferred_until_after_resolver"

        return False, mode, "assist_mode_strong_rule_route_skipped"

    @staticmethod
    def _has_message_understanding_expansion_terms(
        understanding_result: MessageUnderstandingResult,
    ) -> bool:
        return bool(AnswerOrchestrator._message_understanding_expansion_terms(understanding_result))

    @staticmethod
    def _message_understanding_expansion_terms(
        understanding_result: MessageUnderstandingResult,
    ) -> list[str]:
        result: list[str] = []
        for value in (
            understanding_result.service_hint,
            understanding_result.topic,
            understanding_result.territory,
            understanding_result.requested_channel,
            *list(understanding_result.applicant_facts or []),
            *list(getattr(understanding_result, "user_needs", []) or []),
        ):
            if value is None:
                continue
            text = " ".join(str(value).strip().split())
            if not text:
                continue
            if text.lower() in {item.lower() for item in result}:
                continue
            result.append(text[:160])
        return result[:10]

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

        should_call_understanding, understanding_mode, understanding_call_reason = self._should_call_message_understanding(
            rule_intent_type=intent_type,
            routing_payload_json=routing_payload_json,
            query_constraints_json=query_constraints_json,
            request_metadata_json=request_metadata_json,
        )
        routing_payload_json["message_understanding_call_policy"] = {
            "version": "second_step_20_post_resolver_llm_policy_v1",
            "mode": understanding_mode,
            "should_call": should_call_understanding,
            "reason": understanding_call_reason,
        }

        if should_call_understanding:
            understanding_result = await self._run_message_understanding(
                normalized_text,
                deterministic_classification=classification,
            )
            if understanding_result is not None:
                intent_type = self._apply_message_understanding_to_routing(
                    understanding_result=understanding_result,
                    rule_intent_type=intent_type,
                    routing_payload_json=routing_payload_json,
                    query_constraints_json=query_constraints_json,
                )

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

        # second_step_03_deferred_question_event_commit_v1
        # Do not commit/refresh QuestionEvent here. The following answer_event
        # persistence path commits the whole transaction. On a long-lived test
        # database the old commit+refresh step could take 2-4 seconds even when
        # MessageGuard stopped the request before retrieval. question_event_id is
        # generated application-side, and the same AsyncSession can use the
        # flushed row for reuse/generation checks before the final commit.
        await self.db.flush()
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
        session: ConversationSession | ResolvedConversationSession,
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
                *list((routing.query_constraints_json or {}).get("resolver_query_expansion_terms") or []),
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

<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 4c04853102b91a0c6e1fdd3692b3fb98688e2646
        await self._attach_llm_answer_composer_shadow_diagnostics(
            request_payload=payload,
            generation_result=generation_result,
        )

        return generation_result

    async def _attach_llm_answer_composer_shadow_diagnostics(
        self,
        *,
        request_payload: UserQuestionInput,
        generation_result: GenerationResult,
    ) -> None:
        """Attach optional LLM-composer diagnostics to answer_payload_json.

        Default runtime mode is still shadow-only: the composer may produce
        diagnostics and a proposed rewrite, but the deterministic answer remains
        the user-facing answer.

        second_step_50 adds a tightly limited test-only assist switch:
        request_metadata_json.force_llm_answer_composer_replacement = true.
        It is accepted only for the test_console channel with API debug enabled.
        This lets us validate replacement mechanics without changing Telegram/MAX
        production behaviour.
        """
        answer_payload_json = dict(generation_result.answer_payload_json or {})
        replacement_allowed_precheck, replacement_precheck_reason = (
            self._precheck_llm_answer_composer_test_replacement(request_payload)
        )
        policy = {
            "version": "second_step_52_service_discovery_replacement_guard_v1",
            "enabled": self.llm_answer_composer_service is not None,
            "replacement_enabled": replacement_allowed_precheck,
            "replacement_scope": "test_console_only",
            "replacement_reason": replacement_precheck_reason,
            "reason": "service_configured" if self.llm_answer_composer_service is not None else "service_not_configured",
        }

        answer_payload_json["llm_answer_composer_call_policy"] = policy
        generation_result.answer_payload_json = answer_payload_json

        if self.llm_answer_composer_service is None:
            return

        original_answer_text = generation_result.answer_text
        started_at = time.perf_counter()
        try:
            composer_payload = input_from_generation_result(
                question_text=request_payload.question_text,
                generation_result=generation_result,
            )
            composer_result = await self.llm_answer_composer_service.compose(composer_payload)
        except Exception as exc:  # defensive: composer must never break RAG answer
            logger.warning(
                "Runtime LLM answer composer failed; deterministic answer kept",
                extra={
                    "version": "second_step_52_service_discovery_replacement_guard_v1",
                    "error": repr(exc),
                },
            )
            answer_payload_json = dict(generation_result.answer_payload_json or {})
            answer_payload_json["llm_answer_composer"] = {
                "version": "second_step_52_service_discovery_replacement_guard_v1",
                "enabled": True,
                "mode": "shadow",
                "provider_status": "error",
                "status": "fallback",
                "should_replace_answer": False,
                "runtime_replacement_applied": False,
                "runtime_replacement_suppressed": True,
                "runtime_replacement_reason": "composer_exception",
                "error": repr(exc),
            }
            answer_payload_json["llm_answer_composer_timings_sec"] = {
                "total": round(time.perf_counter() - started_at, 6),
            }
            generation_result.answer_payload_json = answer_payload_json
            return

        replacement_allowed, replacement_reason = self._should_apply_llm_answer_composer_test_replacement(
            request_payload=request_payload,
            composer_result=composer_result,
            composer_payload=composer_payload,
        )

        composed_answer_text = str(composer_result.composed_answer_text or "").strip()
        if replacement_allowed and composed_answer_text:
            generation_result.answer_text = composed_answer_text
            generation_result.answer_text_short = self._build_llm_composed_short_answer(composed_answer_text)

        answer_payload_json = dict(generation_result.answer_payload_json or {})
        composer_payload_json = composer_result.to_payload()
        composer_payload_json["runtime_replacement_applied"] = bool(replacement_allowed and composed_answer_text)
        composer_payload_json["runtime_replacement_suppressed"] = not bool(replacement_allowed and composed_answer_text)
        composer_payload_json["runtime_replacement_reason"] = replacement_reason
        composer_payload_json["runtime_replacement_scope"] = "test_console_only"
        composer_payload_json["original_answer_text"] = original_answer_text
        composer_payload_json["should_replace_answer"] = bool(replacement_allowed and composed_answer_text)
        composer_payload_json["final_answer_text"] = generation_result.answer_text

        answer_payload_json["llm_answer_composer"] = composer_payload_json
        answer_payload_json["llm_answer_composer_timings_sec"] = {
            "total": round(time.perf_counter() - started_at, 6),
        }
        generation_result.answer_payload_json = answer_payload_json

    def _precheck_llm_answer_composer_test_replacement(
        self,
        request_payload: UserQuestionInput,
    ) -> tuple[bool, str]:
        metadata = dict(request_payload.request_metadata_json or {})
        if metadata.get("force_llm_answer_composer_replacement") is not True:
            return False, "not_requested"

        if request_payload.channel_code != ChannelTypeEnum.TEST_CONSOLE:
            return False, "blocked_non_test_channel"

        if metadata.get("api_debug_requested") is not True:
            return False, "blocked_without_api_debug"

        return True, "test_replacement_requested"

    def _should_apply_llm_answer_composer_test_replacement(
        self,
        *,
        request_payload: UserQuestionInput,
        composer_result: LLMAnswerComposerResult,
        composer_payload: LLMAnswerComposerInput,
    ) -> tuple[bool, str]:
        precheck_ok, precheck_reason = self._precheck_llm_answer_composer_test_replacement(request_payload)
        if not precheck_ok:
            return False, precheck_reason

        if composer_result.provider_status != "ok":
            return False, f"blocked_provider_status:{composer_result.provider_status or 'empty'}"

        if composer_result.status != "ok":
            return False, f"blocked_composer_status:{composer_result.status or 'empty'}"

        if composer_result.grounding_violations:
            return False, "blocked_grounding_violations"

        if self._is_service_discovery_llm_composer_payload(composer_payload):
            return False, "blocked_service_discovery_answer"

        if self._is_broad_or_clarification_answer(composer_payload.deterministic_answer_text):
            return False, "blocked_broad_or_clarification_answer"

        if not str(composer_result.composed_answer_text or "").strip():
            return False, "blocked_empty_composed_answer"

        return True, "applied_test_only_replacement"

    @staticmethod
    def _is_service_discovery_llm_composer_payload(payload: LLMAnswerComposerInput) -> bool:
        service_resolution = dict(payload.service_resolution or {})
        resolution_status = str(service_resolution.get("resolution_status") or "").strip().lower()
        return resolution_status == "service_discovery"

    @staticmethod
    def _is_broad_or_clarification_answer(answer_text: str) -> bool:
        normalized = " ".join(str(answer_text or "").replace("\u00a0", " ").lower().split())
        if not normalized:
            return False
        markers = (
            "нашёл несколько мер",
            "найдено несколько мер",
            "несколько мер социальной поддержки",
            "могут быть релевантны",
            "это не означает, что право",
            "нельзя надёжно выбрать одну",
            "нужно уточнить",
            "что нужно уточнить",
            "после уточнения можно проверять",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _build_llm_composed_short_answer(answer_text: str, *, limit: int = 700) -> str:
        normalized = " ".join(str(answer_text or "").split()).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip() + "…"

<<<<<<< HEAD
=======
=======
        return generation_result
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
>>>>>>> 4c04853102b91a0c6e1fdd3692b3fb98688e2646

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

        routing_payload_json = dict(question_event.routing_payload_json or {})
        query_constraints_json = dict(question_event.query_constraints_json or {})

        debug_payload_json = {
            "version": "second_step_20_post_resolver_llm_policy_v1",
            "reuse_gate": {
                "should_reuse": reuse_decision.should_reuse,
                "decision_code": reuse_decision.decision_code,
                "confidence_score": reuse_decision.confidence_score,
            },
            "answer_event": {
                "validation_status": str(answer_event.validation_status),
                "answer_mode": str(answer_event.answer_mode),
            },
            "question_routing": {
                "intent_type": str(question_event.intent_type),
                "subject_category_code": question_event.subject_category_code,
                "classifier_version": question_event.classifier_version,
                "routing_payload_json": routing_payload_json,
                "query_constraints_json": query_constraints_json,
            },
        }

        # second_step_05_message_understanding_smoke_and_debug_v1
        # Expose LLM-understanding diagnostics in API debug output. The data was
        # already stored in QuestionEvent.routing_payload_json, but the HTTP
        # response did not surface it, which made shadow-mode checks opaque.
        if "message_understanding_call_policy" in routing_payload_json:
            debug_payload_json["message_understanding_call_policy"] = routing_payload_json.get("message_understanding_call_policy")
        if "message_understanding" in routing_payload_json:
            debug_payload_json["message_understanding"] = routing_payload_json.get("message_understanding")
        if "message_understanding_application" in routing_payload_json:
            debug_payload_json["message_understanding_application"] = routing_payload_json.get("message_understanding_application")

<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 4c04853102b91a0c6e1fdd3692b3fb98688e2646
        answer_payload_json = dict(answer_event.answer_payload_json or {})
        if "llm_answer_composer_call_policy" in answer_payload_json:
            debug_payload_json["llm_answer_composer_call_policy"] = answer_payload_json.get("llm_answer_composer_call_policy")
        if "llm_answer_composer" in answer_payload_json:
            debug_payload_json["llm_answer_composer"] = answer_payload_json.get("llm_answer_composer")

<<<<<<< HEAD
=======
=======
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
>>>>>>> 4c04853102b91a0c6e1fdd3692b3fb98688e2646
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

# ============================================================
# Message understanding helpers
# ============================================================

def _message_understanding_min_confidence(payload: dict[str, Any]) -> float:
    raw = (payload or {}).get("min_confidence_to_apply")
    value = _to_float(raw, default=0.72)
    return max(0.0, min(value, 1.0))


def _to_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default
