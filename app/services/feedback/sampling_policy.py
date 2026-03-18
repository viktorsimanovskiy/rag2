# ============================================================
# File: app/services/feedback/sampling_policy.py
# Purpose:
#   Deterministic feedback sampling policy for answer quality collection.
#
# Responsibilities:
#   - decide whether to request feedback for a newly delivered answer
#   - enforce cooldown between feedback requests
#   - support "every N-th answer" strategy
#   - prioritize specific answer modes / intents / channels
#   - remain deterministic and production-friendly
#
# Design principles:
#   - no random sampling in core policy
#   - conservative user experience
#   - configurable without rewriting orchestrator
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import Select, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.feedback import AnswerEvent, AnswerFeedback, ConversationSession, QuestionEvent
from app.db.models.enums import AnswerModeEnum, ChannelTypeEnum, QuestionIntentEnum

logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================

class SamplingPolicyError(Exception):
    """Base error for sampling policy."""


class SamplingPolicyValidationError(SamplingPolicyError):
    """Raised when input validation fails."""


# ============================================================
# DTOs
# ============================================================

@dataclass(slots=True)
class SamplingDecisionInput:
    """
    Input payload for feedback request decision.
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
class SamplingPolicyConfig:
    """
    Runtime configuration for feedback request policy.

    Main controls:
    - every_n_answers:
        Request feedback on each N-th eligible answer in a session.
    - cooldown_minutes:
        Minimum time between shown feedback prompts within one session.
    - max_feedback_requests_per_session:
        Hard upper bound to avoid over-surveying.
    - always_request_for_reused_answers:
        Whether reused answers should be sampled more aggressively.
    - eligible_answer_modes:
        Which answer modes are generally eligible for feedback prompts.
    - eligible_channels:
        Which channels can show feedback prompts.
    - priority_intents:
        Intents for which feedback is especially valuable.
    - min_confidence_to_sample:
        Optional lower confidence threshold to avoid surveying obviously weak drafts.
    """
    every_n_answers: int = 5
    cooldown_minutes: int = 30
    max_feedback_requests_per_session: int = 3

    always_request_for_reused_answers: bool = True
    always_request_for_priority_intents: bool = False

    eligible_answer_modes: set[AnswerModeEnum] = field(
        default_factory=lambda: {
            AnswerModeEnum.DIRECT_STRUCTURED,
            AnswerModeEnum.GROUNDED_NARRATIVE,
            AnswerModeEnum.REUSED_ANSWER,
        }
    )

    eligible_channels: set[ChannelTypeEnum] = field(
        default_factory=lambda: {
            ChannelTypeEnum.TELEGRAM,
            ChannelTypeEnum.MAX,
            ChannelTypeEnum.WEB,
        }
    )

    priority_intents: set[QuestionIntentEnum] = field(
        default_factory=lambda: {
            QuestionIntentEnum.DOCUMENTS_QUESTION,
            QuestionIntentEnum.DEADLINE_QUESTION,
            QuestionIntentEnum.PROCEDURE_QUESTION,
        }
    )

    excluded_intents: set[QuestionIntentEnum] = field(
        default_factory=lambda: {
            QuestionIntentEnum.AMBIGUOUS_QUESTION,
        }
    )

    min_confidence_to_sample: Optional[float] = 0.40
    policy_version: str = "sampling_policy_v1"


@dataclass(slots=True)
class SamplingDecisionDetails:
    """
    Detailed explanation of the sampling decision.
    """
    should_request_feedback: bool
    decision_code: str
    policy_version: str
    details: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Service
# ============================================================

class SamplingPolicy:
    """
    Deterministic and configurable feedback request policy.
    """

    def __init__(
        self,
        db: AsyncSession,
        *,
        config: Optional[SamplingPolicyConfig] = None,
    ) -> None:
        self.db = db
        self.config = config or SamplingPolicyConfig()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    async def should_request_feedback(
        self,
        payload: SamplingDecisionInput,
    ) -> bool:
        """
        Compatibility method used by orchestrator.
        Returns only bool.
        """
        decision = await self.evaluate(payload)
        return decision.should_request_feedback

    async def evaluate(
        self,
        payload: SamplingDecisionInput,
    ) -> SamplingDecisionDetails:
        """
        Full sampling decision with reasoning.

        Decision flow:
        1. Validate input
        2. Check channel eligibility
        3. Check answer mode eligibility
        4. Check excluded intents
        5. Check confidence threshold
        6. Check session-level limits
        7. Check cooldown
        8. Allow priority paths
        9. Fall back to every-N deterministic sampling
        """
        self._validate_input(payload)

        if payload.channel_code not in self.config.eligible_channels:
            return SamplingDecisionDetails(
                should_request_feedback=False,
                decision_code="channel_not_eligible",
                policy_version=self.config.policy_version,
                details={"channel_code": str(payload.channel_code)},
            )

        if payload.answer_mode not in self.config.eligible_answer_modes:
            return SamplingDecisionDetails(
                should_request_feedback=False,
                decision_code="answer_mode_not_eligible",
                policy_version=self.config.policy_version,
                details={"answer_mode": str(payload.answer_mode)},
            )

        if payload.intent_type in self.config.excluded_intents:
            return SamplingDecisionDetails(
                should_request_feedback=False,
                decision_code="intent_excluded",
                policy_version=self.config.policy_version,
                details={"intent_type": str(payload.intent_type)},
            )

        if (
            self.config.min_confidence_to_sample is not None
            and payload.confidence_score is not None
            and payload.confidence_score < self.config.min_confidence_to_sample
        ):
            return SamplingDecisionDetails(
                should_request_feedback=False,
                decision_code="confidence_below_threshold",
                policy_version=self.config.policy_version,
                details={
                    "confidence_score": payload.confidence_score,
                    "min_confidence_to_sample": self.config.min_confidence_to_sample,
                },
            )

        session_stats = await self._get_session_sampling_stats(payload.session_id)

        if session_stats["feedback_requests_count"] >= self.config.max_feedback_requests_per_session:
            return SamplingDecisionDetails(
                should_request_feedback=False,
                decision_code="session_feedback_limit_reached",
                policy_version=self.config.policy_version,
                details={
                    "feedback_requests_count": session_stats["feedback_requests_count"],
                    "max_feedback_requests_per_session": self.config.max_feedback_requests_per_session,
                },
            )

        cooldown_ok, cooldown_details = self._check_cooldown(session_stats["last_feedback_requested_at"])
        if not cooldown_ok:
            return SamplingDecisionDetails(
                should_request_feedback=False,
                decision_code="cooldown_active",
                policy_version=self.config.policy_version,
                details=cooldown_details,
            )

        # Priority path 1: reused answers
        if self.config.always_request_for_reused_answers and payload.answer_mode == AnswerModeEnum.REUSED_ANSWER:
            return SamplingDecisionDetails(
                should_request_feedback=True,
                decision_code="priority_reused_answer",
                policy_version=self.config.policy_version,
                details={
                    "answer_mode": str(payload.answer_mode),
                    "feedback_requests_count": session_stats["feedback_requests_count"],
                },
            )

        # Priority path 2: selected intents
        if self.config.always_request_for_priority_intents and payload.intent_type in self.config.priority_intents:
            return SamplingDecisionDetails(
                should_request_feedback=True,
                decision_code="priority_intent",
                policy_version=self.config.policy_version,
                details={
                    "intent_type": str(payload.intent_type),
                    "feedback_requests_count": session_stats["feedback_requests_count"],
                },
            )

        # Deterministic every-N rule over eligible answers in the session.
        eligible_answer_sequence_number = session_stats["eligible_answers_count"] + 1

        if eligible_answer_sequence_number % self.config.every_n_answers == 0:
            return SamplingDecisionDetails(
                should_request_feedback=True,
                decision_code="every_n_answers_rule",
                policy_version=self.config.policy_version,
                details={
                    "eligible_answer_sequence_number": eligible_answer_sequence_number,
                    "every_n_answers": self.config.every_n_answers,
                },
            )

        return SamplingDecisionDetails(
            should_request_feedback=False,
            decision_code="not_selected_by_sampling_rule",
            policy_version=self.config.policy_version,
            details={
                "eligible_answer_sequence_number": eligible_answer_sequence_number,
                "every_n_answers": self.config.every_n_answers,
                "feedback_requests_count": session_stats["feedback_requests_count"],
            },
        )

    # --------------------------------------------------------
    # Internal logic
    # --------------------------------------------------------

    def _validate_input(self, payload: SamplingDecisionInput) -> None:
        if self.config.every_n_answers < 1:
            raise SamplingPolicyValidationError("every_n_answers must be >= 1.")

        if self.config.cooldown_minutes < 0:
            raise SamplingPolicyValidationError("cooldown_minutes must be >= 0.")

        if self.config.max_feedback_requests_per_session < 1:
            raise SamplingPolicyValidationError("max_feedback_requests_per_session must be >= 1.")

        if payload.confidence_score is not None:
            if not (0.0 <= payload.confidence_score <= 1.0):
                raise SamplingPolicyValidationError("confidence_score must be between 0 and 1.")

    async def _get_session_sampling_stats(
        self,
        session_id: UUID,
    ) -> dict[str, Any]:
        """
        Collect session-level counters needed for decision making.

        Returns:
        - eligible_answers_count:
            Count of previously created eligible answer events in the session.
        - feedback_requests_count:
            Count of answers in this session that already had feedback recorded.
            For early production this is a pragmatic proxy.
        - last_feedback_requested_at:
            Timestamp of most recent feedback record in session.
            Later this can be replaced by a dedicated delivery-log table.
        """
        await self._ensure_session_exists(session_id)

        # Count previous eligible answer events in this session.
        eligible_answers_stmt: Select[Any] = (
            select(func.count(AnswerEvent.answer_event_id))
            .join(QuestionEvent, QuestionEvent.question_event_id == AnswerEvent.question_event_id)
            .where(QuestionEvent.session_id == session_id)
            .where(AnswerEvent.answer_mode.in_(tuple(self.config.eligible_answer_modes)))
        )
        eligible_answers_result = await self.db.execute(eligible_answers_stmt)
        eligible_answers_count = int(eligible_answers_result.scalar() or 0)

        # Feedback count already received in the session.
        feedback_count_stmt: Select[Any] = (
            select(func.count(AnswerFeedback.feedback_id))
            .where(AnswerFeedback.session_id == session_id)
        )
        feedback_count_result = await self.db.execute(feedback_count_stmt)
        feedback_requests_count = int(feedback_count_result.scalar() or 0)

        # Last feedback timestamp in the session.
        last_feedback_stmt: Select[Any] = (
            select(AnswerFeedback.submitted_at)
            .where(AnswerFeedback.session_id == session_id)
            .order_by(desc(AnswerFeedback.submitted_at))
            .limit(1)
        )
        last_feedback_result = await self.db.execute(last_feedback_stmt)
        last_feedback_requested_at = last_feedback_result.scalar_one_or_none()

        return {
            "eligible_answers_count": eligible_answers_count,
            "feedback_requests_count": feedback_requests_count,
            "last_feedback_requested_at": last_feedback_requested_at,
        }

    def _check_cooldown(
        self,
        last_feedback_requested_at: Optional[datetime],
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check time-based cooldown between feedback prompts.

        Current pragmatic implementation:
        uses latest feedback submission timestamp as proxy.
        Better future implementation:
        use dedicated feedback_prompt_events / delivery log.
        """
        if last_feedback_requested_at is None:
            return True, {
                "last_feedback_requested_at": None,
                "cooldown_minutes": self.config.cooldown_minutes,
            }

        now = self._utcnow()
        cooldown_delta = timedelta(minutes=self.config.cooldown_minutes)
        next_allowed_at = last_feedback_requested_at + cooldown_delta

        if now >= next_allowed_at:
            return True, {
                "last_feedback_requested_at": last_feedback_requested_at.isoformat(),
                "next_allowed_at": next_allowed_at.isoformat(),
                "cooldown_minutes": self.config.cooldown_minutes,
            }

        return False, {
            "last_feedback_requested_at": last_feedback_requested_at.isoformat(),
            "next_allowed_at": next_allowed_at.isoformat(),
            "cooldown_minutes": self.config.cooldown_minutes,
            "seconds_remaining": int((next_allowed_at - now).total_seconds()),
        }

    async def _ensure_session_exists(
        self,
        session_id: UUID,
    ) -> None:
        stmt: Select[Any] = select(ConversationSession.session_id).where(
            ConversationSession.session_id == session_id
        )
        result = await self.db.execute(stmt)
        session_exists = result.scalar_one_or_none()
        if session_exists is None:
            raise SamplingPolicyValidationError(f"ConversationSession not found: {session_id}")

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)