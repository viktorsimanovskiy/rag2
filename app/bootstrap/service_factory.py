# ============================================================
# File: app/bootstrap/service_factory.py
# Purpose:
#   Composition root for the current RAG project state.
#
# Responsibilities:
#   - centralize service wiring
#   - construct services using real current constructor signatures
#   - avoid scattered dependency assembly across handlers/tests
#
# Important:
#   This factory is intentionally aligned with the CURRENT archive code,
#   not with a future refactored target state.
#
# Assumption:
#   feedback_service has already been moved to:
#   app/services/feedback/feedback_service.py
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.answers.answer_orchestrator import (
    AnswerOrchestrator,
    IntentClassifierProtocol,
    QuestionEmbeddingProtocol,
    QuestionNormalizerProtocol,
)
from app.services.answers.runtime_answer_service import RuntimeAnswerService
from app.services.channels.messenger_response_builder import MessengerResponseBuilder
from app.services.feedback.feedback_service import FeedbackService
from app.services.feedback.sampling_policy import SamplingPolicy, SamplingPolicyConfig
from app.services.generation.generation_pipeline import (
    DeterministicAnswerValidatorProtocol,
    GenerationPipeline,
    SemanticAnswerValidatorProtocol,
)
from app.services.retrieval.retrieval_orchestrator import (
    RetrievalOrchestrator,
    RetrievalRerankerProtocol,
)
from app.services.reuse.reuse_gate import ReuseGate
from app.adapters.telegram.telegram_message_adapter import TelegramMessageAdapter


# ============================================================
# Optional external dependencies / providers
# ============================================================

class SupportsNoop(Protocol):
    pass


@dataclass(slots=True)
class ServiceFactoryConfig:
    """
    Optional runtime config for service assembly.

    Only what is actually needed now is included.
    """
    retrieval_reranker: Optional[RetrievalRerankerProtocol] = None
    deterministic_validator: Optional[DeterministicAnswerValidatorProtocol] = None
    semantic_validator: Optional[SemanticAnswerValidatorProtocol] = None
    sampling_policy_config: Optional[SamplingPolicyConfig] = None


# ============================================================
# Factory
# ============================================================

class ServiceFactory:
    """
    Composition root for the current project state.

    This factory does NOT invent missing implementations.
    It wires only what the current codebase already expects.

    Current assembly strategy:
    - FeedbackService
    - SamplingPolicy
    - ReuseGate
    - RetrievalOrchestrator
    - GenerationPipeline
    - RuntimeAnswerService
    - AnswerOrchestrator
    - MessengerResponseBuilder
    - TelegramMessageAdapter

    Note:
    Current AnswerOrchestrator is wired through RuntimeAnswerService for the
    full runtime answer path. This keeps retrieval + generation assembly in a
    dedicated runtime service and leaves AnswerOrchestrator focused on
    session/question/reuse/persistence coordination.
    """

    def __init__(
        self,
        db: AsyncSession,
        *,
        intent_classifier: IntentClassifierProtocol,
        question_normalizer: QuestionNormalizerProtocol,
        question_embedding_service: Optional[QuestionEmbeddingProtocol] = None,
        config: Optional[ServiceFactoryConfig] = None,
    ) -> None:
        self.db = db
        self.intent_classifier = intent_classifier
        self.question_normalizer = question_normalizer
        self.question_embedding_service = question_embedding_service
        self.config = config or ServiceFactoryConfig()

        self._feedback_service: Optional[FeedbackService] = None
        self._sampling_policy: Optional[SamplingPolicy] = None
        self._reuse_gate: Optional[ReuseGate] = None
        self._retrieval_orchestrator: Optional[RetrievalOrchestrator] = None
        self._generation_pipeline: Optional[GenerationPipeline] = None
        self._runtime_answer_service: Optional[RuntimeAnswerService] = None
        self._answer_orchestrator: Optional[AnswerOrchestrator] = None
        self._messenger_response_builder: Optional[MessengerResponseBuilder] = None
        self._telegram_message_adapter: Optional[TelegramMessageAdapter] = None

    # --------------------------------------------------------
    # Core services
    # --------------------------------------------------------

    def get_feedback_service(self) -> FeedbackService:
        if self._feedback_service is None:
            self._feedback_service = FeedbackService(self.db)
        return self._feedback_service

    def get_sampling_policy(self) -> SamplingPolicy:
        if self._sampling_policy is None:
            self._sampling_policy = SamplingPolicy(
                self.db,
                config=self.config.sampling_policy_config,
            )
        return self._sampling_policy

    def get_reuse_gate(self) -> ReuseGate:
        if self._reuse_gate is None:
            self._reuse_gate = ReuseGate(self.db)
        return self._reuse_gate

    def get_retrieval_orchestrator(self) -> RetrievalOrchestrator:
        if self._retrieval_orchestrator is None:
            self._retrieval_orchestrator = RetrievalOrchestrator(
                self.db,
                reranker=self.config.retrieval_reranker,
            )
        return self._retrieval_orchestrator

    def get_generation_pipeline(self) -> GenerationPipeline:
        if self._generation_pipeline is None:
            self._generation_pipeline = GenerationPipeline(
                self.db,
                deterministic_validator=self.config.deterministic_validator,
                semantic_validator=self.config.semantic_validator,
            )
        return self._generation_pipeline

    def get_runtime_answer_service(self) -> RuntimeAnswerService:
        if self._runtime_answer_service is None:
            self._runtime_answer_service = RuntimeAnswerService(
                retrieval_orchestrator=self.get_retrieval_orchestrator(),
                generation_pipeline=self.get_generation_pipeline(),
            )
        return self._runtime_answer_service

    # --------------------------------------------------------
    # Application services
    # --------------------------------------------------------

    def get_answer_orchestrator(self) -> AnswerOrchestrator:
        """
        Wire AnswerOrchestrator through RuntimeAnswerService so that the real
        runtime path becomes:

            AnswerOrchestrator -> RuntimeAnswerService -> Retrieval + Generation
        """
        if self._answer_orchestrator is None:
            self._answer_orchestrator = AnswerOrchestrator(
                self.db,
                feedback_service=self.get_feedback_service(),
                reuse_gate=self.get_reuse_gate(),
                intent_classifier=self.intent_classifier,
                question_normalizer=self.question_normalizer,
                question_embedding_service=self.question_embedding_service,
                runtime_answer_service=self.get_runtime_answer_service(),
                sampling_policy=self.get_sampling_policy(),
            )
        return self._answer_orchestrator

    def get_messenger_response_builder(self) -> MessengerResponseBuilder:
        if self._messenger_response_builder is None:
            self._messenger_response_builder = MessengerResponseBuilder()
        return self._messenger_response_builder

    def get_telegram_message_adapter(self) -> TelegramMessageAdapter:
        if self._telegram_message_adapter is None:
            self._telegram_message_adapter = TelegramMessageAdapter()
        return self._telegram_message_adapter

    # --------------------------------------------------------
    # Convenience bundle methods
    # --------------------------------------------------------

    def build_runtime_bundle(self) -> dict[str, object]:
        """
        Useful for handlers / entrypoints / tests.
        """
        return {
            "feedback_service": self.get_feedback_service(),
            "sampling_policy": self.get_sampling_policy(),
            "reuse_gate": self.get_reuse_gate(),
            "retrieval_orchestrator": self.get_retrieval_orchestrator(),
            "generation_pipeline": self.get_generation_pipeline(),
            "runtime_answer_service": self.get_runtime_answer_service(),
            "answer_orchestrator": self.get_answer_orchestrator(),
            "messenger_response_builder": self.get_messenger_response_builder(),
            "telegram_message_adapter": self.get_telegram_message_adapter(),
        }