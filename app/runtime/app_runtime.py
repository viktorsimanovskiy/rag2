# ============================================================
# File: app/runtime/app_runtime.py
# Purpose:
#   Application runtime bootstrap for the current RAG project.
#
# Responsibilities:
#   - manage application lifecycle
#   - use centralized DB session manager
#   - provide per-request ServiceFactory instances
#   - expose ready-to-use channel handlers
#
# Important:
#   - framework-agnostic
#   - does NOT start aiogram / FastAPI itself
#   - does NOT duplicate DB engine/session logic
# ============================================================

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.bootstrap.service_factory import ServiceFactory, ServiceFactoryConfig
from app.channels.telegram_bot import TelegramBotHandler
from app.config.settings import DatabaseSettings
from app.db.session import DatabaseSessionManager
from app.services.answers.answer_orchestrator import (
    IntentClassifierProtocol,
    QuestionEmbeddingProtocol,
    QuestionNormalizerProtocol,
)

logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================

class AppRuntimeError(Exception):
    """Base runtime error."""


class AppRuntimeNotStartedError(AppRuntimeError):
    """Raised when runtime resources are used before startup."""


# ============================================================
# Config
# ============================================================

@dataclass(slots=True)
class AppRuntimeConfig:
    """
    Runtime bootstrap configuration.
    """
    database: DatabaseSettings

    intent_classifier: Optional[IntentClassifierProtocol] = None
    question_normalizer: Optional[QuestionNormalizerProtocol] = None
    question_embedding_service: Optional[QuestionEmbeddingProtocol] = None

    service_factory_config: Optional[ServiceFactoryConfig] = None


# ============================================================
# No-op defaults
# ============================================================

class DefaultQuestionNormalizer(QuestionNormalizerProtocol):
    async def normalize(self, question_text: str) -> str:
        return " ".join(question_text.strip().split())


class DefaultIntentClassifier(IntentClassifierProtocol):
    async def classify(self, question_text: str) -> dict:
        from app.db.models.enums import QuestionIntentEnum

        return {
            "intent_type": QuestionIntentEnum.OTHER,
            "measure_code": None,
            "subject_category_code": None,
            "classifier_version": "default_intent_classifier_v1",
            "routing_payload_json": {
                "source": "default_classifier",
                "note": "fallback bootstrap classifier",
            },
            "query_constraints_json": {},
        }


# ============================================================
# Runtime
# ============================================================

class AppRuntime:
    """
    Application runtime bootstrap.

    Lifecycle:
        runtime = AppRuntime(config)
        await runtime.startup()

        async with runtime.session_scope() as session:
            factory = runtime.build_service_factory(session)
            ...

        await runtime.shutdown()
    """

    def __init__(self, config: AppRuntimeConfig) -> None:
        self.config = config

        self._db_manager: Optional[DatabaseSessionManager] = None
        self._is_started: bool = False

        self._intent_classifier = (
            config.intent_classifier
            if config.intent_classifier is not None
            else DefaultIntentClassifier()
        )
        self._question_normalizer = (
            config.question_normalizer
            if config.question_normalizer is not None
            else DefaultQuestionNormalizer()
        )
        self._question_embedding_service = config.question_embedding_service
        self._service_factory_config = config.service_factory_config or ServiceFactoryConfig()

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    async def startup(self) -> None:
        """
        Initialize runtime resources.
        """
        if self._is_started:
            logger.info("AppRuntime.startup() called, but runtime is already started")
            return

        self._db_manager = DatabaseSessionManager(self.config.database)
        self._db_manager.initialize()

        await self._db_manager.check_connection()

        self._is_started = True

        logger.info("AppRuntime started")

    async def shutdown(self) -> None:
        """
        Dispose runtime resources.
        """
        if not self._is_started:
            logger.info("AppRuntime.shutdown() called, but runtime is not started")
            return

        if self._db_manager is not None:
            await self._db_manager.dispose()

        self._db_manager = None
        self._is_started = False

        logger.info("AppRuntime stopped")

    # --------------------------------------------------------
    # Session handling
    # --------------------------------------------------------

    @asynccontextmanager
    async def session_scope(self) -> AsyncIterator[AsyncSession]:
        """
        Delegate request session scope to DatabaseSessionManager.
        """
        db_manager = self._require_db_manager()

        async with db_manager.session_scope() as session:
            yield session

    # --------------------------------------------------------
    # Factory / handlers
    # --------------------------------------------------------

    def build_service_factory(self, session: AsyncSession) -> ServiceFactory:
        """
        Build per-request ServiceFactory.
        """
        self._ensure_started()

        return ServiceFactory(
            session,
            intent_classifier=self._intent_classifier,
            question_normalizer=self._question_normalizer,
            question_embedding_service=self._question_embedding_service,
            config=self._service_factory_config,
        )

    def build_telegram_bot_handler(self, session: AsyncSession) -> TelegramBotHandler:
        """
        Build Telegram handler for one request scope.
        """
        factory = self.build_service_factory(session)
        return TelegramBotHandler(service_factory=factory)

    # --------------------------------------------------------
    # Convenience accessors
    # --------------------------------------------------------

    @property
    def is_started(self) -> bool:
        return self._is_started

    def get_db_manager(self) -> DatabaseSessionManager:
        return self._require_db_manager()

    def get_engine(self) -> AsyncEngine:
        return self._require_db_manager().engine

    def get_session_maker(self) -> async_sessionmaker[AsyncSession]:
        return self._require_db_manager().session_factory

    # --------------------------------------------------------
    # Internal guards
    # --------------------------------------------------------

    def _ensure_started(self) -> None:
        if not self._is_started:
            raise AppRuntimeNotStartedError(
                "AppRuntime is not started. Call await runtime.startup() first."
            )

    def _require_db_manager(self) -> DatabaseSessionManager:
        self._ensure_started()

        if self._db_manager is None:
            raise AppRuntimeNotStartedError(
                "DatabaseSessionManager is not initialized."
            )

        return self._db_manager