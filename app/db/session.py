# ============================================================
# File: app/db/session.py
# Purpose:
#   Centralized async DB session manager for the current RAG project.
#
# Responsibilities:
#   - create and own AsyncEngine
#   - create and own async_sessionmaker
#   - provide request-scoped AsyncSession via session_scope()
#   - expose health check and graceful shutdown
#
# Important:
#   - this file must stay infrastructure-only
#   - no ORM models here
#   - no business logic here
# ============================================================

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config.settings import DatabaseSettings

logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================

class DatabaseSessionManagerError(Exception):
    """Base DB session manager error."""


class DatabaseSessionManagerNotInitializedError(DatabaseSessionManagerError):
    """Raised when DB manager is used before initialize()."""


class DatabaseConnectionCheckError(DatabaseSessionManagerError):
    """Raised when DB connection health check fails."""


# ============================================================
# Session manager
# ============================================================

class DatabaseSessionManager:
    """
    Centralized async DB session manager.

    Expected lifecycle:
        manager = DatabaseSessionManager(settings)
        manager.initialize()
        await manager.check_connection()

        async with manager.session_scope() as session:
            ...

        await manager.dispose()

    Design notes:
    - initialize() is intentionally sync to match current AppRuntime usage
    - session_scope() commits on success, rolls back on error
    - expire_on_commit=False is used for predictable service-layer behavior
    """

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def initialize(self) -> None:
        """
        Create engine and session factory.

        Safe to call only once per manager instance.
        """
        if self._engine is not None or self._session_factory is not None:
            logger.info("DatabaseSessionManager.initialize() called, but manager is already initialized")
            return

        self._engine = create_async_engine(
            self._settings.url,
            echo=self._settings.sql_echo,
            pool_pre_ping=self._settings.pool_pre_ping,
            future=True,
        )

        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            autoflush=False,
            expire_on_commit=False,
        )

        logger.info("DatabaseSessionManager initialized")

    async def dispose(self) -> None:
        """
        Dispose engine and drop references to session factory.
        """
        if self._engine is not None:
            await self._engine.dispose()
            logger.info("Database engine disposed")

        self._engine = None
        self._session_factory = None

    # --------------------------------------------------------
    # Health check
    # --------------------------------------------------------

    async def check_connection(self) -> None:
        """
        Verify that DB connection is alive.

        Raises:
            DatabaseSessionManagerNotInitializedError
            DatabaseConnectionCheckError
        """
        engine = self.engine

        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            raise DatabaseConnectionCheckError(
                "Failed to establish database connection."
            ) from exc

        logger.info("Database connection check passed")

    # --------------------------------------------------------
    # Session scope
    # --------------------------------------------------------

    @asynccontextmanager
    async def session_scope(self) -> AsyncIterator[AsyncSession]:
        """
        Provide request-scoped AsyncSession.

        Behavior:
        - opens one session
        - yields session to caller
        - commits on success
        - rolls back on failure
        - always closes session
        """
        session_factory = self.session_factory
        session = session_factory()

        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception("Database session rolled back due to exception")
            raise
        finally:
            await session.close()

    # --------------------------------------------------------
    # Explicit session creation
    # --------------------------------------------------------

    def create_session(self) -> AsyncSession:
        """
        Create raw AsyncSession without context manager.

        Use sparingly. Prefer session_scope() for request handling.
        """
        return self.session_factory()

    # --------------------------------------------------------
    # Public accessors
    # --------------------------------------------------------

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise DatabaseSessionManagerNotInitializedError(
                "DatabaseSessionManager is not initialized. Call initialize() first."
            )
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            raise DatabaseSessionManagerNotInitializedError(
                "DatabaseSessionManager is not initialized. Call initialize() first."
            )
        return self._session_factory

    @property
    def is_initialized(self) -> bool:
        return self._engine is not None and self._session_factory is not None