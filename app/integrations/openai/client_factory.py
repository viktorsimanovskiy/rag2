# ============================================================
# File: app/integrations/openai/client_factory.py
# Purpose:
#   Centralized factory for creating OpenAI-compatible clients.
#
# Responsibilities:
#   - create AsyncOpenAI client from application settings
#   - enforce one standard construction path across the project
#   - support OpenAI proxy/provider via base_url
#
# Non-responsibilities:
#   - does NOT call models directly
#   - does NOT contain prompt logic
#   - does NOT contain embedding logic
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

from app.config.settings import OpenAISettings


# ============================================================
# Exceptions
# ============================================================

class OpenAIClientFactoryError(Exception):
    """Base client factory error."""


class OpenAIClientSettingsError(OpenAIClientFactoryError):
    """Raised when OpenAI settings are invalid."""


# ============================================================
# DTO
# ============================================================

@dataclass(slots=True, frozen=True)
class OpenAIClientFactoryConfig:
    """
    Optional factory-level overrides.

    Normally the project should rely on settings.py,
    but this DTO leaves room for controlled overrides in tests.
    """
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_seconds: Optional[int] = None
    max_retries: Optional[int] = None
    organization: Optional[str] = None
    project: Optional[str] = None


# ============================================================
# Factory
# ============================================================

class OpenAIClientFactory:
    """
    Centralized AsyncOpenAI client factory.

    Main design goal:
    the entire project should construct OpenAI clients in one place only.
    """

    def __init__(
        self,
        settings: OpenAISettings,
        *,
        overrides: Optional[OpenAIClientFactoryConfig] = None,
    ) -> None:
        self.settings = settings
        self.overrides = overrides or OpenAIClientFactoryConfig()

    def create_async_client(self) -> AsyncOpenAI:
        """
        Build AsyncOpenAI client using validated settings.
        """
        api_key = self.overrides.api_key or self.settings.api_key
        base_url = self.overrides.base_url or self.settings.base_url
        timeout_seconds = (
            self.overrides.timeout_seconds
            if self.overrides.timeout_seconds is not None
            else self.settings.timeout_seconds
        )
        max_retries = (
            self.overrides.max_retries
            if self.overrides.max_retries is not None
            else self.settings.max_retries
        )
        organization = (
            self.overrides.organization
            if self.overrides.organization is not None
            else self.settings.organization
        )
        project = (
            self.overrides.project
            if self.overrides.project is not None
            else self.settings.project
        )

        self._validate(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout_seconds,
            "max_retries": max_retries,
        }

        if organization:
            client_kwargs["organization"] = organization

        if project:
            client_kwargs["project"] = project

        return AsyncOpenAI(**client_kwargs)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def _validate(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: int,
        max_retries: int,
    ) -> None:
        if not api_key or not api_key.strip():
            raise OpenAIClientSettingsError("OpenAI api_key must not be empty.")

        if not base_url or not base_url.strip():
            raise OpenAIClientSettingsError("OpenAI base_url must not be empty.")

        if timeout_seconds < 1:
            raise OpenAIClientSettingsError("OpenAI timeout_seconds must be >= 1.")

        if max_retries < 0:
            raise OpenAIClientSettingsError("OpenAI max_retries must be >= 0.")