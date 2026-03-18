# ============================================================
# File: app/config/settings.py
# Purpose:
#   Centralized application settings loaded from environment.
#
# Responsibilities:
#   - read runtime configuration from environment variables
#   - validate required settings
#   - provide typed access to DB / OpenAI / Telegram / logging config
#
# Important:
#   - no business logic here
#   - no SDK client creation here
#   - no DB/session creation here
# ============================================================
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
from dataclasses import dataclass


# ============================================================
# Exceptions
# ============================================================

class SettingsError(Exception):
    """Base settings error."""


class MissingRequiredSettingError(SettingsError):
    """Raised when a required environment variable is missing."""


class InvalidSettingError(SettingsError):
    """Raised when an environment variable has invalid value."""


# ============================================================
# Helpers
# ============================================================

def _get_env(
    name: str,
    *,
    default: str | None = None,
    required: bool = False,
) -> str:
    value = os.getenv(name, default)

    if required and (value is None or value.strip() == ""):
        raise MissingRequiredSettingError(
            f"Required environment variable is missing: {name}"
        )

    if value is None:
        return ""

    return value.strip()


def _get_bool_env(
    name: str,
    *,
    default: bool = False,
) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()

    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise InvalidSettingError(
        f"Environment variable {name} must be a boolean-like value, got: {raw}"
    )


def _get_int_env(
    name: str,
    *,
    default: int,
    min_value: int | None = None,
) -> int:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw.strip())
        except Exception as exc:
            raise InvalidSettingError(
                f"Environment variable {name} must be an integer, got: {raw}"
            ) from exc

    if min_value is not None and value < min_value:
        raise InvalidSettingError(
            f"Environment variable {name} must be >= {min_value}, got: {value}"
        )

    return value


def _normalize_url(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        return normalized
    return normalized.rstrip("/")


# ============================================================
# Settings models
# ============================================================

@dataclass(slots=True, frozen=True)
class DatabaseSettings:
    """
    Database settings.

    Для Supabase здесь обычно будет обычный PostgreSQL DSN вида:
    postgresql+asyncpg://USER:PASSWORD@HOST:PORT/postgres
    """
    url: str
    sql_echo: bool
    pool_pre_ping: bool


@dataclass(slots=True, frozen=True)
class OpenAISettings:
    """
    OpenAI-compatible settings.

    Важно:
    base_url обязателен, потому что у тебя OpenAI идет через посредника.
    """
    api_key: str
    base_url: str
    timeout_seconds: int
    max_retries: int
    organization: str | None = None
    project: str | None = None


@dataclass(slots=True, frozen=True)
class TelegramSettings:
    bot_token: str
    enabled: bool
    polling_timeout_seconds: int


@dataclass(slots=True, frozen=True)
class LoggingSettings:
    level: str


@dataclass(slots=True, frozen=True)
class AppSettings:
    environment: str
    debug: bool
    database: DatabaseSettings
    openai: OpenAISettings
    telegram: TelegramSettings
    logging: LoggingSettings


# ============================================================
# Public API
# ============================================================

def load_settings() -> AppSettings:
    """
    Load all application settings from environment variables.

    Required:
    - APP_DATABASE_URL
    - APP_OPENAI_API_KEY
    - APP_OPENAI_BASE_URL

    Optional:
    - APP_ENV
    - APP_DEBUG
    - APP_SQL_ECHO
    - APP_DB_POOL_PRE_PING
    - APP_OPENAI_TIMEOUT_SECONDS
    - APP_OPENAI_MAX_RETRIES
    - APP_OPENAI_ORGANIZATION
    - APP_OPENAI_PROJECT
    - APP_TELEGRAM_ENABLED
    - APP_TELEGRAM_BOT_TOKEN
    - APP_TELEGRAM_POLLING_TIMEOUT_SECONDS
    - APP_LOG_LEVEL
    """
    environment = _get_env("APP_ENV", default="dev")
    debug = _get_bool_env("APP_DEBUG", default=False)

    database = DatabaseSettings(
        url=_get_env("APP_DATABASE_URL", required=True),
        sql_echo=_get_bool_env("APP_SQL_ECHO", default=False),
        pool_pre_ping=_get_bool_env("APP_DB_POOL_PRE_PING", default=True),
    )

    openai = OpenAISettings(
        api_key=_get_env("APP_OPENAI_API_KEY", required=True),
        base_url=_normalize_url(
            _get_env("APP_OPENAI_BASE_URL", required=True)
        ),
        timeout_seconds=_get_int_env(
            "APP_OPENAI_TIMEOUT_SECONDS",
            default=60,
            min_value=1,
        ),
        max_retries=_get_int_env(
            "APP_OPENAI_MAX_RETRIES",
            default=3,
            min_value=0,
        ),
        organization=_get_env("APP_OPENAI_ORGANIZATION", default="") or None,
        project=_get_env("APP_OPENAI_PROJECT", default="") or None,
    )

    telegram_enabled = _get_bool_env("APP_TELEGRAM_ENABLED", default=False)
    telegram_bot_token = _get_env(
        "APP_TELEGRAM_BOT_TOKEN",
        required=telegram_enabled,
        default="",
    )

    telegram = TelegramSettings(
        bot_token=telegram_bot_token,
        enabled=telegram_enabled,
        polling_timeout_seconds=_get_int_env(
            "APP_TELEGRAM_POLLING_TIMEOUT_SECONDS",
            default=30,
            min_value=1,
        ),
    )

    logging_settings = LoggingSettings(
        level=_get_env("APP_LOG_LEVEL", default="INFO").upper(),
    )

    return AppSettings(
        environment=environment,
        debug=debug,
        database=database,
        openai=openai,
        telegram=telegram,
        logging=logging_settings,
    )