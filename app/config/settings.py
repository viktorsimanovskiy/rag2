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




def _get_float_env(
    name: str,
    *,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw.strip())
        except Exception as exc:
            raise InvalidSettingError(
                f"Environment variable {name} must be a float, got: {raw}"
            ) from exc

    if min_value is not None and value < min_value:
        raise InvalidSettingError(
            f"Environment variable {name} must be >= {min_value}, got: {value}"
        )
    if max_value is not None and value > max_value:
        raise InvalidSettingError(
            f"Environment variable {name} must be <= {max_value}, got: {value}"
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
class MessageUnderstandingSettings:
    enabled: bool
    mode: str
    model_name: str
    temperature: float
    max_output_tokens: int
    min_confidence_to_apply: float
    request_timeout_seconds: int


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
    message_understanding: MessageUnderstandingSettings
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
    - APP_MESSAGE_UNDERSTANDING_ENABLED
    - APP_MESSAGE_UNDERSTANDING_MODE
    - APP_MESSAGE_UNDERSTANDING_MODEL
    - APP_MESSAGE_UNDERSTANDING_TEMPERATURE
    - APP_MESSAGE_UNDERSTANDING_MAX_OUTPUT_TOKENS
    - APP_MESSAGE_UNDERSTANDING_MIN_CONFIDENCE
    - APP_MESSAGE_UNDERSTANDING_TIMEOUT_SECONDS
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

    message_understanding_mode = _get_env(
        "APP_MESSAGE_UNDERSTANDING_MODE",
        default="shadow",
    ).lower()
    if message_understanding_mode not in {"shadow", "assist", "enforce"}:
        raise InvalidSettingError(
            "APP_MESSAGE_UNDERSTANDING_MODE must be one of: shadow, assist, enforce"
        )

    message_understanding = MessageUnderstandingSettings(
        enabled=_get_bool_env("APP_MESSAGE_UNDERSTANDING_ENABLED", default=False),
        mode=message_understanding_mode,
        model_name=_get_env("APP_MESSAGE_UNDERSTANDING_MODEL", default="gpt-4.1-mini"),
        temperature=_get_float_env(
            "APP_MESSAGE_UNDERSTANDING_TEMPERATURE",
            default=0.0,
            min_value=0.0,
            max_value=2.0,
        ),
        max_output_tokens=_get_int_env(
            "APP_MESSAGE_UNDERSTANDING_MAX_OUTPUT_TOKENS",
            default=700,
            min_value=64,
        ),
        min_confidence_to_apply=_get_float_env(
            "APP_MESSAGE_UNDERSTANDING_MIN_CONFIDENCE",
            default=0.72,
            min_value=0.0,
            max_value=1.0,
        ),
        request_timeout_seconds=_get_int_env(
            "APP_MESSAGE_UNDERSTANDING_TIMEOUT_SECONDS",
            default=20,
            min_value=1,
        ),
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
        message_understanding=message_understanding,
        telegram=telegram,
        logging=logging_settings,
    )