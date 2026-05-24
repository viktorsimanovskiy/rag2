# ============================================================
# File: app/api/schemas.py
# Purpose:
#   Request/response schemas for the RAG2 HTTP API.
# ============================================================

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class AnswerRequest(BaseModel):
    """
    Входной запрос от n8n или другого внешнего канала.

    Минимально достаточно передать question_text. Остальные поля нужны,
    чтобы связать вопрос с каналом, пользователем и чатом.
    """

    question_text: str = Field(..., min_length=1, max_length=10000)

    channel: str = Field(default="web", max_length=50)
    external_session_id: Optional[str] = Field(default=None, max_length=500)
    external_user_id: Optional[str] = Field(default=None, max_length=500)
    external_chat_id: Optional[str] = Field(default=None, max_length=500)
    user_platform_name: Optional[str] = Field(default=None, max_length=500)

    language_code: str = Field(default="ru", max_length=20)
    intent_type: Optional[str] = Field(default=None, max_length=100)
    debug: bool = False

    request_metadata_json: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question_text")
    @classmethod
    def validate_question_text(cls, value: str) -> str:
        normalized = " ".join(str(value).split()).strip()
        if not normalized:
            raise ValueError("question_text must not be empty")
        return normalized

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, value: str) -> str:
        normalized = " ".join(str(value or "").split()).strip().lower()
        if not normalized:
            return "web"
        return normalized

    @field_validator("language_code")
    @classmethod
    def validate_language_code(cls, value: str) -> str:
        normalized = " ".join(str(value or "").split()).strip().lower()
        return normalized or "ru"

    @field_validator("intent_type")
    @classmethod
    def validate_intent_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = " ".join(str(value).split()).strip().lower()
        return normalized or None


class HealthResponse(BaseModel):
    status: str
    app: str = "rag2"
    runtime_started: bool


class AnswerResponse(BaseModel):
    """
    Ответ API в форме, удобной для n8n.
    """

    ok: bool

    answer_text: str
    answer_text_short: Optional[str] = None
    answer_mode: str

    session_id: UUID
    question_event_id: UUID
    answer_event_id: UUID

    channel: str
    external_session_id: str
    external_user_id: Optional[str] = None
    external_chat_id: Optional[str] = None

    was_reused: bool
    should_request_feedback: bool

    citations: list[dict[str, Any]] = Field(default_factory=list)
    service_resolution: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    debug: Optional[dict[str, Any]] = None


class ErrorResponse(BaseModel):
    ok: bool = False
    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
