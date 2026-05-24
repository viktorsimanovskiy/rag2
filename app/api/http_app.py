# ============================================================
# File: app/api/http_app.py
# Purpose:
#   Minimal HTTP API for n8n / messenger integrations.
#
# Endpoints:
#   GET  /api/v1/health
#   GET  /api/v1/ready
#   POST /api/v1/answer
#
# Design:
#   - external messengers stay outside the domain core;
#   - n8n calls this API with normalized message data;
#   - all RAG logic remains inside existing services.
# ============================================================

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import AnswerRequest, AnswerResponse, ErrorResponse, HealthResponse
from app.config.settings import AppSettings, SettingsError, load_settings
from app.db.models.enums import ChannelTypeEnum, QuestionIntentEnum
from app.db.models.feedback import AnswerEvent
from app.runtime.app_runtime import AppRuntime, AppRuntimeConfig
from app.services.answers.answer_orchestrator import (
    OrchestratorValidationError,
    UserQuestionInput,
)

logger = logging.getLogger(__name__)


# ============================================================
# Application factory
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    runtime = AppRuntime(
        AppRuntimeConfig(
            database=settings.database,
        )
    )

    await runtime.startup()
    app.state.settings = settings
    app.state.runtime = runtime

    try:
        yield
    finally:
        await runtime.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG2 HTTP API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(SettingsError)
    async def settings_error_handler(_: Request, exc: SettingsError) -> JSONResponse:
        logger.exception("API settings error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error_code="settings_error",
                message="Ошибка настроек приложения.",
                details={"error": str(exc)},
            ).model_dump(mode="json"),
        )

    @app.exception_handler(OrchestratorValidationError)
    async def validation_error_handler(_: Request, exc: OrchestratorValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error_code="question_validation_error",
                message="Некорректный вопрос или параметры запроса.",
                details={"error": str(exc)},
            ).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unexpected API error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error_code="internal_error",
                message="Внутренняя ошибка при обработке вопроса.",
                details={"error": repr(exc)},
            ).model_dump(mode="json"),
        )

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        runtime = _get_runtime(request)
        return HealthResponse(
            status="ok",
            runtime_started=runtime.is_started,
        )

    @app.get("/api/v1/ready", response_model=HealthResponse)
    async def ready(request: Request) -> HealthResponse:
        runtime = _get_runtime(request)
        try:
            await runtime.get_db_manager().check_connection()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "ok": False,
                    "error_code": "database_not_ready",
                    "message": "База данных недоступна.",
                    "details": {"error": repr(exc)},
                },
            ) from exc

        return HealthResponse(
            status="ready",
            runtime_started=runtime.is_started,
        )

    @app.post("/api/v1/answer", response_model=AnswerResponse)
    async def answer(payload: AnswerRequest, request: Request) -> AnswerResponse:
        runtime = _get_runtime(request)
        channel_code = _parse_channel(payload.channel)
        external_session_id = _build_external_session_id(payload)
        request_metadata_json = _build_request_metadata(payload)

        async with runtime.session_scope() as session:
            factory = runtime.build_service_factory(session)
            orchestrator = factory.get_answer_orchestrator()

            outgoing = await orchestrator.handle_user_question(
                UserQuestionInput(
                    channel_code=channel_code,
                    external_session_id=external_session_id,
                    external_user_id=payload.external_user_id,
                    external_chat_id=payload.external_chat_id,
                    user_platform_name=payload.user_platform_name,
                    question_text=payload.question_text,
                    language_code=payload.language_code,
                    request_metadata_json=request_metadata_json,
                )
            )

            answer_event = await _load_answer_event(
                session,
                answer_event_id=outgoing.answer_event_id,
            )

        answer_payload_json = dict(answer_event.answer_payload_json or {})
        runtime_info = dict(answer_payload_json.get("runtime_answer_service") or {})
        runtime_debug = dict(runtime_info.get("debug_payload_json") or {})

        service_resolution = _extract_service_resolution(
            answer_payload_json=answer_payload_json,
            runtime_debug=runtime_debug,
        )
        warnings = _extract_warnings(answer_payload_json)

        debug_payload: dict[str, Any] | None = None
        if payload.debug:
            debug_payload = {
                "request_metadata_json": request_metadata_json,
                "answer_payload_json": answer_payload_json,
                "delivery_payload_json": outgoing.delivery_payload_json,
                "debug_payload_json": outgoing.debug_payload_json,
            }

        return AnswerResponse(
            ok=True,
            answer_text=outgoing.answer_text,
            answer_text_short=outgoing.answer_text_short,
            answer_mode=_enum_to_str(outgoing.answer_mode),
            session_id=outgoing.session_id,
            question_event_id=outgoing.question_event_id,
            answer_event_id=outgoing.answer_event_id,
            channel=channel_code.value,
            external_session_id=external_session_id,
            external_user_id=payload.external_user_id,
            external_chat_id=payload.external_chat_id,
            was_reused=outgoing.was_reused,
            should_request_feedback=outgoing.should_request_feedback,
            citations=outgoing.citations_json,
            service_resolution=service_resolution,
            warnings=warnings,
            debug=debug_payload,
        )

    return app


app = create_app()


# ============================================================
# Helpers
# ============================================================

def _get_runtime(request: Request) -> AppRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if not isinstance(runtime, AppRuntime):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "ok": False,
                "error_code": "runtime_not_started",
                "message": "Приложение ещё не готово к обработке запросов.",
            },
        )
    return runtime


def _parse_channel(value: str) -> ChannelTypeEnum:
    normalized = (value or "").strip().lower()
    aliases = {
        "tg": ChannelTypeEnum.TELEGRAM,
        "telegram": ChannelTypeEnum.TELEGRAM,
        "max": ChannelTypeEnum.MAX,
        "web": ChannelTypeEnum.WEB,
        "test": ChannelTypeEnum.TEST_CONSOLE,
        "test_console": ChannelTypeEnum.TEST_CONSOLE,
        "unknown": ChannelTypeEnum.UNKNOWN,
    }

    result = aliases.get(normalized)
    if result is not None:
        return result

    supported = ", ".join(sorted(aliases.keys()))
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            "ok": False,
            "error_code": "unsupported_channel",
            "message": "Неподдерживаемый канал.",
            "details": {
                "channel": value,
                "supported": supported,
            },
        },
    )


def _build_external_session_id(payload: AnswerRequest) -> str:
    if payload.external_session_id and payload.external_session_id.strip():
        return " ".join(payload.external_session_id.split()).strip()

    stable_id = (
        payload.external_chat_id
        or payload.external_user_id
        or "anonymous"
    )
    return f"{payload.channel}:{stable_id}"


def _build_request_metadata(payload: AnswerRequest) -> dict[str, Any]:
    metadata = dict(payload.request_metadata_json or {})
    metadata.setdefault("source", "http_api")
    metadata.setdefault("api_debug_requested", bool(payload.debug))

    if payload.intent_type:
        # AnswerOrchestrator reads this key and uses it as an explicit override.
        metadata["forced_intent_type"] = _parse_intent_type(payload.intent_type).value
        metadata["forced_intent_source"] = "api_request"

    return metadata


def _parse_intent_type(value: str) -> QuestionIntentEnum:
    normalized = (value or "").strip().lower()
    aliases = {
        "documents": QuestionIntentEnum.DOCUMENTS_QUESTION,
        "docs": QuestionIntentEnum.DOCUMENTS_QUESTION,
        "refusal": QuestionIntentEnum.REJECTION_QUESTION,
        "rejection": QuestionIntentEnum.REJECTION_QUESTION,
        "deadline": QuestionIntentEnum.DEADLINE_QUESTION,
        "payment": QuestionIntentEnum.PAYMENT_TIMING_QUESTION,
        "amount": QuestionIntentEnum.AMOUNT_QUESTION,
        "eligibility": QuestionIntentEnum.ELIGIBILITY_QUESTION,
        "procedure": QuestionIntentEnum.PROCEDURE_QUESTION,
        "appeal": QuestionIntentEnum.APPEAL_QUESTION,
        "form": QuestionIntentEnum.FORM_QUESTION,
        "mixed": QuestionIntentEnum.MIXED_QUESTION,
        "ambiguous": QuestionIntentEnum.AMBIGUOUS_QUESTION,
        "other": QuestionIntentEnum.OTHER,
    }

    if normalized in aliases:
        return aliases[normalized]

    try:
        return QuestionIntentEnum(normalized)
    except ValueError as exc:
        supported = ", ".join(item.value for item in QuestionIntentEnum)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "ok": False,
                "error_code": "unsupported_intent_type",
                "message": "Неподдерживаемый тип вопроса.",
                "details": {
                    "intent_type": value,
                    "supported": supported,
                },
            },
        ) from exc


async def _load_answer_event(
    session: AsyncSession,
    *,
    answer_event_id: Any,
) -> AnswerEvent:
    stmt = select(AnswerEvent).where(AnswerEvent.answer_event_id == answer_event_id)
    result = await session.execute(stmt)
    answer_event = result.scalar_one_or_none()
    if answer_event is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "ok": False,
                "error_code": "answer_event_not_found",
                "message": "Ответ создан, но запись ответа не найдена.",
            },
        )
    return answer_event


def _extract_service_resolution(
    *,
    answer_payload_json: dict[str, Any],
    runtime_debug: dict[str, Any],
) -> dict[str, Any]:
    direct = runtime_debug.get("service_resolution")
    if isinstance(direct, dict):
        return direct

    runtime_info = answer_payload_json.get("runtime_answer_service")
    if isinstance(runtime_info, dict):
        nested_debug = runtime_info.get("debug_payload_json")
        if isinstance(nested_debug, dict):
            nested = nested_debug.get("service_resolution")
            if isinstance(nested, dict):
                return nested

    return {}


def _extract_warnings(answer_payload_json: dict[str, Any]) -> list[str]:
    raw = answer_payload_json.get("warnings")
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    return []


def _enum_to_str(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw)
