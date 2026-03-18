# ============================================================
# File: app/channels/telegram_bot.py
# Purpose:
#   Telegram runtime entrypoint built around the CURRENT project contracts.
#
# Responsibilities:
#   - accept Telegram-like message/callback updates
#   - convert message update into UserQuestionInput
#   - call AnswerOrchestrator
#   - build MessengerResponse via MessengerResponseBuilder
#   - adapt MessengerResponse into TelegramSendMessage
#   - parse and persist feedback callbacks
#
# Important:
#   This file is intentionally aligned with the CURRENT archive code:
#   - MessengerResponseBuilder.build(...)
#   - TelegramMessageAdapter.build_send_message(...)
#   - TelegramMessageAdapter.parse_callback(...)
#   - callback format: fb:<answer_event_id>:<score>
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID

from sqlalchemy import Select, select

from app.bootstrap.service_factory import ServiceFactory
from app.db.models.enums import ChannelTypeEnum
from app.db.models.feedback import AnswerEvent, QuestionEvent
from app.adapters.telegram.telegram_message_adapter import (
    ParsedFeedbackCallback,
    TelegramCallbackEvent,
)
from app.services.answers.answer_orchestrator import UserQuestionInput
from app.services.channels.messenger_response_builder import MessengerResponseBuildInput
from app.services.feedback.feedback_service import FeedbackInput

logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================

class TelegramBotError(Exception):
    """Base Telegram runtime error."""


class TelegramBotValidationError(TelegramBotError):
    """Raised when incoming Telegram payload is invalid."""


class TelegramBotCallbackError(TelegramBotError):
    """Raised when callback processing fails."""


# ============================================================
# Incoming DTOs (framework-agnostic)
# ============================================================

@dataclass(slots=True)
class TelegramUser:
    telegram_user_id: str
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    language_code: Optional[str] = None


@dataclass(slots=True)
class TelegramChat:
    telegram_chat_id: str
    chat_type: Optional[str] = None
    title: Optional[str] = None


@dataclass(slots=True)
class TelegramMessageUpdate:
    update_id: str
    message_id: int
    text: str
    chat: TelegramChat
    user: TelegramUser
    metadata_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TelegramCallbackUpdate:
    update_id: str
    callback_query_id: str
    callback_data: str
    chat: TelegramChat
    user: TelegramUser
    message_id: Optional[int] = None
    metadata_json: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Outgoing DTOs (framework-agnostic)
# ============================================================

@dataclass(slots=True)
class TelegramOutgoingMessage:
    chat_id: str
    text: str
    parse_mode: Optional[str] = None
    reply_markup: Optional[dict[str, Any]] = None
    metadata_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TelegramOutgoingAck:
    callback_query_id: str
    text: str
    show_alert: bool = False
    metadata_json: dict[str, Any] = field(default_factory=dict)


TelegramMessageSender = Callable[[TelegramOutgoingMessage], Awaitable[None]]
TelegramCallbackAcker = Callable[[TelegramOutgoingAck], Awaitable[None]]


# ============================================================
# Runtime handler
# ============================================================

class TelegramBotHandler:
    """
    High-level Telegram runtime handler.

    This class is intentionally thin:
    - orchestration stays in AnswerOrchestrator
    - formatting stays in MessengerResponseBuilder
    - Telegram specifics stay in TelegramMessageAdapter
    """

    def __init__(self, *, service_factory: ServiceFactory) -> None:
        self.service_factory = service_factory

    # --------------------------------------------------------
    # Message flow
    # --------------------------------------------------------

    async def handle_message_update(
        self,
        update: TelegramMessageUpdate,
        *,
        send_message: TelegramMessageSender,
    ) -> None:
        self._validate_message_update(update)

        orchestrator = self.service_factory.get_answer_orchestrator()
        response_builder = self.service_factory.get_messenger_response_builder()
        telegram_adapter = self.service_factory.get_telegram_message_adapter()

        question_input = self._build_user_question_input(update)

        logger.info(
            "Handling Telegram message",
            extra={
                "update_id": update.update_id,
                "message_id": update.message_id,
                "chat_id": update.chat.telegram_chat_id,
                "telegram_user_id": update.user.telegram_user_id,
            },
        )

        outgoing_answer = await orchestrator.handle_user_question(question_input)

        messenger_response = response_builder.build(
            MessengerResponseBuildInput(
                channel_code=ChannelTypeEnum.TELEGRAM,
                payload=outgoing_answer,
                include_citations=True,
                include_feedback_controls=True,
                include_service_note_for_reuse=False,
                use_short_answer_when_available=False,
            )
        )

        telegram_message = telegram_adapter.build_send_message(
            chat_id=update.chat.telegram_chat_id,
            response=messenger_response,
        )

        await send_message(
            TelegramOutgoingMessage(
                chat_id=telegram_message.chat_id,
                text=telegram_message.text,
                parse_mode=telegram_message.parse_mode,
                reply_markup=telegram_message.reply_markup,
                metadata_json={
                    "source": "telegram_bot_handler",
                    "update_id": update.update_id,
                    "message_id": update.message_id,
                    "answer_event_id": str(outgoing_answer.answer_event_id),
                    "question_event_id": str(outgoing_answer.question_event_id),
                    "session_id": str(outgoing_answer.session_id),
                },
            )
        )

        logger.info(
            "Telegram message handled successfully",
            extra={
                "update_id": update.update_id,
                "chat_id": update.chat.telegram_chat_id,
                "answer_event_id": str(outgoing_answer.answer_event_id),
            },
        )

    # --------------------------------------------------------
    # Callback flow
    # --------------------------------------------------------

    async def handle_callback_update(
        self,
        update: TelegramCallbackUpdate,
        *,
        ack_callback: TelegramCallbackAcker,
    ) -> None:
        self._validate_callback_update(update)

        telegram_adapter = self.service_factory.get_telegram_message_adapter()

        parsed = telegram_adapter.parse_callback(
            TelegramCallbackEvent(
                user_id=update.user.telegram_user_id,
                chat_id=update.chat.telegram_chat_id,
                message_id=update.message_id or 0,
                callback_data=update.callback_data,
            )
        )

        if parsed is None:
            raise TelegramBotCallbackError(
                f"Unsupported callback payload: {update.callback_data}"
            )

        if parsed.action == "rating":
            await self._handle_rating_callback(
                update=update,
                parsed=parsed,
                ack_callback=ack_callback,
            )
            return

        if parsed.action == "comment":
            await ack_callback(
                TelegramOutgoingAck(
                    callback_query_id=update.callback_query_id,
                    text="Комментарии к ответу будут поддержаны на следующем этапе.",
                    show_alert=False,
                    metadata_json={
                        "source": "telegram_bot_handler",
                        "action": "comment_not_implemented",
                        "answer_event_id": parsed.answer_event_id,
                    },
                )
            )
            return

        raise TelegramBotCallbackError(f"Unsupported callback action: {parsed.action}")

    async def _handle_rating_callback(
        self,
        *,
        update: TelegramCallbackUpdate,
        parsed: ParsedFeedbackCallback,
        ack_callback: TelegramCallbackAcker,
    ) -> None:
        if parsed.score is None:
            raise TelegramBotCallbackError("Rating callback does not contain score.")

        feedback_service = self.service_factory.get_feedback_service()

        answer_event_id = UUID(parsed.answer_event_id)
        session_id = await self._resolve_session_id_by_answer_event_id(answer_event_id)

        await feedback_service.record_feedback(
            FeedbackInput(
                answer_event_id=answer_event_id,
                session_id=session_id,
                score=parsed.score,
                comment_text=None,
                metadata_json={
                    "source": "telegram_callback",
                    "update_id": update.update_id,
                    "callback_query_id": update.callback_query_id,
                    "chat_id": update.chat.telegram_chat_id,
                    "telegram_user_id": update.user.telegram_user_id,
                    "raw_callback_data": update.callback_data,
                },
            )
        )

        await ack_callback(
            TelegramOutgoingAck(
                callback_query_id=update.callback_query_id,
                text=self._build_feedback_ack_text(parsed.score),
                show_alert=False,
                metadata_json={
                    "source": "telegram_bot_handler",
                    "action": "rating_recorded",
                    "answer_event_id": parsed.answer_event_id,
                    "score": parsed.score,
                },
            )
        )

        logger.info(
            "Telegram feedback recorded",
            extra={
                "update_id": update.update_id,
                "callback_query_id": update.callback_query_id,
                "answer_event_id": parsed.answer_event_id,
                "score": parsed.score,
            },
        )

    # --------------------------------------------------------
    # Internal builders
    # --------------------------------------------------------

    def _build_user_question_input(
        self,
        update: TelegramMessageUpdate,
    ) -> UserQuestionInput:
        language_code = update.user.language_code or "ru"

        return UserQuestionInput(
            channel_code=ChannelTypeEnum.TELEGRAM,
            external_session_id=self._build_external_session_id(update),
            external_user_id=update.user.telegram_user_id,
            external_chat_id=update.chat.telegram_chat_id,
            user_platform_name="telegram",
            question_text=update.text.strip(),
            language_code=language_code,
            request_metadata_json={
                "telegram": {
                    "update_id": update.update_id,
                    "message_id": update.message_id,
                    "chat_id": update.chat.telegram_chat_id,
                    "chat_type": update.chat.chat_type,
                    "user_id": update.user.telegram_user_id,
                    "username": update.user.username,
                    "first_name": update.user.first_name,
                    "last_name": update.user.last_name,
                },
                "raw_metadata": update.metadata_json,
            },
        )

    def _build_external_session_id(
        self,
        update: TelegramMessageUpdate,
    ) -> str:
        """
        Conservative policy:
        one Telegram chat = one conversation session.
        """
        return f"telegram_chat:{update.chat.telegram_chat_id}"

    # --------------------------------------------------------
    # DB helpers
    # --------------------------------------------------------

    async def _resolve_session_id_by_answer_event_id(
        self,
        answer_event_id: UUID,
    ) -> UUID:
        """
        Current callback payload contains only answer_event_id.
        Therefore session_id is resolved through:
            answer_event -> question_event -> session_id
        """
        stmt: Select[Any] = (
            select(QuestionEvent.session_id)
            .join(
                AnswerEvent,
                AnswerEvent.question_event_id == QuestionEvent.question_event_id,
            )
            .where(AnswerEvent.answer_event_id == answer_event_id)
        )

        result = await self.service_factory.db.execute(stmt)
        session_id = result.scalar_one_or_none()

        if session_id is None:
            raise TelegramBotCallbackError(
                f"Cannot resolve session_id for answer_event_id={answer_event_id}"
            )

        return session_id

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def _validate_message_update(
        self,
        update: TelegramMessageUpdate,
    ) -> None:
        if not update.update_id:
            raise TelegramBotValidationError("update_id is required.")
        if not update.text or not update.text.strip():
            raise TelegramBotValidationError("message text is empty.")
        if not update.chat.telegram_chat_id:
            raise TelegramBotValidationError("telegram_chat_id is required.")
        if not update.user.telegram_user_id:
            raise TelegramBotValidationError("telegram_user_id is required.")

    def _validate_callback_update(
        self,
        update: TelegramCallbackUpdate,
    ) -> None:
        if not update.update_id:
            raise TelegramBotValidationError("update_id is required.")
        if not update.callback_query_id:
            raise TelegramBotValidationError("callback_query_id is required.")
        if not update.callback_data or not update.callback_data.strip():
            raise TelegramBotValidationError("callback_data is empty.")
        if not update.chat.telegram_chat_id:
            raise TelegramBotValidationError("telegram_chat_id is required.")
        if not update.user.telegram_user_id:
            raise TelegramBotValidationError("telegram_user_id is required.")

    # --------------------------------------------------------
    # UI helpers
    # --------------------------------------------------------

    def _build_feedback_ack_text(
        self,
        score: int,
    ) -> str:
        if score >= 4:
            return "Спасибо за оценку."
        if score == 3:
            return "Спасибо, мы учтём эту оценку."
        return "Спасибо. Это поможет улучшить ответы."