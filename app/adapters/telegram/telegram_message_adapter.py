# ============================================================
# File: app/adapters/telegram/telegram_message_adapter.py
# Purpose:
#   Translate MessengerResponse objects into Telegram API payloads
#   and parse incoming Telegram callbacks.
#
# Responsibilities:
#   - convert MessengerKeyboard -> Telegram InlineKeyboard
#   - build send_message payload
#   - process rating callbacks (1..5)
#   - process comment callbacks
#   - isolate Telegram-specific logic from application layer
#
# Design principles
#   - no DB logic here
#   - pure adapter layer
#   - stable contract with orchestrator
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.services.channels.messenger_response_builder import (
    MessengerButton,
    MessengerKeyboard,
    MessengerResponse,
    MessengerMarkupMode,
)


# ============================================================
# Telegram DTO
# ============================================================

@dataclass(slots=True)
class TelegramSendMessage:
    chat_id: str
    text: str
    parse_mode: Optional[str]
    reply_markup: Optional[dict[str, Any]]


@dataclass(slots=True)
class TelegramCallbackEvent:
    """
    Normalized callback event from Telegram.
    """

    user_id: str
    chat_id: str
    message_id: int
    callback_data: str


@dataclass(slots=True)
class ParsedFeedbackCallback:
    """
    Result of parsing callback payload.

    Example payloads:

    fb:<answer_event_id>:5
    fb_comment:<answer_event_id>
    """

    action: str
    answer_event_id: str
    score: Optional[int] = None


# ============================================================
# Adapter
# ============================================================

class TelegramMessageAdapter:
    """
    Adapter converting internal MessengerResponse into Telegram payload.
    """

    # --------------------------------------------------------
    # Send message
    # --------------------------------------------------------

    def build_send_message(
        self,
        *,
        chat_id: str,
        response: MessengerResponse,
    ) -> TelegramSendMessage:
        """
        Convert MessengerResponse -> TelegramSendMessage
        """

        parse_mode = self._resolve_parse_mode(response.markup_mode)

        keyboard = None
        if response.keyboard:
            keyboard = self._convert_keyboard(response.keyboard)

        return TelegramSendMessage(
            chat_id=chat_id,
            text=response.text,
            parse_mode=parse_mode,
            reply_markup=keyboard,
        )

    # --------------------------------------------------------
    # Keyboard conversion
    # --------------------------------------------------------

    def _convert_keyboard(
        self,
        keyboard: MessengerKeyboard,
    ) -> dict[str, Any]:
        """
        Convert MessengerKeyboard -> Telegram inline keyboard
        """

        rows = []

        for row in keyboard.rows:
            telegram_row = []

            for btn in row:
                telegram_button = {
                    "text": btn.text,
                }

                if btn.callback_data:
                    telegram_button["callback_data"] = btn.callback_data

                if btn.url:
                    telegram_button["url"] = btn.url

                telegram_row.append(telegram_button)

            rows.append(telegram_row)

        return {"inline_keyboard": rows}

    # --------------------------------------------------------
    # Parse callback
    # --------------------------------------------------------

    def parse_callback(
        self,
        event: TelegramCallbackEvent,
    ) -> Optional[ParsedFeedbackCallback]:
        """
        Parse Telegram callback payload.

        Supported formats:

        fb:<answer_event_id>:<score>
        fb_comment:<answer_event_id>
        """

        payload = event.callback_data

        if payload.startswith("fb:"):

            parts = payload.split(":")

            if len(parts) != 3:
                return None

            answer_event_id = parts[1]

            try:
                score = int(parts[2])
            except ValueError:
                return None

            return ParsedFeedbackCallback(
                action="rating",
                answer_event_id=answer_event_id,
                score=score,
            )

        if payload.startswith("fb_comment:"):

            parts = payload.split(":")

            if len(parts) != 2:
                return None

            answer_event_id = parts[1]

            return ParsedFeedbackCallback(
                action="comment",
                answer_event_id=answer_event_id,
            )

        return None

    # --------------------------------------------------------
    # Parse mode
    # --------------------------------------------------------

    def _resolve_parse_mode(
        self,
        markup_mode: MessengerMarkupMode,
    ) -> Optional[str]:

        if markup_mode == MessengerMarkupMode.MARKDOWN:
            return "MarkdownV2"

        if markup_mode == MessengerMarkupMode.HTML:
            return "HTML"

        return None