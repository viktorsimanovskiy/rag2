# ============================================================
# File: app/services/channels/messenger_response_builder.py
# Purpose:
#   Build messenger-specific response payloads from transport-agnostic
#   OutgoingAnswerPayload objects produced by the answer orchestrator.
#
# Responsibilities:
#   - format answer text for Telegram / MAX
#   - append citations in user-friendly form
#   - build feedback rating controls (1..5)
#   - keep delivery concerns out of core RAG logic
#   - provide stable response contracts for channel adapters
#
# Design principles:
#   - builder is pure application-layer formatting logic
#   - no DB access
#   - no retrieval / generation logic
#   - deterministic output
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from app.db.models.enums import AnswerModeEnum, ChannelTypeEnum


# ============================================================
# Shared DTOs
# ============================================================

@dataclass(slots=True)
class OutgoingAnswerPayload:
    """
    Local contract expected from answer orchestrator.

    If your project already defines this DTO elsewhere
    (for example in answer_orchestrator.py or app/contracts),
    import that version instead of duplicating this class.
    """
    answer_event_id: UUID
    session_id: UUID
    question_event_id: UUID

    answer_text: str
    answer_text_short: Optional[str]
    citations_json: list[dict[str, Any]]

    answer_mode: AnswerModeEnum
    was_reused: bool
    reused_from_answer_event_id: Optional[UUID]

    should_request_feedback: bool
    feedback_payload_json: dict[str, Any]

    delivery_payload_json: dict[str, Any] = field(default_factory=dict)
    debug_payload_json: dict[str, Any] = field(default_factory=dict)


class MessengerMarkupMode(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN = "plain"


@dataclass(slots=True)
class MessengerButton:
    """
    Generic button description for messenger adapters.
    """
    text: str
    callback_data: Optional[str] = None
    url: Optional[str] = None


@dataclass(slots=True)
class MessengerKeyboard:
    """
    Messenger-agnostic inline keyboard representation.
    Rows are preserved to let adapters render appropriately.
    """
    rows: list[list[MessengerButton]] = field(default_factory=list)


@dataclass(slots=True)
class MessengerResponse:
    """
    Final response structure that channel adapters can translate
    to Telegram/MAX SDK-specific payloads.
    """
    channel_code: ChannelTypeEnum
    text: str
    markup_mode: MessengerMarkupMode
    keyboard: Optional[MessengerKeyboard]

    metadata_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MessengerResponseBuildInput:
    """
    Input for channel response builder.
    """
    channel_code: ChannelTypeEnum
    payload: OutgoingAnswerPayload

    include_citations: bool = True
    include_feedback_controls: bool = True
    include_service_note_for_reuse: bool = False
    use_short_answer_when_available: bool = False


@dataclass(slots=True)
class CitationView:
    """
    Normalized, display-oriented citation.
    """
    label: str
    url: Optional[str] = None


# ============================================================
# Builder
# ============================================================

class MessengerResponseBuilder:
    """
    Converts transport-agnostic answer payloads into response structures
    suitable for Telegram/MAX adapters.
    """

    def build(
        self,
        data: MessengerResponseBuildInput,
    ) -> MessengerResponse:
        payload = data.payload

        base_text = self._select_answer_text(
            payload=payload,
            use_short_answer_when_available=data.use_short_answer_when_available,
        )

        text_parts: list[str] = []

        reuse_note = self._build_reuse_note(
            payload=payload,
            include_service_note_for_reuse=data.include_service_note_for_reuse,
        )
        if reuse_note:
            text_parts.append(reuse_note)

        text_parts.append(base_text.strip())

        if data.include_citations:
            citation_block = self._build_citation_block(
                channel_code=data.channel_code,
                citations_json=payload.citations_json,
            )
            if citation_block:
                text_parts.append(citation_block)

        final_text = "\n\n".join(part for part in text_parts if part and part.strip())

        keyboard = None
        if data.include_feedback_controls and payload.should_request_feedback:
            keyboard = self._build_feedback_keyboard(
                channel_code=data.channel_code,
                payload=payload,
            )

        markup_mode = self._resolve_markup_mode(data.channel_code)

        metadata_json = {
            "answer_event_id": str(payload.answer_event_id),
            "question_event_id": str(payload.question_event_id),
            "session_id": str(payload.session_id),
            "answer_mode": str(payload.answer_mode),
            "was_reused": payload.was_reused,
            "reused_from_answer_event_id": (
                str(payload.reused_from_answer_event_id)
                if payload.reused_from_answer_event_id
                else None
            ),
            "feedback_enabled": payload.should_request_feedback,
        }

        return MessengerResponse(
            channel_code=data.channel_code,
            text=final_text,
            markup_mode=markup_mode,
            keyboard=keyboard,
            metadata_json=metadata_json,
        )

    # --------------------------------------------------------
    # Core text building
    # --------------------------------------------------------

    def _select_answer_text(
        self,
        *,
        payload: OutgoingAnswerPayload,
        use_short_answer_when_available: bool,
    ) -> str:
        if (
            use_short_answer_when_available
            and payload.answer_text_short
            and payload.answer_text_short.strip()
        ):
            return payload.answer_text_short.strip()

        return payload.answer_text.strip()

    def _build_reuse_note(
        self,
        *,
        payload: OutgoingAnswerPayload,
        include_service_note_for_reuse: bool,
    ) -> Optional[str]:
        if not payload.was_reused:
            return None

        if not include_service_note_for_reuse:
            return None

        # Это сервисная пометка. По умолчанию выключена, чтобы не перегружать UX.
        return "_Ответ сформирован с использованием ранее подтверждённого ответа по актуальным источникам._"

    def _build_citation_block(
        self,
        *,
        channel_code: ChannelTypeEnum,
        citations_json: list[dict[str, Any]],
    ) -> Optional[str]:
        citations = self._normalize_citations(citations_json)
        if not citations:
            return None

        lines: list[str] = []
        header = self._format_bold("Источники", channel_code)
        lines.append(header)

        for idx, citation in enumerate(citations, start=1):
            if citation.url:
                line = self._format_linked_citation(
                    index=idx,
                    label=citation.label,
                    url=citation.url,
                    channel_code=channel_code,
                )
            else:
                line = f"{idx}. {citation.label}"
            lines.append(line)

        return "\n".join(lines)

    def _normalize_citations(
        self,
        citations_json: list[dict[str, Any]],
    ) -> list[CitationView]:
        normalized: list[CitationView] = []

        for item in citations_json or []:
            label = self._first_non_empty(
                item.get("display_label"),
                item.get("label"),
                item.get("title"),
                item.get("citation_text"),
                item.get("anchor_text"),
                "Источник",
            )

            url = self._first_non_empty(
                item.get("download_url"),
                item.get("document_url"),
                item.get("url"),
            )

            normalized.append(
                CitationView(
                    label=str(label).strip(),
                    url=str(url).strip() if url else None,
                )
            )

        return normalized

    # --------------------------------------------------------
    # Feedback controls
    # --------------------------------------------------------

    def _build_feedback_keyboard(
        self,
        *,
        channel_code: ChannelTypeEnum,
        payload: OutgoingAnswerPayload,
    ) -> MessengerKeyboard:
        """
        Builds 1..5 rating keyboard.

        callback_data format:
            fb:<answer_event_id>:<score>

        Additional comment flow can be launched by channel adapter later.
        """
        _ = channel_code  # reserved for future channel-specific branching

        answer_event_id = str(payload.answer_event_id)

        rating_row = [
            MessengerButton(text="1", callback_data=f"fb:{answer_event_id}:1"),
            MessengerButton(text="2", callback_data=f"fb:{answer_event_id}:2"),
            MessengerButton(text="3", callback_data=f"fb:{answer_event_id}:3"),
            MessengerButton(text="4", callback_data=f"fb:{answer_event_id}:4"),
            MessengerButton(text="5", callback_data=f"fb:{answer_event_id}:5"),
        ]

        comment_row = [
            MessengerButton(
                text="Оставить комментарий",
                callback_data=f"fb_comment:{answer_event_id}",
            )
        ]

        return MessengerKeyboard(rows=[rating_row, comment_row])

    # --------------------------------------------------------
    # Formatting helpers
    # --------------------------------------------------------

    def _resolve_markup_mode(
        self,
        channel_code: ChannelTypeEnum,
    ) -> MessengerMarkupMode:
        if channel_code in {ChannelTypeEnum.TELEGRAM, ChannelTypeEnum.MAX}:
            return MessengerMarkupMode.MARKDOWN
        return MessengerMarkupMode.PLAIN

    def _format_bold(
        self,
        text: str,
        channel_code: ChannelTypeEnum,
    ) -> str:
        markup = self._resolve_markup_mode(channel_code)
        if markup == MessengerMarkupMode.MARKDOWN:
            return f"*{self._escape_markdown(text)}*"
        if markup == MessengerMarkupMode.HTML:
            return f"<b>{text}</b>"
        return text

    def _format_linked_citation(
        self,
        *,
        index: int,
        label: str,
        url: str,
        channel_code: ChannelTypeEnum,
    ) -> str:
        markup = self._resolve_markup_mode(channel_code)

        safe_label = label.strip() or "Источник"
        safe_url = url.strip()

        if markup == MessengerMarkupMode.MARKDOWN:
            return f"{index}. [{self._escape_markdown(safe_label)}]({safe_url})"

        if markup == MessengerMarkupMode.HTML:
            return f'{index}. <a href="{safe_url}">{safe_label}</a>'

        return f"{index}. {safe_label}: {safe_url}"

    def _escape_markdown(
        self,
        text: str,
    ) -> str:
        """
        Conservative escaping for common MarkdownV2-sensitive characters.
        Safe enough as an application-layer default.
        """
        chars_to_escape = r"_*[]()~`>#+-=|{}.!"
        result = text
        for ch in chars_to_escape:
            result = result.replace(ch, f"\\{ch}")
        return result

    def _first_non_empty(
        self,
        *values: Any,
    ) -> Optional[str]:
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None