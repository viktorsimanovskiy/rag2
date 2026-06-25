# ============================================================
# File: app/services/answers/message_guard.py
# Purpose:
#   Fast rule-based pre-retrieval guard for incoming user messages.
#
# Responsibilities:
#   - block technical pings, greetings, thanks and obviously non-domain input
#     before intent/retrieval;
#   - block suspicious code/SQL/script-like payloads before RAG;
#   - return a safe service response without legal evidence when retrieval is
#     not appropriate.
#
# Design:
#   - no LLM calls;
#   - no legal conclusions;
#   - conservative allow-list behavior for short service/social messages;
#   - domain questions continue to normal RAG path.
# ============================================================

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


MESSAGE_GUARD_VERSION = "second_step_01_message_guard_v1"


@dataclass(slots=True)
class MessageGuardResult:
    """Result of fast message pre-check before retrieval."""

    should_run_rag: bool
    message_kind: str
    reason_code: str
    confidence_score: float
    answer_text: Optional[str] = None
    normalized_text: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    guard_version: str = MESSAGE_GUARD_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "guard_version": self.guard_version,
            "should_run_rag": self.should_run_rag,
            "message_kind": self.message_kind,
            "reason_code": self.reason_code,
            "confidence_score": self.confidence_score,
            "answer_text": self.answer_text,
            "normalized_text": self.normalized_text,
            "details": self.details,
        }


class RuleBasedMessageGuard:
    """Cheap first-line guard for messages that should not enter retrieval."""

    _TECHNICAL_MESSAGES = {
        "проверка связи",
        "связь",
        "тест",
        "test",
        "ping",
        "пинг",
        "ау",
        "бот работает",
        "работаешь",
        "ты работаешь",
        "проверка",
    }

    _START_MESSAGES = {
        "/start",
        "start",
        "старт",
        "начать",
        "запуск",
    }

    _HELP_MESSAGES = {
        "/help",
        "help",
        "помощь",
        "что ты умеешь",
        "как пользоваться",
        "команды",
    }

    _GREETING_MESSAGES = {
        "привет",
        "здравствуйте",
        "добрый день",
        "доброе утро",
        "добрый вечер",
        "хай",
        "hello",
        "hi",
    }

    _THANKS_MESSAGES = {
        "спасибо",
        "благодарю",
        "понял",
        "поняла",
        "ок",
        "окей",
        "хорошо",
    }

    _DOMAIN_HINTS_RE = re.compile(
        r"\b("
        r"документ|документы|список|перечень|выплат|пособ|субсид|компенсац|"
        r"льгот|мера|поддержк|положен|право|заявлен|мфц|епгу|рпгу|отказ|"
        r"срок|решени|получ|оформ|соцконтракт|тжс|едв|жку|донор|ветеран|"
        r"инвалид|ребен|ребён|семь|семья|малоимущ|многодет|чс|пожар|дров|"
        r"погребен|захорон|санатор|путев|проезд|маткапитал|сертификат|"
        r"удостоверени|реабилит|чернобыл|сво|таймыр|эвенк|краснояр"
        r")",
        re.IGNORECASE,
    )

    _SUSPICIOUS_CODE_RE = re.compile(
        r"(<\s*script\b|</\s*script\s*>|javascript\s*:|onerror\s*=|onload\s*=|"
        r"\b(select|insert|update|delete|drop|truncate|alter|union)\b\s+.*\b(from|table|where|into|database)\b|"
        r"\b(or|and)\b\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+['\"]?|"
        r"--\s*$|/\*|\*/|\bexec\s*\(|\beval\s*\(|\bos\.system\s*\()",
        re.IGNORECASE | re.DOTALL,
    )

    _URL_ONLY_RE = re.compile(r"^(https?://\S+|www\.\S+)$", re.IGNORECASE)

    async def check(self, message_text: str, *, channel_code: str | None = None) -> MessageGuardResult:
        original = message_text or ""
        normalized = self._normalize(original)
        lowered = normalized.lower()

        if not normalized:
            return self._blocked(
                normalized,
                message_kind="empty_message",
                reason_code="empty_message",
                confidence_score=1.0,
                answer_text="Напиши, пожалуйста, вопрос по мере социальной поддержки: например про документы, срок, отказ, выплату или порядок обращения.",
            )

        if len(normalized) > 5000:
            return self._blocked(
                normalized,
                message_kind="too_long_message",
                reason_code="message_too_long_for_chat_guard",
                confidence_score=0.95,
                answer_text="Сообщение слишком длинное для быстрого ответа. Сформулируй, пожалуйста, один конкретный вопрос по мере социальной поддержки.",
                details={"length": len(normalized)},
            )

        if self._SUSPICIOUS_CODE_RE.search(normalized):
            return self._blocked(
                normalized,
                message_kind="unsafe_or_code_like_message",
                reason_code="suspicious_code_or_injection_pattern",
                confidence_score=0.95,
                answer_text="Я могу отвечать на вопросы по мерам социальной поддержки. Сообщения с кодом, командами или подозрительными конструкциями в поиск по нормативным актам не передаются.",
            )

        if self._URL_ONLY_RE.match(normalized):
            return self._blocked(
                normalized,
                message_kind="url_only_message",
                reason_code="url_only_message",
                confidence_score=0.9,
                answer_text="Я не открываю ссылки в этом чате. Напиши сам вопрос текстом: какая мера поддержки интересует и что нужно узнать — документы, срок, отказ или порядок обращения.",
            )

        if self._is_mostly_symbols(normalized):
            return self._blocked(
                normalized,
                message_kind="garbage_message",
                reason_code="mostly_symbols_or_unreadable",
                confidence_score=0.9,
                answer_text="Не получилось распознать вопрос. Напиши, пожалуйста, обычным текстом, какая мера поддержки интересует.",
            )

        if lowered in self._START_MESSAGES:
            return self._blocked(
                normalized,
                message_kind="start_command",
                reason_code="start_command",
                confidence_score=1.0,
                answer_text=(
                    "Я могу помочь с вопросами по мерам социальной поддержки Красноярского края: "
                    "документы, сроки, основания отказа, порядок обращения и возможные меры по жизненной ситуации. "
                    "Напиши вопрос обычным текстом."
                ),
            )

        if lowered in self._HELP_MESSAGES:
            return self._blocked(
                normalized,
                message_kind="help_request",
                reason_code="help_request",
                confidence_score=1.0,
                answer_text=(
                    "Можно спросить, например: какие документы нужны для выплаты, какой срок рассмотрения, "
                    "почему могут отказать, куда обратиться или что может быть положено по жизненной ситуации."
                ),
            )

        if lowered in self._TECHNICAL_MESSAGES:
            return self._blocked(
                normalized,
                message_kind="technical_ping",
                reason_code="technical_ping",
                confidence_score=1.0,
                answer_text="Связь есть. Можешь задать вопрос по мерам социальной поддержки, документам, срокам, отказам или порядку обращения.",
            )

        if lowered in self._GREETING_MESSAGES:
            return self._blocked(
                normalized,
                message_kind="greeting",
                reason_code="greeting_without_domain_question",
                confidence_score=0.95,
                answer_text="Привет. Напиши вопрос по мере социальной поддержки: например про документы, срок, отказ, выплату или порядок обращения.",
            )

        if lowered in self._THANKS_MESSAGES:
            return self._blocked(
                normalized,
                message_kind="thanks_or_ack",
                reason_code="thanks_or_ack_without_domain_question",
                confidence_score=0.95,
                answer_text="Пожалуйста. Если нужно, задай следующий вопрос по мере социальной поддержки.",
            )

        token_count = len(re.findall(r"[а-яa-z0-9ё]+", lowered, flags=re.IGNORECASE))
        if token_count <= 1 and not self._DOMAIN_HINTS_RE.search(lowered):
            return self._blocked(
                normalized,
                message_kind="too_short_non_domain_message",
                reason_code="too_short_without_domain_hint",
                confidence_score=0.85,
                answer_text="Сформулируй, пожалуйста, вопрос чуть подробнее: какая мера поддержки интересует и что нужно узнать.",
                details={"token_count": token_count},
            )

        return MessageGuardResult(
            should_run_rag=True,
            message_kind="domain_or_potential_domain_message",
            reason_code="allowed_to_rag",
            confidence_score=0.75,
            answer_text=None,
            normalized_text=normalized,
            details={
                "channel_code": channel_code,
                "token_count": token_count,
                "has_domain_hint": bool(self._DOMAIN_HINTS_RE.search(lowered)),
            },
        )

    def _blocked(
        self,
        normalized_text: str,
        *,
        message_kind: str,
        reason_code: str,
        confidence_score: float,
        answer_text: str,
        details: Optional[dict[str, Any]] = None,
    ) -> MessageGuardResult:
        return MessageGuardResult(
            should_run_rag=False,
            message_kind=message_kind,
            reason_code=reason_code,
            confidence_score=confidence_score,
            answer_text=answer_text,
            normalized_text=normalized_text,
            details=dict(details or {}),
        )

    def _normalize(self, value: str) -> str:
        text = str(value or "")
        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_mostly_symbols(self, value: str) -> bool:
        if len(value) < 6:
            return False
        chars = [ch for ch in value if not ch.isspace()]
        if not chars:
            return True
        alnum = sum(1 for ch in chars if ch.isalnum())
        return (alnum / max(len(chars), 1)) < 0.35
