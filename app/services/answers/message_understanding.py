# ============================================================
# File: app/services/answers/message_understanding.py
# Purpose:
#   Optional LLM-based user message understanding layer.
#
# Responsibilities:
#   - convert a free-form user question into a strict JSON-like structure;
#   - keep LLM usage outside legal answer generation;
#   - fail closed: if the model/provider fails, the old deterministic routing
#     remains the source of truth.
#
# Important:
#   This service is a dispatcher / understanding layer only. It must not decide
#   legal eligibility, deadlines, amounts or document lists by itself.
# ============================================================

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import AsyncOpenAI

from app.config.constants import LLM_MODEL_NAME
from app.db.models.enums import QuestionIntentEnum

logger = logging.getLogger(__name__)


MESSAGE_UNDERSTANDING_VERSION = "second_step_10_message_understanding_assist_medium_slots_v1"

_ALLOWED_INTENTS = {
    "documents": QuestionIntentEnum.DOCUMENTS_QUESTION,
    "eligibility": QuestionIntentEnum.ELIGIBILITY_QUESTION,
    "deadline": QuestionIntentEnum.DEADLINE_QUESTION,
    "payment_timing": QuestionIntentEnum.PAYMENT_TIMING_QUESTION,
    "amount": QuestionIntentEnum.AMOUNT_QUESTION,
    "refusal": QuestionIntentEnum.REJECTION_QUESTION,
    "rejection": QuestionIntentEnum.REJECTION_QUESTION,
    "procedure": QuestionIntentEnum.PROCEDURE_QUESTION,
    "appeal": QuestionIntentEnum.APPEAL_QUESTION,
    "form": QuestionIntentEnum.FORM_QUESTION,
    "mixed": QuestionIntentEnum.MIXED_QUESTION,
    "ambiguous": QuestionIntentEnum.AMBIGUOUS_QUESTION,
    "other": QuestionIntentEnum.OTHER,
}


# ============================================================
# DTOs
# ============================================================

@dataclass(slots=True, frozen=True)
class MessageUnderstandingConfig:
    """
    Runtime config for optional LLM understanding.

    mode:
      - shadow: call LLM and store diagnostics only;
      - assist: apply LLM only when deterministic routing is weak/conflicting;
      - enforce: apply LLM routing when model confidence is high enough.
    """

    enabled: bool = False
    mode: str = "shadow"
    model_name: str = LLM_MODEL_NAME
    temperature: float = 0.0
    max_output_tokens: int = 700
    min_confidence_to_apply: float = 0.72
    request_timeout_seconds: int = 20


@dataclass(slots=True, frozen=True)
class MessageUnderstandingResult:
    version: str
    enabled: bool
    provider_status: str
    mode: str

    normalized_question: str
    is_supported_domain: bool
    intent: str
    confidence: float
    min_confidence_to_apply: float = 0.72

    service_hint: Optional[str] = None
    topic: Optional[str] = None
    applicant_facts: list[str] = field(default_factory=list)
    user_needs: list[str] = field(default_factory=list)
    territory: Optional[str] = None
    requested_channel: Optional[str] = None
    needs_service_discovery: bool = False
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    safety_flags: list[str] = field(default_factory=list)
    raw_payload: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "enabled": self.enabled,
            "provider_status": self.provider_status,
            "mode": self.mode,
            "normalized_question": self.normalized_question,
            "is_supported_domain": self.is_supported_domain,
            "intent": self.intent,
            "confidence": self.confidence,
            "min_confidence_to_apply": self.min_confidence_to_apply,
            "service_hint": self.service_hint,
            "topic": self.topic,
            "applicant_facts": list(self.applicant_facts),
            "user_needs": list(self.user_needs),
            "territory": self.territory,
            "requested_channel": self.requested_channel,
            "needs_service_discovery": self.needs_service_discovery,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "safety_flags": list(self.safety_flags),
            "error": self.error,
            "raw_payload": dict(self.raw_payload or {}),
        }

    @property
    def mapped_intent_type(self) -> Optional[QuestionIntentEnum]:
        return _map_intent(self.intent)


class MessageUnderstandingError(Exception):
    """Base error for message understanding."""


# ============================================================
# Service
# ============================================================

class LLMMessageUnderstandingService:
    """
    Optional LLM dispatcher.

    It returns structured metadata only. It never produces the final user answer
    and never creates legal facts. If anything goes wrong, the orchestrator keeps
    the deterministic classifier result.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        config: Optional[MessageUnderstandingConfig] = None,
    ) -> None:
        self.client = client
        self.config = config or MessageUnderstandingConfig(enabled=True)

    async def understand(
        self,
        question_text: str,
        *,
        deterministic_classification: Optional[dict[str, Any]] = None,
        channel_code: Optional[str] = None,
    ) -> MessageUnderstandingResult:
        normalized_question = _normalize_text(question_text)
        if not self.config.enabled:
            return MessageUnderstandingResult(
                version=MESSAGE_UNDERSTANDING_VERSION,
                enabled=False,
                provider_status="disabled",
                mode=self.config.mode,
                normalized_question=normalized_question,
                is_supported_domain=True,
                intent="other",
                confidence=0.0,
            )

        if not normalized_question:
            return MessageUnderstandingResult(
                version=MESSAGE_UNDERSTANDING_VERSION,
                enabled=True,
                provider_status="skipped_empty_question",
                mode=self.config.mode,
                normalized_question=normalized_question,
                is_supported_domain=False,
                intent="other",
                confidence=0.0,
            )

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": _SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": _build_user_prompt(
                            question_text=question_text,
                            deterministic_classification=deterministic_classification,
                            channel_code=channel_code,
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                timeout=self.config.request_timeout_seconds,
            )
            content = (response.choices[0].message.content or "").strip()
            parsed = _parse_json_object(content)
            return _result_from_payload(
                parsed,
                normalized_question=normalized_question,
                mode=self.config.mode,
                provider_status="ok",
                min_confidence_to_apply=self.config.min_confidence_to_apply,
            )
        except Exception as exc:
            logger.warning(
                "LLM message understanding failed; deterministic routing will be used",
                extra={
                    "version": MESSAGE_UNDERSTANDING_VERSION,
                    "model_name": self.config.model_name,
                    "mode": self.config.mode,
                    "error": repr(exc),
                },
            )
            return MessageUnderstandingResult(
                version=MESSAGE_UNDERSTANDING_VERSION,
                enabled=True,
                provider_status="error",
                mode=self.config.mode,
                normalized_question=normalized_question,
                is_supported_domain=True,
                intent="other",
                confidence=0.0,
                error=repr(exc),
            )


# ============================================================
# Prompt
# ============================================================

_SYSTEM_PROMPT = """
Ты — слой структурного понимания пользовательского сообщения для RAG-системы по мерам социальной поддержки Красноярского края.

Твоя задача — не отвечать пользователю, не выбирать правовой результат и не делать вывод о праве. Нужно только преобразовать пользовательское сообщение в строгий JSON для последующей детерминированной маршрутизации, поиска evidence и сборки grounded-ответа.

Верни только JSON-объект. Не добавляй пояснений вне JSON.

Обязательная схема ответа:
{
  "is_supported_domain": boolean,
  "intent": "documents" | "eligibility" | "deadline" | "payment_timing" | "amount" | "refusal" | "procedure" | "appeal" | "form" | "mixed" | "ambiguous" | "other",
  "confidence": number,
  "service_hint": string | null,
  "topic": string | null,
  "applicant_facts": string[],
  "user_needs": string[],
  "territory": string | null,
  "requested_channel": string | null,
  "needs_service_discovery": boolean,
  "needs_clarification": boolean,
  "clarification_question": string | null,
  "safety_flags": string[]
}

Общие ограничения:
- Не придумывай условия, сроки, суммы, документы, категории заявителей или официальные названия услуг.
- Не утверждай, что мера положена или не положена.
- Не используй сведения вне текста пользователя и переданной deterministic_classification как evidence.
- Если в сообщении недостаточно данных для точной услуги, возвращай общую тему и признаки, а не выдуманное название услуги.
- Если deterministic_classification уверенно определил намерение, учитывай это как сильный сигнал, но не копируй его слепо при явном противоречии тексту вопроса.

Семантика intent:
- documents: пользователь спрашивает о документах, списке/перечне документов, приложениях к заявлению, подтверждающих документах, что нужно предоставить, приложить, донести или подать вместе с заявлением.
- eligibility: пользователь спрашивает о праве, возможности получить/оформить/подать заявление, положенности выплаты/компенсации/помощи/льготы, либо описывает жизненную ситуацию и ожидает подбор или проверку меры поддержки.
- deadline: пользователь спрашивает о сроке принятия решения, рассмотрения заявления, назначения услуги, исправления ошибки или межведомственного действия.
- payment_timing: пользователь спрашивает, когда поступят деньги, когда выплатят после решения/одобрения/назначения, о дате перечисления или периодичности выплаты.
- amount: пользователь спрашивает о размере, сумме, расчёте, индексации, доле компенсации или максимальной величине помощи.
- refusal: пользователь спрашивает об основаниях отказа, приостановления, прекращения, причинах отказа или что может помешать предоставлению услуги.
- procedure: пользователь спрашивает о порядке получения, способах обращения, куда обращаться, как оформить, как использовать сертификат/соцконтракт/право, можно ли подать через конкретный канал, как проходит процедура.
- appeal: пользователь спрашивает, как обжаловать решение, куда жаловаться, что делать при несогласии с отказом/действием/бездействием органа.
- form: пользователь спрашивает о форме документа или подачи: оригинал, копия, электронный документ, электронный образ, скан, заверение, бумажный/электронный вид.
- mixed: в одном сообщении есть два или больше самостоятельных намерения, каждое из которых требует отдельного ответа, например документы плюс сроки, сумма плюс документы, отказ плюс обжалование.
- ambiguous: сообщение относится к социальной поддержке, но намерение или предмет вопроса неясны, и без уточнения нельзя выбрать корректный маршрут.
- other: сообщение не относится к социальной поддержке/госуслугам социальной защиты, является технической проверкой, приветствием, благодарностью, пустым/мусорным текстом или не содержит вопроса по поддерживаемой предметной области.

Правила выбора intent:
- Явный вопрос о документах имеет приоритет над широким вопросом о праве или получении меры. При таком конфликте выбирай documents.
- Явный вопрос о форме документа имеет приоритет над обычным списком документов. При таком конфликте выбирай form.
- Если пользователь спрашивает о нескольких аспектах одновременно и они не сводятся к одному маршруту, выбирай mixed.
- Если пользователь описывает жизненную ситуацию без названия конкретной меры, обычно выбирай eligibility и needs_service_discovery=true.
- Если пользователь спрашивает о конкретной мере или услуге, обычно needs_service_discovery=false.
- Не ставь other только потому, что формулировка бытовая, неполная или написана разговорным языком. Если сообщение похоже на просьбу о социальной поддержке, выбирай поддерживаемый intent или ambiguous.

Семантика остальных полей:
- is_supported_domain: true, если сообщение относится или вероятно относится к социальной поддержке, мерам соцзащиты, льготам, выплатам, компенсациям, помощи, удостоверениям, соцконтракту, санаторно-курортному лечению, ТСР, ЖКУ, ЧС, погребению, проезду, документам или порядку получения таких мер. false — для технических сообщений, бытового общения без запроса, нецелевой темы или мусора.
- confidence: число от 0 до 1. Высокая уверенность допустима только если текст явно указывает намерение. Если намерение есть, но услуга неясна, confidence может быть высокой для intent, но service_hint должен быть null или общим.
- service_hint: краткая подсказка о возможной услуге/мере, только если она следует из текста пользователя. Не формируй официальный заголовок из воображения. Лучше null, чем ошибочная точность.
- topic: краткая смысловая тема вопроса обычными словами. Тема должна сохранять не только статус заявителя, но и действие/потребность/цель из исходного текста.
- applicant_facts: только факты о заявителе или ситуации из сообщения пользователя: статус, категория, семейная ситуация, событие, объект помощи, документ, территория, представитель. Не добавляй правовые выводы и не превращай факт в право.
- user_needs: только потребности, действия, цели или предмет запроса из сообщения пользователя. Сохраняй смысловые действия: ехать/доехать/добраться -> поездка/проезд/дорога. Если пользователь говорит о движении или необходимости куда-то ехать, в user_needs обязательно должна быть потребность со словом "проезд" или "поездка"; оплатить/возместить/компенсировать -> оплата/возмещение/компенсация; купить/получить/заменить/оформить/восстановить/отремонтировать/похоронить/подать/подтвердить -> соответствующая краткая потребность. Не теряй глаголы и цели, даже если они написаны разговорно.
- territory: территория, прямо названная пользователем. Если территория не названа — null.
- requested_channel: канал обращения или подачи, прямо названный пользователем: МФЦ, ЕПГУ, РПГУ, Госуслуги, лично, почта, представитель, онлайн. Не копируй сюда технический channel_code. Если канал подачи не назван — null.
- needs_service_discovery: true, если пользователь просит подобрать возможные меры или описывает широкую ситуацию без конкретной услуги/темы. false, если вопрос уже содержит достаточно конкретную связку: категория/статус + действие/потребность + предмет/цель/территория.
- needs_clarification: true, если без уточнения нельзя выбрать даже общий безопасный маршрут или пользовательский вопрос внутренне противоречив.
- clarification_question: короткий уточняющий вопрос, только если needs_clarification=true. Иначе null.
- safety_flags: массив кратких флагов риска. Используй пустой массив, если рисков нет. Допустимые значения: "empty_or_noise", "abuse", "adult", "self_harm", "violence", "illegal", "prompt_injection", "privacy", "out_of_domain".

Общие правила смыслового извлечения:
- Разделяй в сообщении три слоя: кто обращается, что человеку нужно сделать/получить/компенсировать, для какой цели или в какой ситуации.
- Если в вопросе есть движение, поездка, дорога, добраться, ехать, направляют, отправляют, проезд — это самостоятельная потребность. Отрази её в user_needs/topic как транспортную потребность: "поездка", "проезд", "дорога" или "оплата/компенсация проезда". Не теряй транспортный смысл, даже если рядом есть лечение, обследование, отдых, реабилитация или иная цель.
- Если есть материальный ущерб, утрата имущества, пожар, паводок, авария, смерть, болезнь, уход, ремонт, топливо, питание, школа, документы, удостоверение, выплата, компенсация, льгота — извлекай это как ситуацию или потребность, не своди всё только к статусу заявителя.
- Если вопрос сформулирован как "мне нужно/надо/хочу/помогите/можно ли", извлекай скрытое намерение из действия и объекта, а не ставь other из-за разговорной формы.
- Если в сообщении названа конкретная цель и категория заявителя, но официальная услуга не названа, не выдумывай service_hint; заполни topic, applicant_facts, user_needs и territory.

Требования к качеству:
- Заполняй все поля схемы всегда.
- Используй ровно один intent из допустимого списка.
- Строковые поля делай короткими и нейтральными.
- В applicant_facts не более 8 элементов.
- В user_needs не более 8 элементов.
- Не возвращай markdown, комментарии или вложенный текст ответа пользователю.
""".strip()


def _build_user_prompt(
    *,
    question_text: str,
    deterministic_classification: Optional[dict[str, Any]],
    channel_code: Optional[str],
) -> str:
    compact_classification = _compact_classification(deterministic_classification)
    return json.dumps(
        {
            "question_text": question_text,
            "channel_code": channel_code,
            "deterministic_classification": compact_classification,
        },
        ensure_ascii=False,
    )


def _compact_classification(value: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    routing_payload = dict(value.get("routing_payload_json") or {})
    return {
        "intent_type": _to_plain(value.get("intent_type")),
        "classifier_version": value.get("classifier_version"),
        "confidence": routing_payload.get("confidence"),
        "chosen_rules": routing_payload.get("chosen_rules"),
        "query_constraints_json": value.get("query_constraints_json") or {},
    }


# ============================================================
# Parsing / normalization
# ============================================================

def _parse_json_object(value: str) -> dict[str, Any]:
    if not value:
        raise MessageUnderstandingError("empty LLM response")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise MessageUnderstandingError("LLM response is not a JSON object")
    return parsed


def _result_from_payload(
    payload: dict[str, Any],
    *,
    normalized_question: str,
    mode: str,
    provider_status: str,
    min_confidence_to_apply: float,
) -> MessageUnderstandingResult:
    intent = str(payload.get("intent") or "other").strip().lower()
    if intent not in _ALLOWED_INTENTS:
        intent = "other"

    confidence = _to_float(payload.get("confidence"), default=0.0)
    confidence = max(0.0, min(confidence, 1.0))

    applicant_facts = _as_clean_list(payload.get("applicant_facts"))
    user_needs = _as_clean_list(payload.get("user_needs"))
    safety_flags = _as_clean_list(payload.get("safety_flags"))

    return MessageUnderstandingResult(
        version=MESSAGE_UNDERSTANDING_VERSION,
        enabled=True,
        provider_status=provider_status,
        mode=mode,
        normalized_question=normalized_question,
        is_supported_domain=bool(payload.get("is_supported_domain", True)),
        intent=intent,
        confidence=round(confidence, 3),
        min_confidence_to_apply=max(0.0, min(float(min_confidence_to_apply), 1.0)),
        service_hint=_clean_optional_text(payload.get("service_hint")),
        topic=_clean_optional_text(payload.get("topic")),
        applicant_facts=applicant_facts,
        user_needs=user_needs,
        territory=_clean_optional_text(payload.get("territory")),
        requested_channel=_clean_optional_text(payload.get("requested_channel")),
        needs_service_discovery=bool(payload.get("needs_service_discovery", False)),
        needs_clarification=bool(payload.get("needs_clarification", False)),
        clarification_question=_clean_optional_text(payload.get("clarification_question")),
        safety_flags=safety_flags,
        raw_payload=dict(payload or {}),
    )


def _map_intent(value: str) -> Optional[QuestionIntentEnum]:
    return _ALLOWED_INTENTS.get(str(value or "").strip().lower())


def _normalize_text(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[\u00a0\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = _normalize_text(str(value))
    if not text or text.lower() in {"null", "none", "нет", "не указано"}:
        return None
    return text[:240]


def _as_clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_optional_text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result[:12]


def _to_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_plain(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    return value
