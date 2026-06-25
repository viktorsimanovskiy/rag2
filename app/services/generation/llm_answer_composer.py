# ============================================================
# File: app/services/generation/llm_answer_composer.py
# Purpose:
#   Optional LLM answer composer for rewriting already-grounded answers.
#
# Version:
#   second_step_53_llm_answer_composer_structure_preservation_v1
#
# Design constraints:
#   - LLM is an editor/composer only;
#   - LLM must not create legal facts, conditions, deadlines, amounts or documents;
#   - deterministic builders and retrieved evidence remain the source of truth;
#   - if the composer is uncertain or violates grounding checks, the caller keeps
#     the deterministic answer.
# ============================================================

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import AsyncOpenAI

from app.config.constants import LLM_MODEL_NAME

logger = logging.getLogger(__name__)

LLM_ANSWER_COMPOSER_VERSION = "second_step_53_llm_answer_composer_structure_preservation_v1"

_SUPPORTED_ANSWER_MODES = {"direct_structured", "grounded_narrative"}
_SKIP_ANSWER_MODES = {"safe_no_answer"}


@dataclass(slots=True, frozen=True)
class LLMAnswerComposerConfig:
    """Runtime config for optional answer composer.

    mode:
      - shadow: call the model and return diagnostics only;
      - assist: can be used by caller to replace answer when status is ok;
      - disabled: no model call.
    """

    enabled: bool = False
    mode: str = "shadow"
    model_name: str = LLM_MODEL_NAME
    temperature: float = 0.0
    max_output_tokens: int = 1000
    request_timeout_seconds: int = 35
    max_input_chars: int = 12000
    max_output_chars: int = 4500


@dataclass(slots=True, frozen=True)
class AnswerCitationForComposer:
    index: int
    source_type: str
    display_label: str
    citation_text: str
    document_name: Optional[str] = None
    metadata_json: dict[str, Any] = field(default_factory=dict)

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "source_type": self.source_type,
            "display_label": self.display_label,
            "citation_text": self.citation_text,
            "document_name": self.document_name,
            "metadata_json": _compact_metadata(self.metadata_json),
        }


@dataclass(slots=True, frozen=True)
class LLMAnswerComposerInput:
    question_text: str
    deterministic_answer_text: str
    answer_mode: str
    citations: list[AnswerCitationForComposer] = field(default_factory=list)
    service_resolution: dict[str, Any] = field(default_factory=dict)
    evidence_metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class LLMAnswerComposerResult:
    version: str
    enabled: bool
    mode: str
    provider_status: str
    status: str

    answer_mode: str
    should_replace_answer: bool
    final_answer_text: str
    composed_answer_text: Optional[str] = None
    changes_summary: list[str] = field(default_factory=list)
    used_source_indexes: list[int] = field(default_factory=list)
    uncertain_parts: list[str] = field(default_factory=list)
    grounding_violations: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    raw_payload: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "enabled": self.enabled,
            "mode": self.mode,
            "provider_status": self.provider_status,
            "status": self.status,
            "answer_mode": self.answer_mode,
            "should_replace_answer": self.should_replace_answer,
            "final_answer_text": self.final_answer_text,
            "composed_answer_text": self.composed_answer_text,
            "changes_summary": list(self.changes_summary),
            "used_source_indexes": list(self.used_source_indexes),
            "uncertain_parts": list(self.uncertain_parts),
            "grounding_violations": list(self.grounding_violations),
            "validation_warnings": list(self.validation_warnings),
            "error": self.error,
            "raw_payload": dict(self.raw_payload or {}),
        }


class LLMAnswerComposerError(Exception):
    """Base composer error."""


class LLMAnswerComposerService:
    """Optional LLM composer for already-grounded answers.

    The service never performs retrieval and never decides the legal answer.
    It can only rewrite the deterministic answer into clearer Russian while
    preserving the facts already present in the deterministic answer/citations.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        config: Optional[LLMAnswerComposerConfig] = None,
    ) -> None:
        self.client = client
        self.config = config or LLMAnswerComposerConfig(enabled=True)

    async def compose(self, payload: LLMAnswerComposerInput) -> LLMAnswerComposerResult:
        normalized_mode = _normalize_answer_mode(payload.answer_mode)
        deterministic_answer = _clean_answer_text(payload.deterministic_answer_text)

        if not self.config.enabled or self.config.mode == "disabled":
            return _skip_result(
                payload=payload,
                mode=self.config.mode,
                deterministic_answer=deterministic_answer,
                reason="disabled",
            )

        if not deterministic_answer:
            return _skip_result(
                payload=payload,
                mode=self.config.mode,
                deterministic_answer=deterministic_answer,
                reason="empty_deterministic_answer",
            )

        if normalized_mode in _SKIP_ANSWER_MODES:
            return _skip_result(
                payload=payload,
                mode=self.config.mode,
                deterministic_answer=deterministic_answer,
                reason="safe_no_answer_not_composed",
            )

        if normalized_mode not in _SUPPORTED_ANSWER_MODES:
            return _skip_result(
                payload=payload,
                mode=self.config.mode,
                deterministic_answer=deterministic_answer,
                reason=f"unsupported_answer_mode:{normalized_mode or 'empty'}",
            )

        if not payload.citations:
            return _skip_result(
                payload=payload,
                mode=self.config.mode,
                deterministic_answer=deterministic_answer,
                reason="no_citations",
            )

        mismatch_reason = _detect_question_answer_mismatch(
            question_text=payload.question_text,
            deterministic_answer_text=deterministic_answer,
            answer_mode=normalized_mode,
        )
        if mismatch_reason:
            return _skip_result(
                payload=payload,
                mode=self.config.mode,
                deterministic_answer=deterministic_answer,
                reason=mismatch_reason,
            )

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(payload, self.config)},
                ],
                response_format={"type": "json_object"},
                timeout=self.config.request_timeout_seconds,
            )
            content = (response.choices[0].message.content or "").strip()
            parsed = _parse_json_object(content)
            return _result_from_model_payload(
                model_payload=parsed,
                input_payload=payload,
                mode=self.config.mode,
                deterministic_answer=deterministic_answer,
                max_output_chars=self.config.max_output_chars,
            )
        except Exception as exc:
            logger.warning(
                "LLM answer composer failed; deterministic answer will be used",
                extra={
                    "version": LLM_ANSWER_COMPOSER_VERSION,
                    "model_name": self.config.model_name,
                    "mode": self.config.mode,
                    "error": repr(exc),
                },
            )
            return LLMAnswerComposerResult(
                version=LLM_ANSWER_COMPOSER_VERSION,
                enabled=True,
                mode=self.config.mode,
                provider_status="error",
                status="fallback",
                answer_mode=normalized_mode,
                should_replace_answer=False,
                final_answer_text=deterministic_answer,
                grounding_violations=["provider_error"],
                error=repr(exc),
            )


_SYSTEM_PROMPT = """
Ты — редактор ответа RAG-системы по мерам социальной поддержки Красноярского края.

Твоя задача: сделать уже подготовленный grounded-ответ понятнее и компактнее для пользователя, сохраняя его проверяемую структуру.

Строгие запреты:
- не добавляй новые правовые условия;
- не добавляй новые сроки, суммы, документы, категории заявителей или основания отказа;
- не делай вывод о праве заявителя, если такой вывод прямо не был в исходном grounded-ответе;
- не используй внешние знания;
- не исправляй нормативный смысл по памяти;
- не придумывай ссылки, каналы подачи, названия услуг или реквизиты НПА.

Разрешено:
- менять порядок предложений;
- сокращать повторы;
- делать формулировки более простыми;
- группировать уже имеющиеся пункты;
- не превращать маркированный список документов в новый нумерованный список, если это создаёт новые номера пунктов;
- сохранять предупреждение о зависимости итогового вывода от условий регламента и документов;
- сохранять списки списками: если в исходном ответе есть перечень категорий, документов или условий, не сжимай его в длинный абзац;
- использовать короткие абзацы и маркированные пункты "•" для перечислений;
- не ухудшать читаемость ради краткости.

Источники истины:
1. deterministic_answer_text — основной готовый ответ системы;
2. citations — проверочные фрагменты evidence;
3. service_resolution — только название/контекст найденной услуги, если оно уже есть.

Если для улучшения ответа не хватает данных, верни исходный ответ без изменений.

Верни строго JSON-объект:
{
  "composed_answer_text": "строка",
  "changes_summary": ["кратко что изменено"],
  "used_source_indexes": [0, 1, 2],
  "uncertain_parts": []
}

Индекс 0 означает исходный deterministic_answer_text. Индексы citations начинаются с 1.
""".strip()


def _build_user_prompt(payload: LLMAnswerComposerInput, config: LLMAnswerComposerConfig) -> str:
    prompt_payload = {
        "question_text": _clip(payload.question_text, 1500),
        "answer_mode": _normalize_answer_mode(payload.answer_mode),
        "deterministic_answer_text": _clip(payload.deterministic_answer_text, 6500),
        "citations": [item.to_prompt_payload() for item in payload.citations[:8]],
        "service_resolution": _compact_service_resolution(payload.service_resolution),
        "evidence_metrics": _compact_evidence_metrics(payload.evidence_metrics),
        "warnings": [str(item)[:300] for item in (payload.warnings or [])[:5]],
        "output_rules": {
            "max_chars": config.max_output_chars,
            "must_preserve_legal_meaning": True,
            "must_not_introduce_new_facts": True,
            "if_unsure_return_original": True,
            "preserve_list_structure": True,
            "prefer_short_paragraphs": True,
        },
    }
    text = json.dumps(prompt_payload, ensure_ascii=False)
    return _clip(text, config.max_input_chars)


def _result_from_model_payload(
    *,
    model_payload: dict[str, Any],
    input_payload: LLMAnswerComposerInput,
    mode: str,
    deterministic_answer: str,
    max_output_chars: int,
) -> LLMAnswerComposerResult:
    answer_mode = _normalize_answer_mode(input_payload.answer_mode)
    composed = _clean_answer_text(model_payload.get("composed_answer_text"))
    changes_summary = _string_list(model_payload.get("changes_summary"), limit=8, item_limit=240)
    uncertain_parts = _string_list(model_payload.get("uncertain_parts"), limit=8, item_limit=240)
    used_source_indexes = _int_list(model_payload.get("used_source_indexes"), limit=20)

    validation_warnings: list[str] = []
    violations: list[str] = []

    if not composed:
        violations.append("empty_composed_answer")
    if len(composed) > max_output_chars:
        violations.append("composed_answer_too_long")
    if len(composed) > max(len(deterministic_answer) * 1.25, len(deterministic_answer) + 700):
        validation_warnings.append("composed_answer_is_much_longer_than_source")

    source_text = _source_text_for_validation(input_payload, deterministic_answer)
    violations.extend(_find_number_violations(composed, source_text))
    violations.extend(_find_date_violations(composed, source_text))
    violations.extend(_find_forbidden_phrase_violations(composed))
    violations.extend(_find_structure_violations(composed, deterministic_answer))
    validation_warnings.extend(_find_structure_warnings(composed, deterministic_answer))

    source_count = 1 + len(input_payload.citations)
    invalid_indexes = [idx for idx in used_source_indexes if idx < 0 or idx > source_count]
    if invalid_indexes:
        violations.append("invalid_used_source_indexes:" + ",".join(str(i) for i in invalid_indexes))

    if violations:
        return LLMAnswerComposerResult(
            version=LLM_ANSWER_COMPOSER_VERSION,
            enabled=True,
            mode=mode,
            provider_status="ok",
            status="rejected",
            answer_mode=answer_mode,
            should_replace_answer=False,
            final_answer_text=deterministic_answer,
            composed_answer_text=composed or None,
            changes_summary=changes_summary,
            used_source_indexes=used_source_indexes,
            uncertain_parts=uncertain_parts,
            grounding_violations=violations,
            validation_warnings=validation_warnings,
            raw_payload=dict(model_payload),
        )

    return LLMAnswerComposerResult(
        version=LLM_ANSWER_COMPOSER_VERSION,
        enabled=True,
        mode=mode,
        provider_status="ok",
        status="ok",
        answer_mode=answer_mode,
        should_replace_answer=(mode == "assist"),
        final_answer_text=composed,
        composed_answer_text=composed,
        changes_summary=changes_summary,
        used_source_indexes=used_source_indexes,
        uncertain_parts=uncertain_parts,
        grounding_violations=[],
        validation_warnings=validation_warnings,
        raw_payload=dict(model_payload),
    )


def _skip_result(
    *,
    payload: LLMAnswerComposerInput,
    mode: str,
    deterministic_answer: str,
    reason: str,
) -> LLMAnswerComposerResult:
    return LLMAnswerComposerResult(
        version=LLM_ANSWER_COMPOSER_VERSION,
        enabled=(mode != "disabled"),
        mode=mode,
        provider_status="not_called",
        status="skipped",
        answer_mode=_normalize_answer_mode(payload.answer_mode),
        should_replace_answer=False,
        final_answer_text=deterministic_answer,
        validation_warnings=[reason],
    )


def _parse_json_object(value: str) -> dict[str, Any]:
    if not value:
        raise LLMAnswerComposerError("empty LLM response")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise LLMAnswerComposerError("LLM response is not a JSON object")
    return parsed


def citations_from_api_response(response_payload: dict[str, Any]) -> list[AnswerCitationForComposer]:
    raw_citations = response_payload.get("citations")
    if not isinstance(raw_citations, list):
        return []
    return _citations_from_raw_list(raw_citations)


def input_from_api_response(question_text: str, response_payload: dict[str, Any]) -> LLMAnswerComposerInput:
    debug_payload = response_payload.get("debug") if isinstance(response_payload.get("debug"), dict) else {}
    answer_payload = debug_payload.get("answer_payload_json") if isinstance(debug_payload, dict) else {}
    if not isinstance(answer_payload, dict):
        answer_payload = {}
    return LLMAnswerComposerInput(
        question_text=question_text,
        deterministic_answer_text=str(response_payload.get("answer_text") or ""),
        answer_mode=str(response_payload.get("answer_mode") or ""),
        citations=citations_from_api_response(response_payload),
        service_resolution=dict(response_payload.get("service_resolution") or {}),
        evidence_metrics=dict(answer_payload.get("evidence_metrics") or {}),
        warnings=[str(item) for item in (response_payload.get("warnings") or []) if str(item).strip()],
    )


def citations_from_generation_result(generation_result: Any) -> list[AnswerCitationForComposer]:
    raw_citations = getattr(generation_result, "citations_json", None)
    if not isinstance(raw_citations, list):
        return []
    return _citations_from_raw_list(raw_citations)


def input_from_generation_result(question_text: str, generation_result: Any) -> LLMAnswerComposerInput:
    answer_payload_json = getattr(generation_result, "answer_payload_json", None)
    if not isinstance(answer_payload_json, dict):
        answer_payload_json = {}

    answer_mode = getattr(generation_result, "answer_mode", "")
    answer_mode_value = getattr(answer_mode, "value", answer_mode)

    return LLMAnswerComposerInput(
        question_text=question_text,
        deterministic_answer_text=str(getattr(generation_result, "answer_text", "") or ""),
        answer_mode=str(answer_mode_value or ""),
        citations=citations_from_generation_result(generation_result),
        service_resolution=_extract_service_resolution_from_answer_payload(answer_payload_json),
        evidence_metrics=_extract_evidence_metrics_from_answer_payload(answer_payload_json),
        warnings=_extract_warnings_from_answer_payload(answer_payload_json),
    )


def _citations_from_raw_list(raw_citations: list[Any]) -> list[AnswerCitationForComposer]:
    result: list[AnswerCitationForComposer] = []
    for index, item in enumerate(raw_citations, start=1):
        if not isinstance(item, dict):
            continue
        citation_text = _clean_text(item.get("citation_text"))
        display_label = _clean_text(item.get("display_label"))
        if not citation_text and not display_label:
            continue
        result.append(
            AnswerCitationForComposer(
                index=index,
                source_type=_clean_text(item.get("source_type")) or "unknown",
                display_label=display_label,
                citation_text=citation_text,
                document_name=_clean_text(item.get("document_name")) or None,
                metadata_json=dict(item.get("metadata_json") or {}),
            )
        )
    return result


def _extract_service_resolution_from_answer_payload(answer_payload_json: dict[str, Any]) -> dict[str, Any]:
    direct = answer_payload_json.get("service_resolution")
    if isinstance(direct, dict):
        return dict(direct)

    runtime_payload = answer_payload_json.get("runtime_answer_service_runtime_payload")
    if isinstance(runtime_payload, dict):
        debug_payload = runtime_payload.get("debug_payload_json")
        if isinstance(debug_payload, dict):
            service_resolution = debug_payload.get("service_resolution")
            if isinstance(service_resolution, dict):
                return dict(service_resolution)

    runtime_info = answer_payload_json.get("runtime_answer_service")
    if isinstance(runtime_info, dict):
        debug_payload = runtime_info.get("debug_payload_json")
        if isinstance(debug_payload, dict):
            service_resolution = debug_payload.get("service_resolution")
            if isinstance(service_resolution, dict):
                return dict(service_resolution)

    return {}


def _extract_evidence_metrics_from_answer_payload(answer_payload_json: dict[str, Any]) -> dict[str, Any]:
    direct = answer_payload_json.get("evidence_metrics")
    if isinstance(direct, dict):
        return dict(direct)

    runtime_payload = answer_payload_json.get("runtime_answer_service_runtime_payload")
    if isinstance(runtime_payload, dict):
        debug_payload = runtime_payload.get("debug_payload_json")
        if isinstance(debug_payload, dict):
            metrics = debug_payload.get("evidence_metrics")
            if isinstance(metrics, dict):
                return dict(metrics)

    return {}


def _extract_warnings_from_answer_payload(answer_payload_json: dict[str, Any]) -> list[str]:
    raw = answer_payload_json.get("warnings")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item).strip()]


def _source_text_for_validation(payload: LLMAnswerComposerInput, deterministic_answer: str) -> str:
    chunks = [deterministic_answer]
    for item in payload.citations:
        chunks.append(item.display_label)
        chunks.append(item.citation_text)
        if item.document_name:
            chunks.append(item.document_name)
    service_name = _compact_service_resolution(payload.service_resolution).get("service_name")
    if service_name:
        chunks.append(str(service_name))
    return "\n".join(chunk for chunk in chunks if chunk)


def _detect_question_answer_mismatch(*, question_text: str, deterministic_answer_text: str, answer_mode: str) -> str:
    """Conservative guard: do not compose answers that look off-topic.

    The composer is not allowed to repair retrieval/generation mistakes. If the
    deterministic answer does not appear to address a high-risk question type,
    the service skips the LLM call and keeps the original answer.
    """

    question = _normalize_for_match(question_text)
    answer = _normalize_for_match(deterministic_answer_text)
    mode = _normalize_answer_mode(answer_mode)

    if not question or not answer:
        return ""

    if mode == "safe_no_answer":
        return ""

    asks_amount = _contains_any(question, (
        "какая сумма",
        "какой размер",
        "размер выплаты",
        "сумма выплаты",
        "сколько платят",
        "сколько выплат",
        "сколько денег",
    ))
    if asks_amount and not _looks_like_amount_answer(answer):
        return "answer_question_mismatch:amount_without_amount"

    asks_documents = _contains_any(question, ("какие документы", "документы для", "список документов", "перечень документов"))
    if asks_documents and not _contains_any(answer, ("документ", "заявление", "паспорт", "копия", "удостовер")):
        return "answer_question_mismatch:documents_without_documents"

    asks_deadline = _contains_any(question, ("срок", "когда примут", "когда решение", "когда выплат"))
    if asks_deadline and not _contains_any(answer, ("дн", "срок", "рабоч", "месяц", "год", "решени", "выплат")):
        return "answer_question_mismatch:deadline_without_deadline"

    asks_refusal = _contains_any(question, ("отказ", "отказать", "основания отказа", "причины отказа"))
    if asks_refusal and not _contains_any(answer, ("отказ", "основан", "причин", "непредстав", "несоответ")):
        return "answer_question_mismatch:refusal_without_refusal"

    return ""


def _looks_like_amount_answer(answer: str) -> bool:
    if _contains_any(answer, ("размер", "сумм", "руб", "процент", "величин", "прожиточн", "доход")):
        return True
    return bool(re.search(r"\b\d+(?:[.,]\d+)?\s*(?:руб|рубл|%)", answer, flags=re.IGNORECASE))


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)


def _find_number_violations(answer_text: str, source_text: str) -> list[str]:
    if not answer_text:
        return []
    answer_numbers = _extract_number_like_tokens(answer_text)
    source_numbers = _extract_number_like_tokens(source_text)
    missing = sorted(token for token in answer_numbers if token not in source_numbers)
    return ["new_number_tokens:" + ",".join(missing[:12])] if missing else []


def _find_date_violations(answer_text: str, source_text: str) -> list[str]:
    answer_dates = _extract_date_like_tokens(answer_text)
    source_dates = _extract_date_like_tokens(source_text)
    missing = sorted(token for token in answer_dates if token not in source_dates)
    return ["new_date_tokens:" + ",".join(missing[:12])] if missing else []


def _extract_number_like_tokens(text: str) -> set[str]:
    normalized = str(text or "")
    tokens = set()
    pattern = r"(?<![\w])\d+(?:[.,]\d+)?\s*(?:%|процент[а-я]*|руб(?:\.|л[яеёй]*)?)?"
    for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
        raw = match.group(0)
        token = re.sub(r"\s+", "", raw.lower())
        if not token:
            continue
        if _is_plain_list_marker(normalized, match.start(), match.end(), token):
            continue
        tokens.add(token)
    return tokens


def _is_plain_list_marker(text: str, start: int, end: int, token: str) -> bool:
    # Ignore numbering introduced only to format a list: "1. Заявление", "2) Паспорт".
    # Do not ignore legal references like "пункт 10", dates, sums, percentages or decimals.
    if not re.fullmatch(r"\d{1,2}", token):
        return False
    if end >= len(text):
        return False
    next_char = text[end]
    if next_char not in {".", ")"}:
        return False
    after = text[end + 1] if end + 1 < len(text) else ""
    if after and not after.isspace():
        return False
    before = text[start - 1] if start > 0 else "\n"
    if before not in {"\n", "\r", " ", "\t", ";", ":"}:
        return False
    return True


def _extract_date_like_tokens(text: str) -> set[str]:
    result = set()
    for match in re.finditer(r"\b\d{1,2}[.]\d{1,2}[.]\d{2,4}\b", str(text or "")):
        result.add(match.group(0))
    return result



def _find_structure_violations(answer_text: str, source_answer_text: str) -> list[str]:
    """Reject rewrites that make a structured answer less usable.

    This is not a legal validation. It protects the current UX rule: LLM may
    simplify wording, but must not collapse document/category lists into a long
    paragraph and must preserve the caution that right to a measure depends on
    the regulation and documents.
    """

    violations: list[str] = []
    source_bullets = _count_list_markers(source_answer_text)
    answer_bullets = _count_list_markers(answer_text)

    if source_bullets >= 5 and answer_bullets < 2:
        violations.append("list_structure_lost")

    if _source_has_right_caution(source_answer_text) and not _answer_has_right_caution(answer_text):
        violations.append("right_caution_lost")

    return violations


def _find_structure_warnings(answer_text: str, source_answer_text: str) -> list[str]:
    warnings: list[str] = []
    if _max_paragraph_len(answer_text) > 900:
        warnings.append("long_paragraph_in_composed_answer")
    if _count_list_markers(source_answer_text) >= 3 and _count_list_markers(answer_text) < _count_list_markers(source_answer_text) // 3:
        warnings.append("list_structure_reduced")
    return warnings


def _count_list_markers(text: str) -> int:
    value = str(text or "")
    count = value.count("•")
    count += len(re.findall(r"(?:^|\n)\s*[-–]\s+\S", value))
    count += len(re.findall(r"(?:^|\n)\s*\d{1,2}[.)]\s+\S", value))
    return count


def _max_paragraph_len(text: str) -> int:
    paragraphs = [part.strip() for part in str(text or "").split("\n\n") if part.strip()]
    if not paragraphs:
        return 0
    return max(len(part) for part in paragraphs)


def _source_has_right_caution(text: str) -> bool:
    normalized = _normalize_for_match(text)
    return _contains_any(normalized, (
        "точный вывод о праве",
        "право на меру зависит",
        "право на выплату зависит",
        "зависит от всех условий регламента",
        "зависит от условий регламента",
        "зависит от регламента",
    ))


def _answer_has_right_caution(text: str) -> bool:
    normalized = _normalize_for_match(text)
    return _contains_any(normalized, (
        "зависит от условий регламента",
        "зависит от регламента",
        "зависит от всех условий",
        "точный вывод о праве",
        "право на меру зависит",
        "право на выплату зависит",
        "итоговое право зависит",
    ))


def _find_forbidden_phrase_violations(answer_text: str) -> list[str]:
    normalized = _normalize_for_match(answer_text)
    forbidden_patterns = {
        "definitive_right_claim": r"\b(?:вам|тебе)\s+положен[аоы]?\b",
        "guaranteed_payment_claim": r"\b(?:точно|гарантированно)\s+(?:получите|назначат|предоставят)\b",
        "invented_appeal_to_law": r"\bпо\s+закону\s+(?:вам|тебе)\s+обязаны\b",
    }
    violations = []
    for code, pattern in forbidden_patterns.items():
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            violations.append(code)
    return violations


def _compact_service_resolution(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    service_name = (
        _clean_text(value.get("service_name_short"))
        or _clean_text(value.get("service_name_full"))
        or _clean_text(value.get("service_name"))
    )
    return {
        "resolution_status": _clean_text(value.get("resolution_status")),
        "confidence": _clean_text(value.get("confidence")),
        "service_key": _clean_text(value.get("service_key")),
        "service_name": _clip(service_name, 500),
    }


def _compact_evidence_metrics(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keys = (
        "evidence_quality",
        "guard_reason",
        "service_filter_applied",
        "selected_document_ids_count",
        "selected_row_ids_count",
        "selected_fact_ids_count",
        "strong_candidate_count",
        "top_document_share",
    )
    return {key: value.get(key) for key in keys if key in value}


def _compact_metadata(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keys = ("table_title", "table_number", "row_order", "rerank_score")
    return {key: value.get(key) for key in keys if key in value}


def _string_list(value: Any, *, limit: int, item_limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value[:limit]:
        text = _clip(_clean_text(item), item_limit)
        if text:
            result.append(text)
    return result


def _int_list(value: Any, *, limit: int) -> list[int]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value[:limit]:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result



def _clean_answer_text(value: Any) -> str:
    """Normalize answer text while preserving paragraphs and bullet lists."""

    text = str(value or "").replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = text.split("\n")
    normalized_lines = [" ".join(line.split()).strip() for line in raw_lines]

    result: list[str] = []
    previous_blank = False
    for line in normalized_lines:
        if not line:
            if result and not previous_blank:
                result.append("")
            previous_blank = True
            continue
        result.append(line)
        previous_blank = False

    return "\n".join(result).strip()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split()).strip()


def _clip(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _normalize_answer_mode(value: Any) -> str:
    return _clean_text(value).lower()


def _normalize_for_match(value: str) -> str:
    return " ".join(str(value or "").replace("ё", "е").lower().split())
