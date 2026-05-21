# ============================================================
# File: app/services/answers/intent_classifier.py
# Purpose:
#   Lightweight rule-based classifier of user question intent.
#
# Notes:
#   - no external model call;
#   - deterministic and fast;
#   - intended as the first live runtime classifier before n8n/API demo;
#   - broad questions like "что мне положено" remain eligibility_question,
#     but receive requires_service_discovery=true in routing metadata.
# ============================================================

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.db.models.enums import QuestionIntentEnum


CLASSIFIER_VERSION = "rule_based_intent_classifier_v1"


@dataclass(slots=True)
class IntentRule:
    """
    One weighted rule for intent classification.
    """

    code: str
    intent_type: QuestionIntentEnum
    weight: int
    patterns: tuple[str, ...]
    stop_patterns: tuple[str, ...] = ()
    adds_constraints: dict[str, Any] = field(default_factory=dict)
    adds_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IntentRuleMatch:
    """
    Result of one matched rule.
    """

    code: str
    intent_type: QuestionIntentEnum
    weight: int
    matched_patterns: list[str]
    adds_constraints: dict[str, Any] = field(default_factory=dict)
    adds_payload: dict[str, Any] = field(default_factory=dict)


class RuleBasedIntentClassifier:
    """
    Simple deterministic classifier for live runtime.

    The classifier deliberately keeps the current DB enum unchanged.
    A broad request for possible measures is returned as ELIGIBILITY_QUESTION
    and marked with requires_service_discovery=true in query constraints.
    """

    def __init__(self, *, version: str = CLASSIFIER_VERSION) -> None:
        self.version = version
        self._rules = _build_rules()

    async def classify(self, question_text: str) -> dict[str, Any]:
        """
        Classify normalized or raw user question text.
        """
        normalized_text = normalize_question_text(question_text)
        matches = self._collect_matches(normalized_text)

        if not matches:
            return self._build_result(
                question_text=normalized_text,
                intent_type=QuestionIntentEnum.OTHER,
                confidence=0.15,
                matched_rules=[],
                query_constraints_json={},
                routing_payload_json={
                    "source": self.version,
                    "note": "правила не сработали",
                },
            )

        chosen_intent = self._choose_intent(matches)
        intent_matches = [match for match in matches if match.intent_type == chosen_intent]
        total_score = sum(match.weight for match in matches)
        chosen_score = sum(match.weight for match in intent_matches)
        confidence = self._estimate_confidence(chosen_score, total_score, len(intent_matches))

        query_constraints_json: dict[str, Any] = {}
        routing_payload_json: dict[str, Any] = {
            "source": self.version,
            "matched_rules": [
                {
                    "code": match.code,
                    "intent_type": match.intent_type.value,
                    "weight": match.weight,
                    "matched_patterns": match.matched_patterns,
                }
                for match in matches
            ],
            "chosen_rules": [match.code for match in intent_matches],
            "confidence": confidence,
        }

        for match in intent_matches:
            _merge_dict(query_constraints_json, match.adds_constraints)
            _merge_dict(routing_payload_json, match.adds_payload)

        if chosen_intent == QuestionIntentEnum.PAYMENT_TIMING_QUESTION:
            query_constraints_json.setdefault("deadline_focus", "payment")
        elif chosen_intent == QuestionIntentEnum.DEADLINE_QUESTION:
            query_constraints_json.setdefault("deadline_focus", "decision")

        return self._build_result(
            question_text=normalized_text,
            intent_type=chosen_intent,
            confidence=confidence,
            matched_rules=intent_matches,
            query_constraints_json=query_constraints_json,
            routing_payload_json=routing_payload_json,
        )

    def _collect_matches(self, normalized_text: str) -> list[IntentRuleMatch]:
        matches: list[IntentRuleMatch] = []
        for rule in self._rules:
            if _matches_any(normalized_text, rule.stop_patterns):
                continue

            matched_patterns = [
                pattern
                for pattern in rule.patterns
                if re.search(pattern, normalized_text, flags=re.IGNORECASE)
            ]
            if matched_patterns:
                matches.append(
                    IntentRuleMatch(
                        code=rule.code,
                        intent_type=rule.intent_type,
                        weight=rule.weight,
                        matched_patterns=matched_patterns,
                        adds_constraints=dict(rule.adds_constraints),
                        adds_payload=dict(rule.adds_payload),
                    )
                )
        return matches

    @staticmethod
    def _choose_intent(matches: list[IntentRuleMatch]) -> QuestionIntentEnum:
        scores: dict[QuestionIntentEnum, int] = {}
        for match in matches:
            scores[match.intent_type] = scores.get(match.intent_type, 0) + match.weight

        # Tie-breaker order is deliberately conservative. Questions that mention
        # both "documents" and another administrative detail usually still need
        # the document-table path. A mixed deadline/payment question currently
        # works best through DEADLINE_QUESTION in the existing generation layer.
        priority = [
            QuestionIntentEnum.DOCUMENTS_QUESTION,
            QuestionIntentEnum.REJECTION_QUESTION,
            QuestionIntentEnum.DEADLINE_QUESTION,
            QuestionIntentEnum.PAYMENT_TIMING_QUESTION,
            QuestionIntentEnum.AMOUNT_QUESTION,
            QuestionIntentEnum.FORM_QUESTION,
            QuestionIntentEnum.APPEAL_QUESTION,
            QuestionIntentEnum.PROCEDURE_QUESTION,
            QuestionIntentEnum.ELIGIBILITY_QUESTION,
            QuestionIntentEnum.MIXED_QUESTION,
            QuestionIntentEnum.AMBIGUOUS_QUESTION,
            QuestionIntentEnum.OTHER,
        ]

        best_score = max(scores.values())
        best_intents = {intent for intent, score in scores.items() if score == best_score}
        for intent in priority:
            if intent in best_intents:
                return intent
        return matches[0].intent_type

    @staticmethod
    def _estimate_confidence(chosen_score: int, total_score: int, chosen_matches_count: int) -> float:
        if total_score <= 0:
            return 0.15
        ratio = chosen_score / total_score
        bonus = min(chosen_matches_count, 3) * 0.04
        confidence = 0.45 + ratio * 0.45 + bonus
        return round(max(0.15, min(confidence, 0.98)), 3)

    def _build_result(
        self,
        *,
        question_text: str,
        intent_type: QuestionIntentEnum,
        confidence: float,
        matched_rules: list[IntentRuleMatch],
        query_constraints_json: dict[str, Any],
        routing_payload_json: dict[str, Any],
    ) -> dict[str, Any]:
        routing_payload_json = dict(routing_payload_json or {})
        routing_payload_json.setdefault("source", self.version)
        routing_payload_json.setdefault("confidence", confidence)
        routing_payload_json.setdefault("matched_rules", [match.code for match in matched_rules])
        routing_payload_json.setdefault("normalized_for_classification", question_text)

        return {
            "intent_type": intent_type,
            "subject_category_code": None,
            "classifier_version": self.version,
            "routing_payload_json": routing_payload_json,
            "query_constraints_json": query_constraints_json,
        }


def normalize_question_text(value: str) -> str:
    """
    Normalize text for stable rule matching.
    """
    text = str(value or "").strip().lower().replace("ё", "е")
    text = re.sub(r"[\u00a0\t\r\n]+", " ", text)
    text = re.sub(r"[^0-9a-zа-я_\-\s./]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_rules() -> list[IntentRule]:
    return [
        IntentRule(
            code="service_discovery_broad_entitlement",
            intent_type=QuestionIntentEnum.ELIGIBILITY_QUESTION,
            weight=120,
            patterns=(
                r"\bчто\s+(?:мне|нам|семье)\s+положен[оаы]?\b",
                r"\bкакие\s+(?:меры|выплаты|пособия|льготы|услуги)\s+(?:мне|нам|семье)?\s*(?:положен[ыао]?|можно\s+получить|доступн[ыао])\b",
                r"\bна\s+какую\s+(?:помощь|поддержку|выплату|меру)\s+(?:я|мы)?\s*(?:имею|имеем|можно)\b",
                r"\bподбер(?:и|ите)?\s+(?:меры|выплаты|пособия|льготы|услуги)\b",
                r"\bмер[аы]\s+поддержки\s+для\b",
                r"\bвыплат[а-я]*\s+для\b",
                r"\bльгот[а-я]*\s+для\b",
                r"\bпособи[яй]\s+для\b",
                r"\bмне\s+нужна\s+помощь\b",
            ),
            adds_constraints={
                "requires_service_discovery": True,
                "avoid_single_service_resolution": True,
                "routing_mode": "service_discovery",
            },
            adds_payload={
                "requires_service_discovery": True,
                "routing_note": "широкий вопрос о возможных мерах поддержки",
            },
        ),
        IntentRule(
            code="documents_direct",
            intent_type=QuestionIntentEnum.DOCUMENTS_QUESTION,
            weight=110,
            patterns=(
                r"\bкакие\s+документ[а-я]*\b",
                r"\bкакой\s+пакет\s+документ[а-я]*\b",
                r"\bперечень\s+документ[а-я]*\b",
                r"\bсписок\s+документ[а-я]*\b",
                r"\bчто\s+(?:нужно|надо)\s+(?:приложить|предоставить|подать|донести)\b",
                r"\bнужн[а-я]*\s+ли\s+.*документ[а-я]*\b",
                r"\bдокумент[а-я]*\s+нужн[а-я]*\b",
            ),
        ),
        IntentRule(
            code="representative_documents",
            intent_type=QuestionIntentEnum.DOCUMENTS_QUESTION,
            weight=95,
            patterns=(
                r"\bпредставител[ьяюем]*\b.*\bдоверенн?ост[ьи]\b",
                r"\bдоверенн?ост[ьи]\b.*\bпредставител[ьяюем]*\b",
                r"\bдокумент[а-я]*\s+представител[ьяюем]*\b",
            ),
        ),
        IntentRule(
            code="rejection_direct",
            intent_type=QuestionIntentEnum.REJECTION_QUESTION,
            weight=110,
            patterns=(
                r"\b(?:причин[аы]|основани[ея])\s+(?:для\s+)?отказ[а-я]*\b",
                r"\bпочему\s+(?:могут\s+)?отказать\b",
                r"\bкогда\s+(?:могут\s+)?отказать\b",
                r"\bотказ[а-я]*\s+(?:в\s+)?(?:назначени[ияи]|предоставлени[ияи]|приеме|приеме)\b",
                r"\bприостановлени[ея]\b",
            ),
        ),
        IntentRule(
            code="deadline_decision",
            intent_type=QuestionIntentEnum.DEADLINE_QUESTION,
            weight=105,
            patterns=(
                r"\bсрок\s+(?:принятия\s+)?решени[яе]\b",
                r"\bкогда\s+(?:примут|будет)\s+решени[ея]\b",
                r"\bсколько\s+(?:дней|рабочих\s+дней|времени)\s+.*решени[яе]\b",
                r"\bв\s+какой\s+срок\s+.*решени[яе]\b",
                r"\bуведом[яа]т\s+о\s+решени[ие]\b",
                r"\bсрок\s+уведомлени[яе]\b",
            ),
            adds_constraints={"deadline_focus": "decision"},
        ),
        IntentRule(
            code="deadline_generic",
            intent_type=QuestionIntentEnum.DEADLINE_QUESTION,
            weight=75,
            patterns=(
                r"\bв\s+какой\s+срок\b",
                r"\bсрок[и]?\s+(?:назначени[яе]|предоставлени[яе]|оказани[яе]|рассмотрени[яе])\b",
                r"\bсколько\s+(?:рассматривают|ждать|дней|рабочих\s+дней)\b",
            ),
            adds_constraints={"deadline_focus": "decision"},
        ),
        IntentRule(
            code="payment_timing",
            intent_type=QuestionIntentEnum.PAYMENT_TIMING_QUESTION,
            weight=90,
            patterns=(
                r"\bкогда\s+(?:придет|поступит|перечислят|выплатят|будет)\s+(?:выплат[аы]|едв|деньги|субсиди[яюи])\b",
                r"\bкогда\s+я\s+получу\s+выплат[уы]\b",
                r"\bсрок\s+(?:перечислени[яе]|выплат[ыа])\b",
                r"\bдата\s+(?:выплат[ыа]|перечислени[яе])\b",
            ),
            adds_constraints={"deadline_focus": "payment"},
        ),
        IntentRule(
            code="amount",
            intent_type=QuestionIntentEnum.AMOUNT_QUESTION,
            weight=95,
            patterns=(
                r"\b(?:размер|сумм[ауы]|сколько)\s+(?:выплат[а-я]*|пособи[яе]|субсиди[яи]|едв|компенсаци[яи])\b",
                r"\bкакая\s+сумм[ауы]\b",
                r"\bсколько\s+(?:платят|выплачивают|дадут|можно\s+получить)\b",
            ),
        ),
        IntentRule(
            code="eligibility_direct",
            intent_type=QuestionIntentEnum.ELIGIBILITY_QUESTION,
            weight=85,
            patterns=(
                r"\bкто\s+(?:имеет\s+право|может\s+получить|может\s+подать|получател[ьи])\b",
                r"\bкому\s+(?:положен[аоы]?|предоставляется|назначается)\b",
                r"\bимею\s+ли\s+я\s+право\b",
                r"\bположен[аоы]?\s+ли\b",
                r"\bмне\s+положен[аоы]?\b",
                r"\bнам\s+положен[аоы]?\b",
                r"\bположен[аоы]?\s+(?:едв|субсид[а-я]*|выплат[а-я]*|пособи[ея])\b",
                r"\bмогу\s+ли\s+я\s+получить\b",
                r"\bкатегори[яи]\s+заявител[яеи]" ,
                r"\bперечень\s+категори[йи]\b",
                r"\bуслови[яе]\s+(?:получени[яе]|назначени[яе]|предоставлени[яе])\b",
            ),
        ),
        IntentRule(
            code="procedure_how_to_get",
            intent_type=QuestionIntentEnum.PROCEDURE_QUESTION,
            weight=80,
            patterns=(
                r"\bкак\s+(?:получить|оформить|подать|обратиться|записаться)\b",
                r"\bкуда\s+(?:обращаться|обратиться|подать)\b",
                r"\bспособ[ы]?\s+(?:обращени[яе]|подачи)\b",
                r"\bможно\s+ли\s+подать\s+(?:заявление\s+)?(?:онлайн|через\s+епгу|через\s+мфц|почтой)\b",
                r"\bчерез\s+(?:епгу|госуслуг[аи]?|мфц)\b",
            ),
        ),
        IntentRule(
            code="appeal",
            intent_type=QuestionIntentEnum.APPEAL_QUESTION,
            weight=90,
            patterns=(
                r"\bобжалова[тн][а-я]*\b",
                r"\bжалоб[ауы]\b",
                r"\bоспорить\s+решени[ея]\b",
                r"\bне\s+соглас[ееннаы]*\s+с\s+решени[еям]\b",
            ),
        ),
        IntentRule(
            code="form",
            intent_type=QuestionIntentEnum.FORM_QUESTION,
            weight=90,
            patterns=(
                r"\bформ[ауы]\s+заявлени[яе]\b",
                r"\bбланк\b",
                r"\bобразец\s+(?:заявлени[яе]|заполнени[яе])\b",
                r"\bшаблон\s+заявлени[яе]\b",
            ),
        ),
        IntentRule(
            code="mixed_question",
            intent_type=QuestionIntentEnum.MIXED_QUESTION,
            weight=55,
            patterns=(
                r"\bи\b.*\bи\b.*\?*$",
                r"\bодновременно\b",
                r"\bсразу\s+несколько\b",
            ),
        ),
        IntentRule(
            code="ambiguous_generic_support",
            intent_type=QuestionIntentEnum.AMBIGUOUS_QUESTION,
            weight=45,
            patterns=(
                r"\bпомогите\b",
                r"\bнужна\s+помощь\b",
                r"\bокажите\s+помощь\b",
                r"\bдайте\s+денег\b",
                r"\bне\s+хватает\s+денег\b",
                r"\bпомощь\s+на\s+(?:дрова|уголь|школ[ауеы]|лекарств[ао])\b",
                r"\bнечего\s+есть\b",
                r"\bсгорел\s+(?:дом|жилье|квартира)\b",
            ),
        ),
    ]


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _merge_dict(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in (source or {}).items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_dict(target[key], value)
        else:
            target[key] = value
