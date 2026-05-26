# ============================================================
# File: app/services/retrieval/service_resolver.py
# Purpose:
#   Resolve a user question to a concrete public service from
#   service_registry.
#
# Notes:
#   - this layer replaces the old coarse measure_code approach;
#   - it does not hard-filter retrieval by itself;
#   - callers must use strict service_key filtering only when
#     resolution_status == "resolved".
# ============================================================

from __future__ import annotations

import math
import re
from time import perf_counter
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.services import ServiceRegistry


# ============================================================
# DTOs
# ============================================================

@dataclass(slots=True, frozen=True)
class ServiceResolverInput:
    """
    Input for service resolution.

    question_text:
        Raw or normalized user question.

    max_candidates:
        Number of candidates to keep in the result.

    min_resolved_score:
        Minimum score required for confident resolution.

    ambiguity_margin:
        If the second candidate is closer than this margin to the first one,
        the result is considered ambiguous.
    """

    question_text: str
    max_candidates: int = 7
    min_resolved_score: float = 72.0
    min_candidate_score: float = 18.0
    ambiguity_margin: float = 14.0


@dataclass(slots=True, frozen=True)
class ServiceCandidate:
    """Candidate concrete service."""

    service_key: str
    service_name_short: str
    service_name_full: str
    frgu_1: Optional[str]
    frgu_3: Optional[str]
    cleaned_filename: str

    score: float
    confidence: str
    matched_terms: list[str] = field(default_factory=list)
    matched_aliases: list[str] = field(default_factory=list)
    match_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class ServiceResolutionResult:
    """
    Service resolution result.

    resolution_status:
        - resolved: one confident service was found;
        - ambiguous: several plausible services were found;
        - not_found: no reliable service candidate.
    """

    resolution_status: str
    selected_service: Optional[ServiceCandidate]
    candidates: list[ServiceCandidate]
    debug_payload_json: dict[str, Any] = field(default_factory=dict)

    @property
    def service_key(self) -> Optional[str]:
        if self.selected_service is None:
            return None
        return self.selected_service.service_key


@dataclass(slots=True, frozen=True)
class _SearchPhrase:
    source: str
    original: str
    normalized: str
    tokens: frozenset[str]


@dataclass(slots=True, frozen=True)
class _ServiceRegistryRecord:
    service_key: str
    service_name_short: str
    service_name_full: str
    frgu_1: Optional[str]
    frgu_3: Optional[str]
    cleaned_filename: str
    aliases_json: Any


@dataclass(slots=True, frozen=True)
class _ServiceSearchDocument:
    service_key: str
    service_name_short: str
    service_name_full: str
    frgu_1: Optional[str]
    frgu_3: Optional[str]
    cleaned_filename: str
    aliases: tuple[str, ...]
    phrases: tuple[_SearchPhrase, ...]
    tokens: frozenset[str]


@dataclass(slots=True)
class _ScoreAccumulator:
    score: float = 0.0
    matched_terms: set[str] = field(default_factory=set)
    matched_aliases: set[str] = field(default_factory=set)
    match_reasons: list[str] = field(default_factory=list)


# ============================================================
# Resolver
# ============================================================

class ServiceResolver:
    """
    Resolves the concrete service from service_registry.

    The resolver is intentionally deterministic and conservative. It uses
    aliases and service names from the registry, plus a small lexical scoring
    layer. It does not call an LLM.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self._service_docs_cache: list[_ServiceSearchDocument] | None = None
        self._token_document_frequency_cache: dict[str, int] | None = None

    async def resolve(
        self,
        payload: ServiceResolverInput,
    ) -> ServiceResolutionResult:
        total_started_at = perf_counter()
        normalize_started_at = perf_counter()
        question_text = _normalize_text(payload.question_text)
        normalize_elapsed = perf_counter() - normalize_started_at

        if not question_text:
            return ServiceResolutionResult(
                resolution_status="not_found",
                selected_service=None,
                candidates=[],
                debug_payload_json={
                    "reason": "empty_question",
                    "timings_sec": {
                        "normalize_question": round(normalize_elapsed, 6),
                        "total": round(perf_counter() - total_started_at, 6),
                    },
                },
            )

        index_started_at = perf_counter()
        service_docs, token_document_frequency, index_cache_hit = await self._get_search_index()
        index_elapsed = perf_counter() - index_started_at

        question_tokens_started_at = perf_counter()
        question_tokens = _extract_tokens(question_text)
        question_padded = _padded_text(question_text)
        question_tokens_elapsed = perf_counter() - question_tokens_started_at

        scoring_started_at = perf_counter()
        candidates: list[ServiceCandidate] = []
        for service_doc in service_docs:
            candidate = _score_service(
                service_doc=service_doc,
                question_text=question_text,
                question_padded=question_padded,
                question_tokens=question_tokens,
                token_document_frequency=token_document_frequency,
                total_services=max(1, len(service_docs)),
            )
            if candidate is None:
                continue
            if candidate.score < payload.min_candidate_score:
                continue
            candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                item.score,
                len(item.matched_aliases),
                len(item.matched_terms),
                item.service_name_short,
            ),
            reverse=True,
        )
        candidates = candidates[: max(1, payload.max_candidates)]
        scoring_elapsed = perf_counter() - scoring_started_at

        choose_started_at = perf_counter()
        status, selected = _choose_resolution_status(
            candidates=candidates,
            min_resolved_score=payload.min_resolved_score,
            ambiguity_margin=payload.ambiguity_margin,
        )
        choose_elapsed = perf_counter() - choose_started_at
        total_elapsed = perf_counter() - total_started_at

        return ServiceResolutionResult(
            resolution_status=status,
            selected_service=selected,
            candidates=candidates,
            debug_payload_json={
                "normalized_question": question_text,
                "question_tokens": sorted(question_tokens),
                "active_services_count": len(service_docs),
                "index_cache_hit": index_cache_hit,
                "min_resolved_score": payload.min_resolved_score,
                "min_candidate_score": payload.min_candidate_score,
                "ambiguity_margin": payload.ambiguity_margin,
                "timings_sec": {
                    "normalize_question": round(normalize_elapsed, 6),
                    "load_or_get_index": round(index_elapsed, 6),
                    "extract_question_tokens": round(question_tokens_elapsed, 6),
                    "score_candidates": round(scoring_elapsed, 6),
                    "choose_status": round(choose_elapsed, 6),
                    "total": round(total_elapsed, 6),
                },
            },
        )

    async def _get_search_index(
        self,
    ) -> tuple[list[_ServiceSearchDocument], dict[str, int], bool]:
        if (
            self._service_docs_cache is not None
            and self._token_document_frequency_cache is not None
        ):
            return self._service_docs_cache, self._token_document_frequency_cache, True

        raw_services = await self._load_active_services()
        service_docs = [_build_search_document(service) for service in raw_services]
        token_document_frequency = _build_token_document_frequency(service_docs)

        self._service_docs_cache = service_docs
        self._token_document_frequency_cache = token_document_frequency

        return service_docs, token_document_frequency, False

    def clear_cache(self) -> None:
        self._service_docs_cache = None
        self._token_document_frequency_cache = None

    async def _load_active_services(self) -> list[_ServiceRegistryRecord]:
        """
        Load only the columns needed for service resolution.

        Do not select the full ORM entity here. ServiceRegistry has relationships
        to documents, and document entities have selectin relationships to blocks,
        tables, rows and facts. Loading the ORM entity can therefore accidentally
        pull a large part of the corpus into memory. For resolver indexing we only
        need registry text fields, so a column-level SELECT is safer and much
        faster.
        """
        result = await self.db.execute(
            select(
                ServiceRegistry.service_key,
                ServiceRegistry.service_name_short,
                ServiceRegistry.service_name_full,
                ServiceRegistry.frgu_1,
                ServiceRegistry.frgu_3,
                ServiceRegistry.cleaned_filename,
                ServiceRegistry.aliases_json,
            )
            .where(ServiceRegistry.is_active.is_(True))
            .order_by(ServiceRegistry.service_name_short.asc())
        )

        return [
            _ServiceRegistryRecord(
                service_key=row.service_key,
                service_name_short=row.service_name_short,
                service_name_full=row.service_name_full,
                frgu_1=row.frgu_1,
                frgu_3=row.frgu_3,
                cleaned_filename=row.cleaned_filename,
                aliases_json=row.aliases_json,
            )
            for row in result
        ]


# ============================================================
# Scoring
# ============================================================

def _score_service(
    *,
    service_doc: _ServiceSearchDocument,
    question_text: str,
    question_padded: str,
    question_tokens: set[str],
    token_document_frequency: dict[str, int],
    total_services: int,
) -> Optional[ServiceCandidate]:
    accumulator = _ScoreAccumulator()
    best_score_by_source: dict[str, float] = {
        "short_name": 0.0,
        "full_name": 0.0,
        "alias": 0.0,
    }

    for phrase in service_doc.phrases:
        if not phrase.normalized or not phrase.tokens:
            continue

        phrase_tokens = set(phrase.tokens)
        overlap_tokens = question_tokens.intersection(phrase_tokens)
        if not overlap_tokens:
            continue

        exact_phrase_match = _contains_phrase(question_padded, phrase.normalized)
        exact_token_match = _is_exact_token_match(
            source=phrase.source,
            phrase_tokens=phrase_tokens,
            question_tokens=question_tokens,
        )

        if exact_phrase_match or exact_token_match:
            phrase_score = _exact_phrase_score(
                source=phrase.source,
                phrase_tokens=phrase_tokens,
                token_document_frequency=token_document_frequency,
                total_services=total_services,
            )
            if exact_token_match and not exact_phrase_match:
                phrase_score *= _token_exact_match_multiplier(phrase.source, len(phrase_tokens))

            accumulator.matched_aliases.add(phrase.original)
            accumulator.matched_terms.update(overlap_tokens)
            if exact_phrase_match:
                reason = f"точное совпадение фразы: {phrase.source}: {phrase.original}"
            else:
                reason = f"точное совпадение токенов: {phrase.source}: {phrase.original}"
            accumulator.match_reasons.append(reason)
        else:
            phrase_score = _overlap_score(
                source=phrase.source,
                phrase_tokens=phrase_tokens,
                overlap_tokens=overlap_tokens,
                token_document_frequency=token_document_frequency,
                total_services=total_services,
            )
            if phrase_score <= 0:
                continue

            accumulator.matched_terms.update(overlap_tokens)
            if phrase.source == "alias" and len(overlap_tokens) >= 2:
                accumulator.matched_aliases.add(phrase.original)
            accumulator.match_reasons.append(
                f"частичное совпадение: {phrase.source}: {phrase.original}"
            )

        source_key = phrase.source if phrase.source in best_score_by_source else "alias"
        if phrase_score > best_score_by_source[source_key]:
            best_score_by_source[source_key] = phrase_score

    raw_score = (
        best_score_by_source["alias"]
        + best_score_by_source["short_name"] * 0.75
        + best_score_by_source["full_name"] * 0.35
    )

    if raw_score <= 0:
        return None

    accumulator.score = raw_score

    # Не даём одному и тому же очень общему слову вроде "выплата" или "пособие"
    # случайно сделать услугу уверенной.
    distinct_specific_terms = [
        token
        for token in accumulator.matched_terms
        if _token_idf(token, token_document_frequency, total_services) >= 1.65
    ]
    if not distinct_specific_terms and accumulator.score < 90:
        accumulator.score *= 0.55
        accumulator.match_reasons.append("понижение: нет специфичных совпавших терминов")

    accumulator.score = _apply_question_context_adjustments(
        score=accumulator.score,
        service_doc=service_doc,
        question_text=question_text,
        question_tokens=question_tokens,
        accumulator=accumulator,
    )

    score = min(round(accumulator.score, 4), 100.0)
    if score <= 0:
        return None

    confidence = _confidence_from_score(score)

    return ServiceCandidate(
        service_key=service_doc.service_key,
        service_name_short=service_doc.service_name_short,
        service_name_full=service_doc.service_name_full,
        frgu_1=service_doc.frgu_1,
        frgu_3=service_doc.frgu_3,
        cleaned_filename=service_doc.cleaned_filename,
        score=score,
        confidence=confidence,
        matched_terms=sorted(accumulator.matched_terms),
        matched_aliases=sorted(accumulator.matched_aliases),
        match_reasons=accumulator.match_reasons[:10],
    )


def _exact_phrase_score(
    *,
    source: str,
    phrase_tokens: set[str],
    token_document_frequency: dict[str, int],
    total_services: int,
) -> float:
    specific_weight = _specificity_weight(
        tokens=phrase_tokens,
        token_document_frequency=token_document_frequency,
        total_services=total_services,
    )

    token_count = len(phrase_tokens)
    if source == "short_name":
        base = 62.0
    elif source == "full_name":
        base = 54.0
    else:
        base = 50.0

    score = base + min(24.0, token_count * 4.0) + min(18.0, specific_weight * 4.0)

    if token_count == 1:
        # Однословный alias полезен только если он редкий.
        score *= min(1.0, 0.45 + specific_weight / 5.0)

    return score


def _overlap_score(
    *,
    source: str,
    phrase_tokens: set[str],
    overlap_tokens: set[str],
    token_document_frequency: dict[str, int],
    total_services: int,
) -> float:
    if not phrase_tokens or not overlap_tokens:
        return 0.0

    phrase_weight = sum(
        _token_idf(token, token_document_frequency, total_services)
        for token in phrase_tokens
    )
    overlap_weight = sum(
        _token_idf(token, token_document_frequency, total_services)
        for token in overlap_tokens
    )
    if phrase_weight <= 0:
        return 0.0

    coverage = overlap_weight / phrase_weight
    plain_coverage = len(overlap_tokens) / len(phrase_tokens)

    source_weight = {
        "short_name": 24.0,
        "alias": 20.0,
        "full_name": 14.0,
    }.get(source, 12.0)

    score = source_weight * coverage

    if len(overlap_tokens) >= 2:
        score += 8.0
    if plain_coverage >= 0.80 and len(phrase_tokens) >= 2:
        score += 12.0
    if plain_coverage == 1.0 and len(phrase_tokens) >= 2:
        score += 12.0

    # Уменьшаем вклад совпадения только по общим словам.
    if all(_is_generic_token(token) for token in overlap_tokens):
        score *= 0.35

    return score


def _is_exact_token_match(
    *,
    source: str,
    phrase_tokens: set[str],
    question_tokens: set[str],
) -> bool:
    if not phrase_tokens:
        return False
    if not phrase_tokens.issubset(question_tokens):
        return False

    if source == "alias":
        return True
    if source == "short_name" and len(phrase_tokens) <= 6:
        return True
    if source == "full_name" and len(phrase_tokens) <= 4:
        return True
    return False


def _token_exact_match_multiplier(source: str, token_count: int) -> float:
    if source == "alias":
        return 0.96 if token_count >= 2 else 1.0
    if source == "short_name":
        return 0.88
    return 0.72


def _apply_question_context_adjustments(
    *,
    score: float,
    service_doc: _ServiceSearchDocument,
    question_text: str,
    question_tokens: set[str],
    accumulator: _ScoreAccumulator,
) -> float:
    service_tokens = set(service_doc.tokens)
    adjusted = score

    def q_has_any(tokens: set[str]) -> bool:
        return bool(question_tokens.intersection(tokens))

    def s_has_any(tokens: set[str]) -> bool:
        return bool(service_tokens.intersection(tokens))

    # Пользовательские сокращения и разговорные слова должны быть сильным
    # сигналом, но только когда они есть у самой услуги.
    if "соцконтракт" in question_tokens:
        if "соцконтракт" in service_tokens:
            adjusted += 34.0
            accumulator.match_reasons.append("усиление: вопрос явно про соцконтракт")
        else:
            adjusted *= 0.35
            accumulator.match_reasons.append("понижение: вопрос про соцконтракт, услуга не про соцконтракт")

    sankur_question = "санкур" in question_tokens or {"санаторно", "курортн"}.issubset(question_tokens)
    if sankur_question:
        service_has_sankur = "санкур" in service_tokens or {"санаторно", "курортн"}.issubset(service_tokens)
        if service_has_sankur:
            adjusted += 34.0
            accumulator.match_reasons.append("усиление: вопрос явно про санкур")
        else:
            adjusted *= 0.30
            accumulator.match_reasons.append("понижение: вопрос про санкур, услуга не про санкур")

    if "едв" in question_tokens:
        if "едв" in service_tokens:
            adjusted += 14.0
            accumulator.match_reasons.append("усиление: вопрос явно про ЕДВ")
        else:
            adjusted *= 0.42
            accumulator.match_reasons.append("понижение: вопрос про ЕДВ, услуга не про ЕДВ")

    tjs_question = (
        "тжс" in question_tokens
        or {"трудн", "жизненн"}.issubset(question_tokens)
        or q_has_any({"пожар", "дрова", "бедств", "нечего"})
    )
    if tjs_question:
        service_has_tjs = (
            "тжс" in service_tokens
            or {"трудн", "жизненн"}.issubset(service_tokens)
            or s_has_any({"пожар", "дрова", "бедств", "нечего"})
        )
        if service_has_tjs:
            adjusted += 38.0
            accumulator.match_reasons.append("усиление: вопрос про ТЖС/материальную помощь")
        else:
            adjusted *= 0.45
            accumulator.match_reasons.append("понижение: вопрос про ТЖС, услуга не про ТЖС")

    # Если вопрос про ребенка/детей, детская услуга должна обгонять взрослые
    # услуги с похожими словами.
    question_has_child_context = q_has_any({"дет", "ребенок", "несовершеннолетн"})
    service_has_child_context = s_has_any({"дет", "ребенок", "несовершеннолетн"})
    if question_has_child_context:
        if service_has_child_context:
            adjusted += 12.0
            accumulator.match_reasons.append("усиление: совпал детский контекст")
        elif "санкур" in question_tokens:
            adjusted *= 0.72
            accumulator.match_reasons.append("понижение: вопрос про ребенка, услуга без детского контекста")

    # Вопросы вида "что мне положено" обычно не про одну услугу, а про
    # первичный подбор нескольких возможных мер. Resolver не должен превращать
    # такой вопрос в жесткий service_key-фильтр, но порядок кандидатов должен
    # быть предметно разумным.
    broad_entitlement_question = _is_broad_entitlement_question(question_text)

    question_has_three_children_context = (
        question_has_child_context
        and q_has_any({"3", "три", "трое", "трем", "трех"})
    )
    service_has_large_family_context = (
        s_has_any({"многодетн"})
        or (service_has_child_context and s_has_any({"3", "три", "трое", "трем", "трех"}))
    )
    if question_has_three_children_context:
        if service_has_large_family_context:
            adjusted += 28.0
            accumulator.match_reasons.append("усиление: вопрос про семью с тремя детьми")
        elif broad_entitlement_question and not service_has_child_context:
            adjusted *= 0.72
            accumulator.match_reasons.append("понижение: общий вопрос про детей, услуга без детского контекста")

    # Общий вопрос про санаторно-курортное лечение без слов "дети",
    # "ребенок", "Таймыр" или "Эвенкия" лучше оставить неоднозначным.
    # Иначе resolver слишком уверенно выбирает детский санкур только потому,
    # что он является самым частым / самым сильным кандидатом.
    if sankur_question and service_has_child_context and not question_has_child_context:
        if not q_has_any({"таймыр", "эвенки", "эвенкийск"}):
            adjusted *= 0.78
            accumulator.match_reasons.append(
                "понижение: общий вопрос про санкур без детского контекста"
            )

    # Узкие территориальные услуги не должны побеждать общекраевую услугу,
    # если пользователь сам не назвал Таймыр или Эвенкию. Это особенно важно
    # для общих вопросов "что мне положено": иначе сильный alias вроде
    # "мать-одиночка" может ошибочно вывести наверх услугу только для Эвенкии.
    regional_tokens = {"таймыр", "эвенки", "эвенкийск"}
    service_is_regional = s_has_any(regional_tokens)
    question_is_regional = q_has_any(regional_tokens)
    if service_is_regional and not question_is_regional:
        if broad_entitlement_question:
            adjusted *= 0.34
            accumulator.match_reasons.append(
                "сильное понижение: общий вопрос, узкая территория не указана"
            )
        else:
            adjusted *= 0.52
            accumulator.match_reasons.append("понижение: узкая территория не указана в вопросе")

    # Вопрос про выплату не должен уходить в услугу присвоения звания.
    if s_has_any({"зван", "удостоверен"}) and not q_has_any({"зван", "удостоверен"}):
        if q_has_any({"едв", "выплат", "документ", "документн"}):
            adjusted *= 0.45
            accumulator.match_reasons.append("понижение: вопрос не про присвоение звания/удостоверение")

    # Узкие услуги по зубопротезированию не должны всплывать по общему
    # вопросу про ветерана труда.
    if s_has_any({"зубопротезирован", "протез", "зубн"}) and not q_has_any({"зуб", "протез", "зубопротезирован"}):
        adjusted *= 0.45
        accumulator.match_reasons.append("понижение: вопрос не про зубопротезирование")

    # Разговорные формулировки, где общие слова вроде "ребёнок" или
    # "выплата" раньше ошибочно тянули к пособию при рождении ребёнка.
    # Эти правила не подменяют service_registry, а только помогают resolver'у
    # не путать близкие по словам, но разные по смыслу услуги.
    service_text = _normalize_text(
        " ".join(
            [
                service_doc.service_name_short,
                service_doc.service_name_full,
                " ".join(service_doc.aliases),
                service_doc.cleaned_filename,
            ]
        )
    )

    war_child_question = (
        q_has_any({"войн", "переживш", "детств"})
        and ("ребенок" in question_tokens or "дет" in question_tokens)
    )
    if war_child_question:
        if any(marker in service_text for marker in ("переживш", "войн", "детств")):
            adjusted += 46.0
            accumulator.match_reasons.append("усиление: вопрос про граждан, переживших войну в детстве")
        elif "рожд" in service_tokens or "рожд" in service_text:
            adjusted *= 0.16
            accumulator.match_reasons.append("сильное понижение: вопрос про детей войны, услуга про рождение ребёнка")

    svo_child_question = (
        ("сво" in question_tokens or "военнослужащ" in question_tokens)
        and q_has_any({"погибш", "умерш", "отец", "родител", "семь"})
        and question_has_child_context
    )
    if svo_child_question:
        service_has_svo_child = (
            "сво" in service_tokens
            or "военнослужащ" in service_tokens
            or ("погибш" in service_tokens and service_has_child_context)
        )
        if service_has_svo_child:
            adjusted += 48.0
            accumulator.match_reasons.append("усиление: вопрос про ребёнка погибшего участника СВО/военнослужащего")
        elif "рожд" in service_tokens or "рожд" in service_text:
            adjusted *= 0.14
            accumulator.match_reasons.append("сильное понижение: вопрос про СВО, услуга про рождение ребёнка")

    land_certificate_question = "сертификат" in question_tokens and "земельн" in question_tokens
    land_use_question = land_certificate_question and any(
        marker in question_text
        for marker in ("использ", "распоряд", "потрат", "куп", "строит", "приобрест")
    )
    if land_use_question:
        if "распоряж" in service_text:
            adjusted += 44.0
            accumulator.match_reasons.append("усиление: вопрос про распоряжение земельным сертификатом")
        elif "получен" in service_text or "выдач" in service_text:
            adjusted *= 0.40
            accumulator.match_reasons.append("понижение: вопрос про использование сертификата, услуга про выдачу")

    fallen_defender_question = (
        "погибш" in question_tokens
        and ("защитник" in question_tokens or "отечеств" in question_tokens)
        and question_has_child_context
    )
    fallen_defender_travel_question = fallen_defender_question and q_has_any(
        {"проезд", "захоронен", "гибел", "могил"}
    )
    if fallen_defender_travel_question:
        if "проезд" in service_tokens or "проезд" in service_text:
            adjusted += 42.0
            accumulator.match_reasons.append("усиление: вопрос про проезд ребёнка погибшего защитника Отечества")
        elif "статус" in service_tokens or "удостоверен" in service_tokens:
            adjusted *= 0.50
            accumulator.match_reasons.append("понижение: вопрос про проезд, услуга про статус/удостоверение")
    elif fallen_defender_question and q_has_any({"статус", "выплат", "удостоверен"}):
        if "проезд" in service_tokens or "проезд" in service_text:
            adjusted *= 0.38
            accumulator.match_reasons.append("сильное понижение: вопрос про статус/выплаты, услуга про проезд")
        elif "статус" in service_tokens or "выплат" in service_tokens or "удостоверен" in service_tokens:
            adjusted += 34.0
            accumulator.match_reasons.append("усиление: вопрос про статус/выплаты детям погибших защитников Отечества")

    # Простые бытовые формулировки, которые часто не совпадают с
    # канцелярским названием услуги. Эти правила дают resolver'у
    # предметный сигнал, но не подменяют registry: усиливаем только те
    # услуги, где тот же смысл есть в названии/алиасах/файле.
    dental_question = q_has_any({"зубопротезирование", "стоматологическ", "протез", "зубн"})
    service_has_dental = s_has_any({"зубопротезирование", "стоматологическ", "протез", "зубн"})
    if dental_question:
        if service_has_dental:
            adjusted += 42.0
            accumulator.match_reasons.append("усиление: вопрос про зубопротезирование")
        else:
            adjusted *= 0.42
            accumulator.match_reasons.append("понижение: вопрос про зубопротезирование, услуга не про зубы/протезы")

    utility_question = q_has_any({"коммунальн", "жилищн", "квартплат", "жкх", "жку"})
    service_has_utility = s_has_any({"коммунальн", "жилищн", "квартплат", "жкх", "жку"})
    if utility_question:
        if service_has_utility:
            adjusted += 36.0
            accumulator.match_reasons.append("усиление: вопрос про жильё/коммунальные услуги")
        elif q_has_any({"компенсац", "оплат"}):
            adjusted *= 0.50
            accumulator.match_reasons.append("понижение: вопрос про ЖКУ, услуга не про жильё/коммунальные услуги")

    burial_question = q_has_any({"погребен", "похорон", "могил", "памятник"})
    service_has_burial = s_has_any({"погребен", "похорон", "могил", "памятник"})
    if burial_question:
        if service_has_burial:
            adjusted += 40.0
            accumulator.match_reasons.append("усиление: вопрос про погребение/памятник/могилу")
        else:
            adjusted *= 0.48
            accumulator.match_reasons.append("понижение: вопрос про погребение/могилу, услуга не про это")

    matcap_question = q_has_any({"материнск", "семейн", "капитал"})
    service_has_matcap = s_has_any({"материнск", "семейн", "капитал"})
    if matcap_question:
        if service_has_matcap:
            adjusted += 38.0
            accumulator.match_reasons.append("усиление: вопрос про материнский/семейный капитал")
        elif "земельн" in service_tokens and "сертификат" in service_tokens:
            adjusted *= 0.42
            accumulator.match_reasons.append("понижение: вопрос про маткапитал, услуга про земельный сертификат")

    child_garden_question = question_has_child_context and q_has_any({"сад", "дошкольн"})
    service_has_child_garden = service_has_child_context and s_has_any({"сад", "дошкольн"})
    if child_garden_question:
        if service_has_child_garden:
            adjusted += 34.0
            accumulator.match_reasons.append("усиление: вопрос про место в детском саду")
        elif "школьн" in service_tokens:
            adjusted *= 0.50
            accumulator.match_reasons.append("понижение: вопрос про детский сад, услуга про школу")

    camp_question = question_has_child_context and q_has_any({"лагерь", "оздоровительн"})
    service_has_camp = service_has_child_context and s_has_any({"лагерь", "оздоровительн"})
    if camp_question:
        if service_has_camp:
            adjusted += 38.0
            accumulator.match_reasons.append("усиление: вопрос про детский оздоровительный лагерь")
        elif "санаторн" in service_tokens and "курортн" in service_tokens:
            adjusted *= 0.72
            accumulator.match_reasons.append("понижение: вопрос про лагерь, услуга про санаторно-курортное лечение")

    emergency_question = q_has_any({"чрезвычайн", "ситуац", "пожар", "утрат", "имущество", "травм"})
    service_has_emergency = s_has_any({"чрезвычайн", "ситуац", "пожар", "утрат", "имущество", "вред", "здоров"})
    if emergency_question and q_has_any({"чрезвычайн", "пожар", "утрат", "имущество", "травм"}):
        if service_has_emergency:
            adjusted += 36.0
            accumulator.match_reasons.append("усиление: вопрос про ЧС/утрату имущества/вред здоровью")
        elif not tjs_question:
            adjusted *= 0.55
            accumulator.match_reasons.append("понижение: вопрос про ЧС, услуга не про ЧС")

    social_service_question = q_has_any({"социальн", "обслуживан", "соцработник", "уход", "быт"})
    service_has_social_service = s_has_any({"социальн", "обслуживан", "соцработник", "уход", "быт"})
    if social_service_question and q_has_any({"соцработник", "обслуживан", "уход"}):
        if service_has_social_service:
            adjusted += 34.0
            accumulator.match_reasons.append("усиление: вопрос про социальное обслуживание/уход")
        elif "выплат" in service_tokens or "компенсац" in service_tokens:
            adjusted *= 0.70
            accumulator.match_reasons.append("понижение: вопрос про уход/обслуживание, услуга про выплату/компенсацию")

    computer_question = q_has_any({"компьютер", "дистанционн", "обучен"})
    service_has_computer = s_has_any({"компьютер"})
    if computer_question and "компьютер" in question_tokens:
        if service_has_computer:
            adjusted += 42.0
            accumulator.match_reasons.append("усиление: вопрос про компьютер для инвалида")
        elif "обучен" in service_tokens and "вождени" in service_tokens:
            adjusted *= 0.55
            accumulator.match_reasons.append("понижение: вопрос про компьютер, услуга про обучение вождению")


    # Чисто общий вопрос «ЕДВ» без уточняющих признаков лучше оставить
    # неоднозначным: таких услуг в корпусе несколько.
    if question_tokens == {"едв"} and "едв" in service_tokens:
        adjusted = max(adjusted, 74.0)
        accumulator.match_reasons.append("нейтрально: общий вопрос только про ЕДВ")

    return max(0.0, adjusted)


def _is_broad_entitlement_question(question_text: str) -> bool:
    """Return True for broad discovery questions, not single-service questions."""
    normalized = _normalize_text(question_text)
    if not normalized:
        return False

    broad_phrases = (
        "что мне положено",
        "что положено",
        "что мне полагается",
        "что полагается",
        "какие меры поддержки",
        "какие меры соцподдержки",
        "какая помощь положена",
        "какие выплаты положены",
        "на что могу рассчитывать",
    )
    return any(phrase in normalized for phrase in broad_phrases)


def _choose_resolution_status(
    *,
    candidates: list[ServiceCandidate],
    min_resolved_score: float,
    ambiguity_margin: float,
) -> tuple[str, Optional[ServiceCandidate]]:
    if not candidates:
        return "not_found", None

    first = candidates[0]
    if first.score < min_resolved_score:
        if len(candidates) >= 2 and candidates[1].score >= 40.0:
            return "ambiguous", None
        return "not_found", None

    if len(candidates) == 1:
        return "resolved", first

    second = candidates[1]
    score_gap = first.score - second.score

    # Раньше resolver слишком часто оставлял статус ambiguous даже когда
    # первый кандидат имел 96-100 баллов и точное совпадение с длинным alias /
    # коротким названием. В таком режиме retrieval не получал service_key и
    # generation смешивал таблицы нескольких услуг. Оставляем осторожность для
    # настоящих близких случаев, но закрепляем услугу при сильном предметном
    # сигнале.
    first_specificity = _candidate_specificity_score(first)
    second_specificity = _candidate_specificity_score(second)
    strong_exact_signal = _has_strong_exact_resolution_signal(first)

    if strong_exact_signal:
        if first.score >= 96.0 and score_gap >= 3.0:
            return "resolved", first
        if first.score >= 90.0 and score_gap >= 5.0:
            return "resolved", first
        if first.score >= 86.0 and score_gap >= 8.0:
            return "resolved", first
        if first.score >= 80.0 and score_gap >= 12.0:
            return "resolved", first
        if first.score >= 88.0 and first_specificity > second_specificity + 2.5:
            return "resolved", first

    if second.score >= 45.0 and score_gap < ambiguity_margin:
        return "ambiguous", None

    return "resolved", first


def _has_strong_exact_resolution_signal(candidate: ServiceCandidate) -> bool:
    if not any("точное совпадение" in reason for reason in candidate.match_reasons):
        return False

    for alias in candidate.matched_aliases:
        tokens = _extract_tokens(alias)
        if len(tokens) >= 3:
            return True

    service_tokens = _extract_tokens(candidate.service_name_short)
    matched_specific = set(candidate.matched_terms).intersection(service_tokens)
    return len(matched_specific) >= 3


def _candidate_specificity_score(candidate: ServiceCandidate) -> float:
    score = 0.0
    score += min(8.0, len(candidate.matched_terms) * 1.1)
    score += min(10.0, len(candidate.matched_aliases) * 2.5)
    for alias in candidate.matched_aliases:
        score += min(8.0, len(_extract_tokens(alias)) * 0.8)
    if any("точное совпадение фразы" in reason for reason in candidate.match_reasons):
        score += 4.0
    if any("точное совпадение токенов" in reason for reason in candidate.match_reasons):
        score += 2.0
    return score


def _confidence_from_score(score: float) -> str:
    if score >= 86.0:
        return "high"
    if score >= 72.0:
        return "medium"
    if score >= 45.0:
        return "low"
    return "very_low"


# ============================================================
# Search document preparation
# ============================================================

def _build_search_document(service: _ServiceRegistryRecord) -> _ServiceSearchDocument:
    aliases = _normalize_aliases(service.aliases_json)

    phrase_items: list[tuple[str, str, str]] = []
    phrase_items.append(("short_name", service.service_name_short, _normalize_text(service.service_name_short)))
    phrase_items.append(("full_name", service.service_name_full, _normalize_text(service.service_name_full)))

    for alias in aliases:
        phrase_items.append(("alias", alias, _normalize_text(alias)))

    # Имена файлов часто содержат полезное короткое название услуги.
    filename_hint = _filename_to_service_hint(service.cleaned_filename)
    if filename_hint:
        phrase_items.append(("alias", filename_hint, _normalize_text(filename_hint)))

    deduplicated_phrases: list[_SearchPhrase] = []
    seen_phrases: set[str] = set()
    for source, original, normalized in phrase_items:
        if not normalized or normalized in seen_phrases:
            continue
        phrase_tokens = frozenset(_extract_tokens(normalized))
        if not phrase_tokens:
            continue
        seen_phrases.add(normalized)
        deduplicated_phrases.append(
            _SearchPhrase(
                source=source,
                original=original,
                normalized=normalized,
                tokens=phrase_tokens,
            )
        )

    tokens: set[str] = set()
    for phrase in deduplicated_phrases:
        tokens.update(phrase.tokens)

    return _ServiceSearchDocument(
        service_key=service.service_key,
        service_name_short=service.service_name_short,
        service_name_full=service.service_name_full,
        frgu_1=service.frgu_1,
        frgu_3=service.frgu_3,
        cleaned_filename=service.cleaned_filename,
        aliases=tuple(aliases),
        phrases=tuple(deduplicated_phrases),
        tokens=frozenset(tokens),
    )


def _normalize_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, tuple):
        raw_items = list(value)
    else:
        raw_items = [str(value)]

    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = " ".join(str(item).split()).strip()
        if not text:
            continue
        key = _normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _filename_to_service_hint(cleaned_filename: str) -> str:
    text = re.sub(r"\.docx$", "", cleaned_filename, flags=re.IGNORECASE)
    text = re.sub(r"_cleaned$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_от\s+\d{2}\.\d{2}\.\d{4}\s+N\s+.+$", "", text, flags=re.IGNORECASE)
    text = text.replace("_", " ")
    return " ".join(text.split()).strip()


def _build_token_document_frequency(
    service_docs: list[_ServiceSearchDocument],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for service_doc in service_docs:
        for token in service_doc.tokens:
            result[token] = result.get(token, 0) + 1
    return result


# ============================================================
# Text normalization
# ============================================================

_STOPWORDS = {
    "а", "без", "бы", "в", "во", "вот", "для", "до", "его", "ее", "если",
    "же", "за", "из", "или", "им", "их", "к", "как", "какая", "какие",
    "какой", "кем", "когда", "куда", "ли", "мне", "может", "можно", "мой",
    "моя", "мы", "на", "над", "надо", "не", "него", "нее", "нет", "ни", "но",
    "нужно", "о", "об", "от", "по", "под", "при", "про", "с", "со", "так",
    "то", "у", "чем", "что", "чтобы", "это", "я", "есть", "дать", "дайте",
    "нужен", "нужна", "нужны", "получить", "получения", "получение", "получаю",
    "оформить", "оформления", "предоставление", "предоставления", "услуга", "услуги",
    "меры", "мера", "поддержки", "помощь", "помогите", "заявление", "документы",
    "документ", "список", "перечень", "причины", "отказ", "отказа", "срок",
    "сроки", "решение", "решения", "выплата", "выплаты", "выплату",
    "почему", "могут", "может", "считается", "считать", "считают",
    "предоставлен", "предоставлена", "предоставлено", "предоставлены",
    "каки", "какую", "какои", "кто", "почем", "причин", "нужн",
    "получен", "получит", "подать", "заявлен", "отказать",
    "положено", "меня", "помогит", "считаетс",
}

_GENERIC_TOKENS = {
    "социальная", "социальной", "социальные", "социальную", "поддержка", "поддержки",
    "помощь", "помощи", "выплата", "выплаты", "выплату", "денежная", "денежной",
    "ежемесячная", "ежемесячной", "ежегодная", "единовременная", "компенсация",
    "пособие", "граждан", "граждане", "категорий", "отдельным", "отдельных",
    "предоставление", "получение", "документы", "заявление", "срок", "решение",
}

_TOKEN_RE = re.compile(r"[а-яa-z0-9]+(?:[.,][0-9]+)?", re.IGNORECASE)


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""

    text = str(value).lower().replace("ё", "е")
    text = _expand_common_user_phrases(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\"'«»„“”`]+", " ", text)
    text = re.sub(r"[^0-9a-zа-я.,]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _expand_common_user_phrases(value: str) -> str:
    """
    Expand common citizen wording before tokenization.

    This is intentionally small and deterministic. It does not try to solve
    morphology; it only bridges high-value everyday phrases with the wording
    used in service_registry aliases and service names.
    """
    text = f" {value} "

    replacements = (
        (r"\bжкх\b", " жилищно коммунальные услуги "),
        (r"\bжку\b", " жилищно коммунальные услуги "),
        (r"\bкоммуналк[аеиуой]*\b", " жилищно коммунальные услуги "),
        (r"\bквартплат[аеиуой]*\b", " оплата жилого помещения коммунальные услуги "),
        (r"\bматкапитал[а-я]*\b", " материнский семейный капитал "),
        (r"\bсадик[а-я]*\b", " детский сад "),
        (r"\bдетсад[а-я]*\b", " детский сад "),
        (r"\bавтошкол[аеиуой]*\b", " обучение вождению "),
        (r"\bпохорон[а-я]*\b", " погребение "),
        (r"\bпохороны\b", " погребение "),
        (r"\bпамятник[а-я]*\b", " памятник благоустройство могил "),
        (r"\bзубн[а-я]*\s+протез[а-я]*\b", " зубопротезирование стоматологические протезы "),
        (r"\bзубопротез[а-я]*\b", " зубопротезирование стоматологические протезы "),
        (r"\bделать\s+зуб[а-я]*\b", " зубопротезирование стоматологические протезы "),
        (r"\bлечить\s+зуб[а-я]*\b", " зубопротезирование стоматологические протезы "),
        (r"\bпечк[аеиуой]*\b", " печное отопление "),
        (r"\bпроводк[аеиуой]*\b", " электропроводка "),
        (r"\bтопить\b", " печное отопление "),
        (r"\bтоплени[ея]\b", " печное отопление "),
        (r"\bсоцработник[а-я]*\b", " социальное обслуживание "),
        (r"\bсоцобслуживани[а-я]*\b", " социальное обслуживание "),
        (r"\bчс\b", " чрезвычайная ситуация "),
        (r"\bтср\b", " технические средства реабилитации "),
        (r"\bчаэс\b", " чернобыльская аэс "),
        (r"\bчернобыл[а-я]*\b", " чернобыльская аэс "),
        (r"\bлагер[ьяеюям]*\b", " оздоровительный лагерь "),
        (r"\bгемодиализ[а-я]*\b", " гемодиализ "),
    )

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def _extract_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    normalized = _normalize_text(value)
    for match in _TOKEN_RE.finditer(normalized):
        token = match.group(0).strip(".,")
        if not token:
            continue
        token = _normalize_token(token)
        if len(token) < 3 and not token.isdigit():
            continue
        if token in _STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _normalize_token(token: str) -> str:
    token = token.lower().replace("ё", "е").strip(".,")

    special = {
        "края": "край",
        "краю": "край",
        "краем": "край",
        "детей": "дет",
        "детьми": "дет",
        "ребенка": "ребенок",
        "ребенку": "ребенок",
        "ребенком": "ребенок",
        "деть": "дет",
        "несовершеннолетними": "несовершеннолетн",
        "несовершеннолетних": "несовершеннолетн",
        "несовершеннолетние": "несовершеннолетн",
        "тремя": "трем",
        "трое": "трое",
        "трех": "трех",
        "трём": "трем",
        "многодетная": "многодетн",
        "многодетной": "многодетн",
        "многодетных": "многодетн",
        "многодетные": "многодетн",
        "матери": "мать",
        "матерью": "мать",
        "матерям": "мать",
        "матерей": "мать",
        "матерям": "мать",
        "матерями": "мать",
        "одиночка": "одинок",
        "одиночки": "одинок",
        "одинокой": "одинок",
        "одинокая": "одинок",
        "соцконтракта": "соцконтракт",
        "соцконтракту": "соцконтракт",
        "санкура": "санкур",
        "санкуре": "санкур",
        "таймыра": "таймыр",
        "таймыре": "таймыр",
        "эвенкии": "эвенки",
        "эвенкия": "эвенки",
        "эвенкийский": "эвенкийск",
        "эвенкийского": "эвенкийск",
        "ситуация": "ситуац",
        "ситуации": "ситуац",
        "ситуацией": "ситуац",
        "сгорел": "пожар",
        "сгорела": "пожар",
        "сгорело": "пожар",
        "пожара": "пожар",
        "пожаре": "пожар",
        "дрова": "дрова",
        "дров": "дрова",
    }
    if token in special:
        return special[token]

    # Минимальная нормализация окончаний. Это не морфологический анализатор,
    # а защита от очевидных расхождений вроде "субсидии" / "субсидий" / "субсидия".
    if token.isdigit() or len(token) <= 4:
        return token

    suffixes = (
        "иями", "ями", "ами", "ями", "ого", "его", "ому", "ему",
        "ыми", "ими", "ых", "их", "ою", "ею", "ую", "юю",
        "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий", "ой",
        "ов", "ев", "ей", "ам", "ям", "ах", "ях", "ом", "ем",
        "ия", "ии", "ию", "ью", "ье", "а", "я", "ы", "и", "у", "ю", "е",
    )
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]

    return token


def _padded_text(value: str) -> str:
    return f" {value} "


def _contains_phrase(question_padded: str, normalized_phrase: str) -> bool:
    if not normalized_phrase:
        return False
    return _padded_text(normalized_phrase) in question_padded


def _token_idf(
    token: str,
    token_document_frequency: dict[str, int],
    total_services: int,
) -> float:
    df = token_document_frequency.get(token, 0)
    return 1.0 + math.log((total_services + 1.0) / (df + 1.0))


def _specificity_weight(
    *,
    tokens: set[str],
    token_document_frequency: dict[str, int],
    total_services: int,
) -> float:
    if not tokens:
        return 0.0
    return sum(
        _token_idf(token, token_document_frequency, total_services)
        for token in tokens
        if not _is_generic_token(token)
    ) / max(1, len(tokens))


def _is_generic_token(token: str) -> bool:
    return token in _GENERIC_TOKENS or token in _STOPWORDS
