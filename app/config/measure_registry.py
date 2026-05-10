from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def normalize_measure_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


@dataclass(frozen=True)
class MeasureDefinition:
    code: str
    canonical_name: str
    aliases: tuple[str, ...]


_MEASURE_REGISTRY: tuple[MeasureDefinition, ...] = (
    MeasureDefinition(
        code="edv",
        canonical_name="Ежемесячная денежная выплата",
        aliases=(
            "ЕДВ",
            "ежемесячная денежная выплата",
            "ежемесячной денежной выплаты",
            "предоставлению ежемесячной денежной выплаты",
        ),
    ),
    MeasureDefinition(
        code="subsidy",
        canonical_name="Субсидия на оплату жилого помещения и коммунальных услуг",
        aliases=(
            "субсидия",
            "субсидии",
            "субсидий",
            "субсидию",
            "субсидия на оплату жилого помещения и коммунальных услуг",
            "субсидии на оплату жилого помещения и коммунальных услуг",
            "оплату жилого помещения",
            "коммунальных услуг",
        ),
    ),
    MeasureDefinition(
        code="social_contract",
        canonical_name="Государственная социальная помощь на основании социального контракта",
        aliases=(
            "соцконтракт",
            "соцконтракта",
            "социальный контракт",
            "социального контракта",
            "социальном контракте",
            "социальному контракту",
            "социальным контрактом",
            "государственная социальная помощь на основании социального контракта",
        ),
    ),
    MeasureDefinition(
        code="hardship",
        canonical_name="Единовременная адресная материальная помощь в трудной жизненной ситуации",
        aliases=(
            "тжс",
            "трудная жизненная ситуация",
            "трудной жизненной ситуации",
            "адресная материальная помощь",
            "единовременная адресная материальная помощь",
        ),
    ),
    MeasureDefinition(
        code="sanatorium",
        canonical_name="Бесплатные путевки на санаторно-курортное лечение",
        aliases=(
            "санкур",
            "санаторно-курортное лечение",
            "санаторно-курортного лечения",
            "санаторно курортного лечения",
            "санаторно-курортном лечении",
            "санаторно курортном лечении",
            "бесплатные путевки",
            "бесплатных путевок",
            "путевки на санаторно-курортное лечение",
            "путевок на санаторно-курортное лечение",
        ),
    ),
)

_MEASURE_BY_CODE: dict[str, MeasureDefinition] = {
    item.code: item for item in _MEASURE_REGISTRY
}


def _contains_alias(haystack: str, alias: str) -> bool:
    normalized_haystack = normalize_measure_text(haystack)
    normalized_alias = normalize_measure_text(alias)
    if not normalized_haystack or not normalized_alias:
        return False

    padded_haystack = f" {normalized_haystack} "
    padded_alias = f" {normalized_alias} "

    return padded_alias in padded_haystack or normalized_alias in normalized_haystack


def _deduplicate_preserve_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        clean = (value or "").strip()
        normalized = normalize_measure_text(clean)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(clean)

    return result


def get_measure_definition(measure_code: str | None) -> Optional[MeasureDefinition]:
    if not measure_code:
        return None
    return _MEASURE_BY_CODE.get(normalize_measure_text(measure_code))
    
def resolve_measure_code(*values: str | None) -> str | None:
    """
    Resolve measure code from:
    - explicit code ("edv", "subsidy", ...)
    - canonical name
    - alias occurrence inside free text

    Resolution order follows the order of input values.
    """
    for value in values:
        normalized = normalize_measure_text(value or "")
        if not normalized:
            continue

        direct = _MEASURE_BY_CODE.get(normalized)
        if direct is not None:
            return direct.code

        detected = detect_primary_measure_code(normalized)
        if detected:
            return detected

    return None


def detect_measure_codes(text: str) -> list[str]:
    normalized_text = normalize_measure_text(text)
    if not normalized_text:
        return []

    matched_codes: list[str] = []
    for item in _MEASURE_REGISTRY:
        all_aliases = [item.canonical_name, *item.aliases]
        if any(_contains_alias(normalized_text, alias) for alias in all_aliases):
            matched_codes.append(item.code)

    return matched_codes


def detect_primary_measure_code(text: str) -> str | None:
    codes = detect_measure_codes(text)
    return codes[0] if codes else None


def get_measure_search_terms(measure_code: str | None) -> list[str]:
    definition = get_measure_definition(measure_code)
    if definition is None:
        return []

    raw_terms = _deduplicate_preserve_order(
        [
            definition.canonical_name,
            *definition.aliases,
        ]
    )
    return [normalize_measure_text(value) for value in raw_terms]


def build_measure_alias_records(
    measure_codes: list[str],
    *,
    source: str,
    temporary: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for measure_code in measure_codes:
        definition = get_measure_definition(measure_code)
        if definition is None:
            continue

        alias_values = _deduplicate_preserve_order(
            [
                definition.canonical_name,
                *definition.aliases,
            ]
        )

        for alias in alias_values:
            records.append(
                {
                    "alias": alias,
                    "measure_code": definition.code,
                    "canonical_name": definition.canonical_name,
                    "metadata_json": {
                        "source": source,
                        "temporary": temporary,
                    },
                }
            )

    return records