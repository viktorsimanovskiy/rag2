# ============================================================
# File: scripts/analyze_missing_aliases.py
# Purpose:
#   Build alias/vocabulary suggestions from mass question-bank logs.
#
# Typical usage:
#   python scripts/analyze_missing_aliases.py \
#     --responses-file /home/logs/question_bank/step49_plain_sample/responses.jsonl \
#     --out-dir /home/logs/question_bank/alias_suggestions_step49
#
# Notes:
#   - this script does not change DB or source files;
#   - concrete service aliases should be added to Актуальный_приказ5.xlsx;
#   - generic citizen words should be added to app/config/runtime_vocabulary.json.
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RESPONSES_FILE = "responses.jsonl"
DEFAULT_OUT_DIR = "/home/logs/question_bank/alias_suggestions"
MAX_SUGGESTIONS_PER_CASE = 8

_GENERIC_WORDS = {
    "а", "без", "бы", "в", "во", "для", "до", "же", "за", "из", "или", "к", "как",
    "какая", "какие", "какой", "когда", "куда", "ли", "мне", "может", "можно", "на",
    "надо", "не", "нет", "но", "нужно", "о", "об", "от", "по", "при", "про", "с", "со",
    "так", "то", "у", "чем", "что", "чтобы", "это", "я", "есть", "дать", "дайте",
    "помогите", "помощь", "помощи", "оформить", "получить", "положено", "положена",
    "положены", "заявление", "документы", "документ", "соцзащита", "социальная",
    "социальной", "выплата", "выплаты", "компенсация", "пособие", "услуга", "услуги",
}

_RUNTIME_VOCABULARY_HINTS = {
    "коммуналка", "жкх", "жку", "квартплата", "садик", "детсад", "маткапитал",
    "автошкола", "похороны", "памятник", "зубы", "зубные", "протезы", "печка",
    "дрова", "чс", "тжс", "тср", "чаэс", "чернобыль", "донор", "доноры",
    "соцработник", "гемодиализ", "лагерь", "затопило", "сгорел", "холодильник",
}

_RESOLVER_PROBLEM_FLAGS = {
    "resolver_not_found",
    "resolver_no_candidates",
    "resolver_ambiguous_expected_top1",
    "resolver_ambiguous_expected_in_candidates",
    "resolver_ambiguous_expected_not_in_candidates",
    "resolver_resolved_other_service",
    "answer_target_not_mentioned",
    "answer_safe_no_answer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подготовить предложения по алиасам на основе массового прогона."
    )
    parser.add_argument("--responses-file", default=DEFAULT_RESPONSES_FILE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--only-problems",
        action="store_true",
        help="Обрабатывать только строки с флагами resolver/safe/target-not-mentioned.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    responses_path = Path(args.responses_file)
    if not responses_path.exists():
        raise SystemExit(f"responses.jsonl не найден: {responses_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(responses_path)
    suggestions = build_suggestions(rows, only_problems=args.only_problems)

    write_case_suggestions(out_dir / "missing_alias_suggestions.tsv", suggestions)
    write_service_summary(out_dir / "missing_alias_by_service.tsv", suggestions)
    write_runtime_summary(out_dir / "runtime_vocabulary_candidates.tsv", suggestions)
    write_readme(out_dir / "README.md", responses_path, suggestions)

    print(f"Готово: {out_dir}")
    print(f"Предложения по строкам: {out_dir / 'missing_alias_suggestions.tsv'}")
    print(f"Сводка по услугам: {out_dir / 'missing_alias_by_service.tsv'}")
    print(f"Кандидаты в runtime_vocabulary: {out_dir / 'runtime_vocabulary_candidates.tsv'}")
    return 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Ошибка JSON в {path}:{line_number}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def build_suggestions(rows: list[dict[str, Any]], *, only_problems: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        flags = set(row.get("issue_flags") or [])
        if only_problems and not flags.intersection(_RESOLVER_PROBLEM_FLAGS):
            continue

        candidate_terms = suggest_alias_terms(row)
        runtime_terms = [term for term in candidate_terms if normalize_text(term) in _RUNTIME_VOCABULARY_HINTS]
        if not candidate_terms and not runtime_terms:
            continue

        diagnostics = row.get("diagnostics") or {}
        result.append(
            {
                "question_id": row.get("question_id", ""),
                "service_index": row.get("service_index", ""),
                "expected_service_name_short": row.get("service_name_short", ""),
                "category": row.get("category", ""),
                "question_text": row.get("question_text", ""),
                "issue_flags": "; ".join(row.get("issue_flags") or []),
                "resolver_status": diagnostics.get("resolver_status", ""),
                "top_candidate_name": diagnostics.get("top_candidate_name", ""),
                "top_candidate_score": diagnostics.get("top_candidate_score", ""),
                "expected_candidate_rank": diagnostics.get("expected_candidate_rank", ""),
                "suggested_service_aliases": "; ".join(candidate_terms),
                "suggested_runtime_vocabulary_terms": "; ".join(runtime_terms),
                "where_to_edit": choose_edit_target(runtime_terms),
            }
        )
    return result


def suggest_alias_terms(row: dict[str, Any]) -> list[str]:
    question_text = str(row.get("question_text") or "")
    aliases_hint = str(row.get("aliases_hint") or "")
    service_short = str(row.get("service_name_short") or "")
    service_full = str(row.get("service_name_full") or "")
    existing_blob = normalize_text(" ".join([aliases_hint, service_short, service_full]))

    tokens = [token for token in tokenize(question_text) if token not in _GENERIC_WORDS and len(token) >= 4]
    phrases: list[str] = []

    for token in tokens:
        if token not in existing_blob:
            phrases.append(token)

    words = [word for word in re.findall(r"[А-Яа-яA-Za-z0-9]+", question_text.lower().replace("ё", "е"))]
    for size in (3, 2):
        for index in range(0, max(0, len(words) - size + 1)):
            phrase_words = words[index:index + size]
            if any(word in _GENERIC_WORDS for word in phrase_words):
                continue
            phrase = " ".join(phrase_words)
            phrase_norm = normalize_text(phrase)
            if len(phrase_norm) < 8:
                continue
            if phrase_norm not in existing_blob:
                phrases.append(phrase)

    return unique(phrases)[:MAX_SUGGESTIONS_PER_CASE]


def choose_edit_target(runtime_terms: list[str]) -> str:
    if runtime_terms:
        return "проверить: общие слова — app/config/runtime_vocabulary.json; алиасы конкретной услуги — Актуальный_приказ5.xlsx"
    return "Актуальный_приказ5.xlsx / колонка 'Ключевые слова / алиасы'"


def write_case_suggestions(path: Path, suggestions: list[dict[str, Any]]) -> None:
    headers = [
        "question_id", "service_index", "expected_service_name_short", "category", "question_text",
        "issue_flags", "resolver_status", "top_candidate_name", "top_candidate_score",
        "expected_candidate_rank", "suggested_service_aliases", "suggested_runtime_vocabulary_terms",
        "where_to_edit",
    ]
    write_dicts(path, headers, suggestions)


def write_service_summary(path: Path, suggestions: list[dict[str, Any]]) -> None:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in suggestions:
        service = str(row["expected_service_name_short"])
        for alias in split_semicolon(row["suggested_service_aliases"]):
            grouped[service][alias] += 1

    rows: list[dict[str, Any]] = []
    for service, counter in sorted(grouped.items()):
        rows.append(
            {
                "expected_service_name_short": service,
                "suggested_aliases_ranked": "; ".join(
                    f"{alias} ({count})" for alias, count in counter.most_common(20)
                ),
            }
        )
    write_dicts(path, ["expected_service_name_short", "suggested_aliases_ranked"], rows)


def write_runtime_summary(path: Path, suggestions: list[dict[str, Any]]) -> None:
    counter: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for row in suggestions:
        for term in split_semicolon(row["suggested_runtime_vocabulary_terms"]):
            counter[term] += 1
            if len(examples[term]) < 5:
                examples[term].append(str(row["question_text"]))

    rows = [
        {
            "term": term,
            "count": count,
            "example_questions": " | ".join(examples[term]),
            "where_to_edit": "app/config/runtime_vocabulary.json",
        }
        for term, count in counter.most_common()
    ]
    write_dicts(path, ["term", "count", "example_questions", "where_to_edit"], rows)


def write_readme(path: Path, responses_path: Path, suggestions: list[dict[str, Any]]) -> None:
    text = f"""# Предложения по алиасам

Источник: `{responses_path}`

Строк с предложениями: {len(suggestions)}

Как использовать:

1. Если слово относится к конкретной услуге, добавляй его в `Актуальный_приказ5.xlsx`, колонка `Ключевые слова / алиасы`.
2. После правки Excel выполни импорт `service_registry` и перезапусти API.
3. Если слово является общим бытовым синонимом, добавляй его в `app/config/runtime_vocabulary.json`.
4. После правки `runtime_vocabulary.json` достаточно перезапустить API.
5. Не добавляй все предложения автоматически: это подсказки для ручной проверки.
"""
    path.write_text(text, encoding="utf-8")


def write_dicts(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def split_semicolon(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def tokenize(value: str) -> list[str]:
    return [normalize_token(token) for token in re.findall(r"[a-zа-я0-9]+", value.casefold().replace("ё", "е"))]


def normalize_token(token: str) -> str:
    token = token.casefold().replace("ё", "е")
    suffixes = (
        "иями", "ями", "ами", "ого", "его", "ому", "ему", "ыми", "ими", "ых", "их",
        "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий", "ой", "ов", "ев", "ей",
        "ам", "ям", "ах", "ях", "ом", "ем", "ия", "ии", "ию", "а", "я", "ы", "и", "у", "ю", "е",
    )
    if token.isdigit() or len(token) <= 4:
        return token
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]
    return token


def normalize_text(value: Any) -> str:
    text = str(value or "").casefold().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
