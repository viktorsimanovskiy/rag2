from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import load_settings  # noqa: E402
from app.db.models.enums import QuestionIntentEnum  # noqa: E402
from app.db.session import DatabaseSessionManager  # noqa: E402
from app.services.answers.intent_classifier import RuleBasedIntentClassifier  # noqa: E402
from app.services.retrieval.service_discovery import (  # noqa: E402
    ServiceDiscovery,
    ServiceDiscoveryInput,
)
from app.services.retrieval.applicant_category_taxonomy import normalize_text  # noqa: E402


DEFAULT_JSON_REPORT = Path("/home/logs/service_discovery_diagnostics/report.json")
DEFAULT_TEXT_REPORT = Path("/home/logs/service_discovery_diagnostics/report.txt")

DEFAULT_QUESTIONS = [
    "я мать-одиночка с тремя несовершеннолетними детьми что мне положено",
    "я почетный донор россии что мне положено",
    "я участник вов что мне положено",
    "я инвалид вов что мне положено",
    "меры поддержки для ветеранов труда",
    "у меня стаж работы в крае 40 лет, что мне положено",
    "я пенсионер, мне положена ЕДВ",
    "у меня сгорел дом, помогите",
    "мне нечего есть, прошу помочь",
    "дайте денег на дрова",
    "не хватает денег собрать детей в школу, окажите помощь",
    "положена ли субсидия многодетным",
]


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").split())


def json_safe(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, QuestionIntentEnum):
        return value.value
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def read_questions(args: argparse.Namespace) -> list[str]:
    questions: list[str] = []

    if args.questions_file:
        path = Path(args.questions_file).expanduser().resolve()
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            payload = json.loads(raw)
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, str):
                        questions.append(item)
                    elif isinstance(item, dict):
                        value = item.get("question") or item.get("question_text") or item.get("text")
                        if value:
                            questions.append(str(value))
            elif isinstance(payload, dict):
                for item in payload.get("questions") or payload.get("cases") or []:
                    if isinstance(item, str):
                        questions.append(item)
                    elif isinstance(item, dict):
                        value = item.get("question") or item.get("question_text") or item.get("text")
                        if value:
                            questions.append(str(value))
        else:
            for line in raw.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    questions.append(line)

    for item in args.question or []:
        if item and item.strip():
            questions.append(item.strip())

    if not questions:
        questions = list(DEFAULT_QUESTIONS)

    result: list[str] = []
    seen: set[str] = set()
    for question in questions:
        normalized = clean_text(question)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def classifier_to_json(classification: dict[str, Any]) -> dict[str, Any]:
    routing_payload = classification.get("routing_payload_json") or {}
    constraints = classification.get("query_constraints_json") or {}
    intent = classification.get("intent_type")
    return {
        "intent_type": intent.value if isinstance(intent, QuestionIntentEnum) else str(intent),
        "classifier_version": classification.get("classifier_version"),
        "confidence": routing_payload.get("confidence"),
        "requires_service_discovery": bool(
            constraints.get("requires_service_discovery")
            or routing_payload.get("requires_service_discovery")
        ),
        "routing_mode": constraints.get("routing_mode"),
        "avoid_single_service_resolution": bool(constraints.get("avoid_single_service_resolution")),
        "matched_rules": routing_payload.get("matched_rules") or [],
        "chosen_rules": routing_payload.get("chosen_rules") or [],
        "query_constraints_json": json_safe(constraints),
        "routing_payload_json": json_safe(routing_payload),
    }


def row_to_json(row: Any) -> dict[str, Any]:
    return {
        "row_id": str(row.row_id),
        "row_order": row.row_order,
        "score": row.score,
        "table_id": str(row.table_id),
        "table_number": row.table_number,
        "table_title": row.table_title,
        "applicant_category_id": row.applicant_category_id,
        "applicant_category_name": row.applicant_category_name,
        "row_summary": row.row_summary,
        "matched_signal_codes": list(row.matched_signal_codes),
        "matched_signal_labels": list(row.matched_signal_labels),
        "matched_terms": list(row.matched_terms),
        "citation_json": json_safe(row.citation_json),
        "metadata_json": json_safe(row.metadata_json),
    }


def candidate_to_json(candidate: Any) -> dict[str, Any]:
    return {
        "service_key": candidate.service_key,
        "service_name_short": candidate.service_name_short,
        "service_name_full": candidate.service_name_full,
        "service_frgu_1": candidate.service_frgu_1,
        "service_frgu_3": candidate.service_frgu_3,
        "document_id": str(candidate.document_id),
        "document_name": candidate.document_name,
        "original_filename": candidate.original_filename,
        "score": candidate.score,
        "matched_signal_codes": list(candidate.matched_signal_codes),
        "matched_signal_labels": list(candidate.matched_signal_labels),
        "matched_terms": list(candidate.matched_terms),
        "matched_rows": [row_to_json(row) for row in candidate.matched_rows],
    }


def signal_to_json(signal: Any) -> dict[str, Any]:
    return {
        "code": signal.code,
        "label": signal.label,
        "matched_question_patterns": list(signal.matched_question_patterns),
        "evidence_terms": list(signal.evidence_terms),
        "weight": signal.weight,
    }


def discovery_to_json(result: Any, *, include_answer: bool) -> dict[str, Any]:
    payload = {
        "can_answer": result.can_answer,
        "answer_text_short": result.answer_text_short,
        "warnings": list(result.warnings),
        "signals": [signal_to_json(signal) for signal in result.signals],
        "candidates_count": len(result.candidates),
        "candidates": [candidate_to_json(candidate) for candidate in result.candidates],
        "debug_payload_json": json_safe(result.debug_payload_json),
        "citations_json": json_safe(result.citations_json),
    }
    if include_answer:
        payload["answer_text"] = result.answer_text
    return payload


async def run_one_question(
    *,
    question: str,
    classifier: RuleBasedIntentClassifier,
    discovery: ServiceDiscovery,
    max_services: int,
    max_rows_per_service: int,
    min_score: float,
    include_answer: bool,
) -> dict[str, Any]:
    classification = await classifier.classify(question)
    classification_json = classifier_to_json(classification)

    result = await discovery.discover(
        ServiceDiscoveryInput(
            question_text_raw=question,
            question_text_normalized=normalize_text(question),
            max_services=max_services,
            max_rows_per_service=max_rows_per_service,
            min_score=min_score,
        )
    )

    return {
        "question": question,
        "normalized_question": normalize_text(question),
        "classifier": classification_json,
        "discovery": discovery_to_json(result, include_answer=include_answer),
    }


async def build_report(args: argparse.Namespace) -> dict[str, Any]:
    questions = read_questions(args)
    settings = load_settings()
    manager = DatabaseSessionManager(settings.database)
    manager.initialize()

    try:
        await manager.check_connection()
        async with manager.session_scope() as session:
            classifier = RuleBasedIntentClassifier()
            discovery = ServiceDiscovery(session)
            cases: list[dict[str, Any]] = []
            for question in questions:
                cases.append(
                    await run_one_question(
                        question=question,
                        classifier=classifier,
                        discovery=discovery,
                        max_services=args.max_services,
                        max_rows_per_service=args.max_rows_per_service,
                        min_score=args.min_score,
                        include_answer=args.include_answer,
                    )
                )
    finally:
        await manager.dispose()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "diagnose service discovery candidates and their evidence rows",
        "settings": {
            "max_services": args.max_services,
            "max_rows_per_service": args.max_rows_per_service,
            "min_score": args.min_score,
            "show_raw_row_summary": bool(args.show_raw_row_summary),
        },
        "questions_count": len(questions),
        "cases": cases,
    }


def render_text_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("ДИАГНОСТИКА ПОДБОРА ВОЗМОЖНЫХ МЕР")
    lines.append("=" * 100)
    lines.append(f"Сформировано: {report.get('generated_at')}")
    lines.append(f"Вопросов: {report.get('questions_count')}")
    settings = report.get("settings") or {}
    lines.append(
        "Параметры: "
        f"услуг до {settings.get('max_services')}, "
        f"строк на услугу до {settings.get('max_rows_per_service')}, "
        f"минимальный балл {settings.get('min_score')}"
    )
    lines.append("")
    lines.append(
        "Важно: отчёт показывает, почему услуга попала в список возможных направлений. "
        "Это не вывод о праве заявителя."
    )
    lines.append("")

    for index, case in enumerate(report.get("cases") or [], start=1):
        classifier = case.get("classifier") or {}
        discovery = case.get("discovery") or {}
        lines.append(f"## {index}. {case.get('question')}")
        lines.append(f"Нормализованный вопрос: {case.get('normalized_question')}")
        lines.append(
            "Классификатор: "
            f"тип={classifier.get('intent_type')}; "
            f"уверенность={classifier.get('confidence')}; "
            f"подбор мер={classifier.get('requires_service_discovery')}; "
            f"режим={classifier.get('routing_mode') or '-'}"
        )
        rules = classifier.get("chosen_rules") or classifier.get("matched_rules") or []
        if rules:
            lines.append("Сработавшие правила: " + ", ".join(str(item) for item in rules))

        signals = discovery.get("signals") or []
        if signals:
            lines.append("Распознанные признаки заявителя:")
            for signal in signals:
                lines.append(
                    f"- {signal.get('label')} "
                    f"[{signal.get('code')}], вес {signal.get('weight')}"
                )
        else:
            lines.append("Распознанные признаки заявителя: нет")

        debug = discovery.get("debug_payload_json") or {}
        lines.append(
            "Итог подбора: "
            f"можно ответить={discovery.get('can_answer')}; "
            f"кандидатов={discovery.get('candidates_count')}; "
            f"просмотрено строк={debug.get('scanned_identifier_rows_count')}; "
            f"совпавших строк={debug.get('matched_identifier_rows_count')}"
        )

        warnings = discovery.get("warnings") or []
        if warnings:
            lines.append("Предупреждения:")
            for warning in warnings:
                lines.append(f"- {warning}")

        candidates = discovery.get("candidates") or []
        if not candidates:
            lines.append("Найденные услуги: нет")
            lines.append("")
            continue

        lines.append("Найденные услуги:")
        for rank, candidate in enumerate(candidates, start=1):
            title = candidate.get("service_name_short") or candidate.get("service_name_full") or candidate.get("service_key")
            lines.append(f"{rank}) {title}")
            lines.append(f"   service_key: {candidate.get('service_key')}")
            if candidate.get("service_frgu_1") or candidate.get("service_frgu_3"):
                lines.append(f"   ФРГУ: {candidate.get('service_frgu_1') or '-'} / {candidate.get('service_frgu_3') or '-'}")
            lines.append(f"   Балл услуги: {candidate.get('score')}")
            labels = candidate.get("matched_signal_labels") or []
            if labels:
                lines.append("   Совпавшие признаки: " + "; ".join(str(item) for item in labels))
            terms = candidate.get("matched_terms") or []
            if terms:
                lines.append("   Совпавшие слова/фразы: " + "; ".join(str(item) for item in terms[:18]))
            rows = candidate.get("matched_rows") or []
            lines.append("   Строки-основания:")
            for row in rows:
                category = row.get("applicant_category_name") or row.get("row_summary") or "<нет текста категории>"
                lines.append(
                    f"   - строка {row.get('row_order')}, балл {row.get('score')}: {category}"
                )
                if row.get("applicant_category_id"):
                    lines.append(f"     идентификатор категории: {row.get('applicant_category_id')}")
                if (
                    settings.get("show_raw_row_summary")
                    and row.get("row_summary")
                    and row.get("row_summary") != category
                ):
                    lines.append(f"     исходное краткое описание строки: {row.get('row_summary')}")
                row_terms = row.get("matched_terms") or []
                if row_terms:
                    lines.append("     совпадения: " + "; ".join(str(item) for item in row_terms[:18]))
                if row.get("table_number") or row.get("table_title"):
                    lines.append(f"     таблица: {row.get('table_number') or '-'}; {row.get('table_title') or '-'}")
        if discovery.get("answer_text_short"):
            lines.append("")
            lines.append("Короткий текст ответа:")
            lines.append(str(discovery.get("answer_text_short")))
        lines.append("")

    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    report = await build_report(args)

    json_path = Path(args.json_report).expanduser().resolve()
    text_path = Path(args.text_report).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    text_report = render_text_report(report)
    text_path.write_text(text_report, encoding="utf-8")

    if not args.quiet:
        print(text_report)
        print(f"\nJSON сохранён: {json_path}")
        print(f"Текстовый отчёт сохранён: {text_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Показать, почему service discovery выбрал конкретные услуги и строки категорий заявителей."
    )
    parser.add_argument("--question", action="append", help="Отдельный вопрос. Можно указать несколько раз.")
    parser.add_argument("--questions-file", help="Файл с вопросами: TXT по одному вопросу на строку или JSON.")
    parser.add_argument("--max-services", type=int, default=10, help="Сколько услуг показывать по каждому вопросу.")
    parser.add_argument("--max-rows-per-service", type=int, default=5, help="Сколько строк-оснований показывать на услугу.")
    parser.add_argument("--min-score", type=float, default=2.0, help="Минимальный балл строки для включения в подбор.")
    parser.add_argument("--json-report", default=str(DEFAULT_JSON_REPORT), help="Куда сохранить JSON-отчёт.")
    parser.add_argument("--text-report", default=str(DEFAULT_TEXT_REPORT), help="Куда сохранить текстовый отчёт.")
    parser.add_argument("--include-answer", action="store_true", help="Включить в JSON полный текст ответа.")
    parser.add_argument(
        "--show-raw-row-summary",
        action="store_true",
        help="Показывать в текстовом отчёте исходное краткое описание строки. По умолчанию скрыто, чтобы отчёт не засорялся служебным текстом таблиц.",
    )
    parser.add_argument("--quiet", action="store_true", help="Не печатать отчёт в консоль.")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
