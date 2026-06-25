from __future__ import annotations

import argparse
import asyncio
import json
import sys
from time import perf_counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import load_settings
from app.db.session import DatabaseSessionManager
from app.services.retrieval.service_resolver import ServiceResolver, ServiceResolverInput


EXAMPLE_QUESTIONS = [
    "какие документы нужны для субсидии",
    "причины отказа в едв",
    "я ветеран труда края какие документы нужны для получения едв",
    "как получить соцконтракт",
    "что считается трудной жизненной ситуацией",
    "почему могут отказать в предоставлении санкура для ребенка",
    "кто может подать заявление на санаторно-курортное лечение",
    "я мать-одиночка с тремя несовершеннолетними детьми что мне положено",
    "у меня сгорел дом, помогите",
    "нужна ли прописка в красноярском крае для получения соцконтракта",
]


def candidate_to_dict(candidate: Any) -> dict[str, Any]:
    return {
        "service_key": candidate.service_key,
        "service_name_short": candidate.service_name_short,
        "score": candidate.score,
        "confidence": candidate.confidence,
        "frgu_1": candidate.frgu_1,
        "frgu_3": candidate.frgu_3,
        "cleaned_filename": candidate.cleaned_filename,
        "matched_terms": candidate.matched_terms,
        "matched_aliases": candidate.matched_aliases,
        "match_reasons": candidate.match_reasons,
    }


def result_to_dict(question: str, result: Any) -> dict[str, Any]:
    return {
        "question": question,
        "resolution_status": result.resolution_status,
        "selected_service_key": result.service_key,
        "selected_service_name_short": (
            result.selected_service.service_name_short
            if result.selected_service is not None
            else None
        ),
        "candidates": [candidate_to_dict(candidate) for candidate in result.candidates],
        "debug_payload_json": result.debug_payload_json,
    }


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0.001:
        return f"{value * 1000:.2f} мс"
    if value < 1:
        return f"{value * 1000:.1f} мс"
    return f"{value:.3f} с"


def print_human_result(question: str, result: Any, *, top_k: int, show_timing: bool) -> None:
    timings = result.debug_payload_json.get("timings_sec", {})

    print("=" * 100)
    print(f"ВОПРОС: {question}")
    print(f"СТАТУС: {result.resolution_status}")
    if show_timing:
        print(
            "ВРЕМЯ: "
            f"итого={_format_seconds(timings.get('total'))}; "
            f"индекс={_format_seconds(timings.get('load_or_get_index'))}; "
            f"скоринг={_format_seconds(timings.get('score_candidates'))}; "
            f"кэш индекса={result.debug_payload_json.get('index_cache_hit', False)}"
        )
    if result.selected_service is not None:
        print(
            "ВЫБРАНА УСЛУГА: "
            f"{result.selected_service.service_name_short} | "
            f"score={result.selected_service.score} | "
            f"service_key={result.selected_service.service_key}"
        )
    else:
        print("ВЫБРАНА УСЛУГА: —")

    print("-" * 100)
    print("КАНДИДАТЫ:")
    for index, candidate in enumerate(result.candidates[:top_k], start=1):
        aliases = "; ".join(candidate.matched_aliases[:3]) or "—"
        terms = ", ".join(candidate.matched_terms[:12]) or "—"
        print(
            f"{index}. {candidate.service_name_short} | "
            f"score={candidate.score} | confidence={candidate.confidence}"
        )
        print(f"   service_key: {candidate.service_key}")
        print(f"   совпавшие термины: {terms}")
        print(f"   совпавшие алиасы: {aliases}")
    print("=" * 100)


def load_questions(args: argparse.Namespace) -> list[str]:
    questions: list[str] = []

    for item in args.question or []:
        text = " ".join(item.split()).strip()
        if text:
            questions.append(text)

    if args.questions_file:
        path = Path(args.questions_file)
        for line in path.read_text(encoding="utf-8").splitlines():
            text = " ".join(line.split()).strip()
            if text and not text.startswith("#"):
                questions.append(text)

    if args.examples:
        questions.extend(EXAMPLE_QUESTIONS)

    if not questions:
        questions.extend(EXAMPLE_QUESTIONS)

    seen: set[str] = set()
    result: list[str] = []
    for question in questions:
        key = question.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(question)
    return result


async def run(args: argparse.Namespace) -> int:
    run_started_at = perf_counter()

    settings_started_at = perf_counter()
    settings = load_settings()
    settings_elapsed = perf_counter() - settings_started_at

    initialize_started_at = perf_counter()
    manager = DatabaseSessionManager(settings.database)
    manager.initialize()
    initialize_elapsed = perf_counter() - initialize_started_at

    db_check_started_at = perf_counter()
    await manager.check_connection()
    db_check_elapsed = perf_counter() - db_check_started_at

    load_questions_started_at = perf_counter()
    questions = load_questions(args)
    load_questions_elapsed = perf_counter() - load_questions_started_at

    output_items: list[dict[str, Any]] = []
    questions_elapsed_total = 0.0
    dispose_elapsed = 0.0

    try:
        async with manager.session_scope() as session:
            resolver = ServiceResolver(session)

            for question in questions:
                question_started_at = perf_counter()
                result = await resolver.resolve(
                    ServiceResolverInput(
                        question_text=question,
                        max_candidates=args.top_k,
                        min_resolved_score=args.min_resolved_score,
                        min_candidate_score=args.min_candidate_score,
                        ambiguity_margin=args.ambiguity_margin,
                    )
                )
                question_elapsed = perf_counter() - question_started_at
                questions_elapsed_total += question_elapsed

                item = result_to_dict(question, result)
                item["script_timing_sec"] = round(question_elapsed, 6)
                output_items.append(item)

                if not args.json:
                    print_human_result(
                        question,
                        result,
                        top_k=args.top_k,
                        show_timing=not args.no_timing,
                    )
    finally:
        dispose_started_at = perf_counter()
        await manager.dispose()
        dispose_elapsed = perf_counter() - dispose_started_at

    total_elapsed = perf_counter() - run_started_at
    summary = {
        "settings": round(settings_elapsed, 6),
        "initialize_db_manager": round(initialize_elapsed, 6),
        "check_db_connection": round(db_check_elapsed, 6),
        "load_questions": round(load_questions_elapsed, 6),
        "resolve_questions_total": round(questions_elapsed_total, 6),
        "dispose_db": round(dispose_elapsed, 6),
        "total": round(total_elapsed, 6),
        "questions_count": len(questions),
    }

    if args.json:
        print(
            json.dumps(
                {"items": output_items, "script_timing_sec": summary},
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    elif not args.no_timing:
        print("=" * 100)
        print(
            "ИТОГО ПО СКРИПТУ: "
            f"{_format_seconds(total_elapsed)}; "
            f"подключение к БД={_format_seconds(db_check_elapsed)}; "
            f"вопросы={_format_seconds(questions_elapsed_total)}; "
            f"завершение БД={_format_seconds(dispose_elapsed)}; "
            f"количество вопросов={len(questions)}"
        )
        print("=" * 100)

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test service resolver on service_registry.")
    parser.add_argument(
        "--question",
        action="append",
        help="Question text. Can be passed multiple times.",
    )
    parser.add_argument(
        "--questions-file",
        help="UTF-8 text file with one question per line.",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run built-in example questions.",
    )
    parser.add_argument("--top-k", type=int, default=7, help="How many candidates to print.")
    parser.add_argument("--json", action="store_true", help="Print JSON output only.")
    parser.add_argument("--min-resolved-score", type=float, default=72.0)
    parser.add_argument("--min-candidate-score", type=float, default=18.0)
    parser.add_argument("--ambiguity-margin", type=float, default=14.0)
    parser.add_argument(
        "--no-timing",
        action="store_true",
        help="Do not print timing counters in human-readable output.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(build_arg_parser().parse_args())))
