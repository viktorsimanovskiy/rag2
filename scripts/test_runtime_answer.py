from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import load_settings
from app.db.models.enums import QuestionIntentEnum
from app.runtime.app_runtime import AppRuntime, AppRuntimeConfig
from app.services.answers.runtime_answer_service import RuntimeAnswerInput


def _parse_intent(value: str) -> QuestionIntentEnum:
    normalized = (value or "").strip().lower()

    mapping = {
        "documents": QuestionIntentEnum.DOCUMENTS_QUESTION,
        "documents_question": QuestionIntentEnum.DOCUMENTS_QUESTION,
        "docs": QuestionIntentEnum.DOCUMENTS_QUESTION,
        "deadline": QuestionIntentEnum.DEADLINE_QUESTION,
        "deadlines": QuestionIntentEnum.DEADLINE_QUESTION,
        "deadline_question": QuestionIntentEnum.DEADLINE_QUESTION,
        "procedure": QuestionIntentEnum.PROCEDURE_QUESTION,
        "procedure_question": QuestionIntentEnum.PROCEDURE_QUESTION,
        "refusal": QuestionIntentEnum.REJECTION_QUESTION,
        "refusal_reasons": QuestionIntentEnum.REJECTION_QUESTION,
        "refusal_reasons_question": QuestionIntentEnum.REJECTION_QUESTION,
        "rejection": QuestionIntentEnum.REJECTION_QUESTION,
        "rejection_question": QuestionIntentEnum.REJECTION_QUESTION,
    }

    if normalized not in mapping:
        supported = ", ".join(sorted(mapping.keys()))
        raise ValueError(
            f"Unsupported intent '{value}'. Supported values: {supported}"
        )

    return mapping[normalized]


def _resolve_question(raw_question: str | None, preset: str | None) -> str:
    if raw_question and raw_question.strip():
        return raw_question.strip()

    normalized_preset = (preset or "").strip().lower()

    preset_questions = {
        "documents": "какие документы нужны для едв",
        "documents_epgu": "какие документы нужны для едв при подаче через епгу",
        "deadline": "срок принятия решения по едв",
        "deadline_decision": "срок принятия решения по едв",
        "deadline_review": "срок рассмотрения заявления по едв",
        "deadline_notification": "срок уведомления о решении по едв",
        "deadline_payment": "когда выплатят едв",
        "deadline_registration": "срок регистрации заявления на субсидию",
        "procedure": "как назначается едв",
        "refusal": "по каким основаниям могут отказать в едв",
    }

    if normalized_preset in preset_questions:
        return preset_questions[normalized_preset]

    if normalized_preset:
        raise ValueError(
            f"Unknown preset '{preset}'. Supported presets: {', '.join(sorted(preset_questions.keys()))}"
        )

    return "срок принятия решения по едв"


def _load_questions_from_file(path: str) -> list[str]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Questions file not found: {file_path}")

    questions: list[str] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean:
            continue
        if clean.startswith("#"):
            continue
        questions.append(clean)
    return questions


def _collect_questions(
    *,
    questions: list[str] | None,
    questions_file: str | None,
    preset: str | None,
) -> list[str]:
    result: list[str] = []

    if questions:
        for question in questions:
            clean = (question or "").strip()
            if clean:
                result.append(clean)

    if questions_file:
        result.extend(_load_questions_from_file(questions_file))

    if not result:
        result.append(_resolve_question(None, preset))

    return result


def _result_to_debug_dict(result: object) -> dict:
    if hasattr(result, "__dict__"):
        return dict(vars(result))

    fields = {}
    for name in dir(result):
        if name.startswith("_"):
            continue
        try:
            value = getattr(result, name)
        except Exception:
            continue
        if callable(value):
            continue
        fields[name] = value
    return fields


def _shorten_text(value: str | None, limit: int = 280) -> str:
    if not value:
        return "—"
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _safe_get(mapping: object, key: str, default=None):
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return default


def _get_answer_text(result: object) -> str | None:
    runtime_payload_json = getattr(result, "runtime_payload_json", None) or {}
    generation_result = getattr(result, "generation_result", None)

    for key in (
        "answer_text",
        "final_answer_text",
        "answer_markdown",
        "response_text",
    ):
        value = _safe_get(runtime_payload_json, key)
        if value:
            return str(value)

    for attr in (
        "answer_text",
        "final_answer_text",
        "answer_markdown",
        "response_text",
    ):
        value = getattr(generation_result, attr, None) if generation_result is not None else None
        if value:
            return str(value)

    return None


def _get_generation_meta(result: object) -> dict:
    runtime_payload_json = getattr(result, "runtime_payload_json", None) or {}
    generation_result = getattr(result, "generation_result", None)

    def pick_value(*names: str):
        for name in names:
            value = _safe_get(runtime_payload_json, name)
            if value is not None:
                return value
        for name in names:
            value = getattr(generation_result, name, None) if generation_result is not None else None
            if value is not None:
                return value
        return None

    return {
        "answer_mode": pick_value("answer_mode"),
        "confidence_score": pick_value("confidence_score"),
        "trust_level": pick_value("trust_level"),
    }


def _candidate_brief(candidate: object) -> str:
    source_type = getattr(candidate, "source_type", "?")
    title = getattr(candidate, "title", None) or "без_названия"
    rerank_score = getattr(candidate, "rerank_score", None)
    score = rerank_score if rerank_score is not None else getattr(candidate, "score", None)
    score_text = f"{float(score):.4f}" if score is not None else "—"
    document_name = getattr(candidate, "document_name", None) or "без названия документа"
    snippet = _shorten_text(getattr(candidate, "snippet", None), limit=200)
    return (
        f"[{source_type}] {title} | score={score_text} | {document_name}\n"
        f"    {snippet}"
    )


def _render_compact_result(
    *,
    question_text: str,
    intent: QuestionIntentEnum,
    elapsed: float,
    result: object,
    top_candidates_count: int,
) -> None:
    evidence_package = getattr(result, "evidence_package", None)
    metrics_json = getattr(evidence_package, "metrics_json", {}) or {}
    selected_candidates = list(getattr(evidence_package, "selected_candidates", []) or [])
    strategy_code = getattr(evidence_package, "strategy_code", None)
    answer_text = _get_answer_text(result)
    generation_meta = _get_generation_meta(result)

    best_document_name = None
    if selected_candidates:
        best_document_name = getattr(selected_candidates[0], "document_name", None)

    print("=" * 100)
    print(f"ВОПРОС: {question_text}")
    print(f"ИНТЕНТ: {intent.value}")
    print(f"ВРЕМЯ: {elapsed:.2f} сек")
    if strategy_code:
        print(f"СТРАТЕГИЯ: {strategy_code}")
    print(
        "КАЧЕСТВО: "
        f"{metrics_json.get('evidence_quality')} | "
        f"документов={metrics_json.get('selected_document_ids_count')} | "
        f"сильных={metrics_json.get('strong_candidate_count')} | "
        f"top_share={metrics_json.get('top_document_share')}"
    )
    if generation_meta["answer_mode"] is not None:
        print(f"РЕЖИМ ОТВЕТА: {generation_meta['answer_mode']}")
    if generation_meta["confidence_score"] is not None:
        print(f"УВЕРЕННОСТЬ: {generation_meta['confidence_score']}")
    if generation_meta["trust_level"] is not None:
        print(f"УРОВЕНЬ ДОВЕРИЯ: {generation_meta['trust_level']}")
    if best_document_name:
        print(f"ЛУЧШИЙ ДОКУМЕНТ: {best_document_name}")

    print("-" * 100)
    print("ОТВЕТ:")
    print(_shorten_text(answer_text, limit=1400))

    print("-" * 100)
    print("ЛУЧШИЕ ИСТОЧНИКИ:")
    for index, candidate in enumerate(selected_candidates[:top_candidates_count], start=1):
        print(f"{index}. {_candidate_brief(candidate)}")
    print("=" * 100)
    print()


async def _run_one_question(
    *,
    runtime: AppRuntime,
    question_text: str,
    intent: QuestionIntentEnum,
    question_index: int,
    total_questions: int,
    output_format: str,
    top_candidates_count: int,
) -> None:
    started_at = time.perf_counter()

    async with runtime.session_scope() as session:
        service_factory = runtime.build_service_factory(session)
        service = service_factory.get_runtime_answer_service()

        normalized_question = " ".join(question_text.strip().lower().split())

        result = await service.build_answer(
            RuntimeAnswerInput(
                session_id=uuid4(),
                question_event_id=uuid4(),
                channel_code="CLI_TEST",
                question_text_raw=question_text,
                question_text_normalized=normalized_question,
                language_code="ru",
                intent_type=intent,
            )
        )

    elapsed = time.perf_counter() - started_at

    if output_format == "debug":
        debug_result = _result_to_debug_dict(result)

        print("=" * 100)
        print(f"QUESTION {question_index}/{total_questions}")
        print("-" * 100)
        print(question_text)
        print()
        print("INTENT:")
        print(getattr(intent, "value", str(intent)))
        print()
        print("ELAPSED SECONDS:")
        print(f"{elapsed:.2f}")
        print()
        print("RESULT TYPE:")
        print(type(result).__name__)
        print()
        print("KNOWN RESULT FIELDS:")
        for key in sorted(debug_result.keys()):
            print(f"- {key}")
        print()
        print("RESULT PAYLOAD:")
        print(debug_result)
        print("=" * 100)
        print()
    else:
        _render_compact_result(
            question_text=question_text,
            intent=intent,
            elapsed=elapsed,
            result=result,
            top_candidates_count=top_candidates_count,
        )


async def run(
    *,
    question_texts: list[str],
    intent: QuestionIntentEnum,
    output_format: str,
    top_candidates_count: int,
) -> None:
    settings = load_settings()
    runtime = AppRuntime(
        config=AppRuntimeConfig(
            database=settings.database,
        )
    )

    batch_started_at = time.perf_counter()
    await runtime.startup()

    try:
        total = len(question_texts)
        for idx, question_text in enumerate(question_texts, start=1):
            await _run_one_question(
                runtime=runtime,
                question_text=question_text,
                intent=intent,
                question_index=idx,
                total_questions=total,
                output_format=output_format,
                top_candidates_count=top_candidates_count,
            )
    finally:
        await runtime.shutdown()

    batch_elapsed = time.perf_counter() - batch_started_at
    print("#" * 100)
    print("BATCH FINISHED")
    print(f"TOTAL QUESTIONS: {len(question_texts)}")
    print(f"TOTAL ELAPSED SECONDS: {batch_elapsed:.2f}")
    print("#" * 100)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch smoke-test for runtime answer path with one shared AppRuntime startup"
    )
    parser.add_argument(
        "--intent",
        required=True,
        help="documents | deadline | procedure | refusal",
    )
    parser.add_argument(
        "--question",
        action="append",
        required=False,
        help="Raw user question text. Can be passed multiple times.",
    )
    parser.add_argument(
        "--questions-file",
        required=False,
        help="UTF-8 text file: one question per line.",
    )
    parser.add_argument(
        "--preset",
        required=False,
        help=(
            "Fallback preset if no explicit questions are passed: "
            "documents, documents_epgu, deadline, deadline_decision, "
            "deadline_review, deadline_notification, deadline_payment, "
            "deadline_registration, procedure, refusal"
        ),
    )
    parser.add_argument(
        "--output",
        choices=("compact", "debug"),
        default="compact",
        help="compact = короткая человекочитаемая сводка, debug = старый подробный вывод.",
    )
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=3,
        help="Сколько лучших источников печатать в компактном режиме.",
    )
    args = parser.parse_args()

    intent = _parse_intent(args.intent)
    question_texts = _collect_questions(
        questions=args.question,
        questions_file=args.questions_file,
        preset=args.preset,
    )

    asyncio.run(
        run(
            question_texts=question_texts,
            intent=intent,
            output_format=args.output,
            top_candidates_count=args.top_candidates,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())