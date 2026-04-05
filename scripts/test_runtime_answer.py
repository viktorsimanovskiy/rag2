from __future__ import annotations

import argparse
import asyncio
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


_MEASURE_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "edv": (
        " едв ",
        "ежемесячной денежной выплаты",
        "ежемесячная денежная выплата",
    ),
    "subsidy": (
        "субсид",
        "оплату жилого помещения",
        "коммунальных услуг",
    ),
    "social_contract": (
        "соцконтракт",
        "социального контракта",
        "социальный контракт",
    ),
    "hardship": (
        " тжс ",
        "трудной жизненной ситуации",
        "адресной материальной помощи",
    ),
    "sanatorium": (
        "санкур",
        "санаторно-курорт",
        "бесплатных путевок",
        "путевок на санаторно-курортное лечение",
    ),
}


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


def _infer_measure_code(question_text: str) -> str | None:
    normalized = " ".join((question_text or "").strip().lower().split())
    if not normalized:
        return None

    padded = f" {normalized} "
    for measure_code, aliases in _MEASURE_ALIAS_MAP.items():
        if any(alias in padded or alias in normalized for alias in aliases):
            return measure_code

    return None


def _resolve_measure_code_for_question(
    *,
    question_text: str,
    explicit_measure_code: str | None,
) -> str | None:
    if explicit_measure_code:
        return explicit_measure_code.strip().lower() or None
    return _infer_measure_code(question_text)


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


async def _run_one_question(
    *,
    runtime: AppRuntime,
    question_text: str,
    intent: QuestionIntentEnum,
    question_index: int,
    total_questions: int,
    explicit_measure_code: str | None,
) -> None:
    started_at = time.perf_counter()

    async with runtime.session_scope() as session:
        service_factory = runtime.build_service_factory(session)
        service = service_factory.get_runtime_answer_service()

        normalized_question = " ".join(question_text.strip().lower().split())
        measure_code = _resolve_measure_code_for_question(
            question_text=normalized_question,
            explicit_measure_code=explicit_measure_code,
        )

        result = await service.build_answer(
            RuntimeAnswerInput(
                session_id=uuid4(),
                question_event_id=uuid4(),
                channel_code="CLI_TEST",
                question_text_raw=question_text,
                question_text_normalized=normalized_question,
                language_code="ru",
                intent_type=intent,
                measure_code=measure_code,
            )
        )

    elapsed = time.perf_counter() - started_at
    debug_result = _result_to_debug_dict(result)

    print("=" * 100)
    print(f"QUESTION {question_index}/{total_questions}")
    print("-" * 100)
    print(question_text)
    print()
    print("INTENT:")
    print(getattr(intent, "value", str(intent)))
    print()
    print("MEASURE CODE:")
    print(measure_code)
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


async def run(
    *,
    question_texts: list[str],
    intent: QuestionIntentEnum,
    explicit_measure_code: str | None,
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
                explicit_measure_code=explicit_measure_code,
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
        "--measure-code",
        required=False,
        help=(
            "Explicit measure code override for all questions in the batch. "
            "If omitted, the script will try to infer it from question text."
        ),
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
            explicit_measure_code=args.measure_code,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())