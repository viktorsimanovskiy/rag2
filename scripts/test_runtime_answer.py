from __future__ import annotations

import argparse
import asyncio
import sys
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
        "deadline_payment": "срок выплаты едв",
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


def _safe_getattr(obj: object, name: str, default=None):
    return getattr(obj, name, default)


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


async def run(
    *,
    question_text: str,
    intent: QuestionIntentEnum,
) -> None:
    settings = load_settings()
    runtime = AppRuntime(
        config=AppRuntimeConfig(
            database=settings.database,
        )
    )

    await runtime.startup()
    try:
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

            debug_result = _result_to_debug_dict(result)

            print("=" * 80)
            print("QUESTION:")
            print(question_text)
            print()
            print("INTENT:")
            print(getattr(intent, "value", str(intent)))
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
            print("=" * 80)
    finally:
        await runtime.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test for runtime answer path"
    )
    parser.add_argument(
        "--intent",
        required=True,
        help="documents | deadline | procedure | refusal",
    )
    parser.add_argument(
        "--question",
        required=False,
        help="Raw user question text",
    )
    parser.add_argument(
        "--preset",
        required=False,
        help="Question preset: documents, documents_epgu, deadline, deadline_decision, deadline_review, deadline_notification, deadline_payment, procedure, refusal",
    )
    args = parser.parse_args()

    intent = _parse_intent(args.intent)
    question_text = _resolve_question(args.question, args.preset)

    asyncio.run(
        run(
            question_text=question_text,
            intent=intent,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())