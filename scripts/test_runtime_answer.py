from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import load_settings
from app.db.models.enums import ChannelTypeEnum, QuestionIntentEnum
from app.db.session import DatabaseSessionManager
from app.runtime.app_runtime import build_app_runtime
from app.services.answers.runtime_answer_service import RuntimeAnswerInput


QUESTION_PRESETS: dict[str, tuple[str, QuestionIntentEnum]] = {
    "documents": (
        "какие документы нужны для едв",
        QuestionIntentEnum.DOCUMENTS_QUESTION,
    ),
    "documents_epgu": (
        "какие документы нужны для едв при подаче через епгу",
        QuestionIntentEnum.DOCUMENTS_QUESTION,
    ),
}


def _parse_intent(value: str):
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
        "deadline_review": "срок рассмотрения заявления по едв",
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


async def run(
    *,
    question_text: str,
    intent,
) -> None:
    runtime = await build_app_runtime()
    try:
        service = runtime.runtime_answer_service

        result = await service.build_answer(
            RuntimeAnswerInput(
                question_text_raw=question_text,
                intent_type=intent,
            )
        )

        print("=" * 80)
        print("QUESTION:")
        print(question_text)
        print()
        print("INTENT:")
        print(getattr(intent, "value", str(intent)))
        print()
        print("ANSWER:")
        print(result.answer_text or "<empty>")
        print()
        print("ANSWER MODE:")
        print(getattr(result.answer_mode, "value", str(result.answer_mode)))
        print()
        print("CONFIDENCE:")
        print(result.confidence_score)
        print()
        print("CITATIONS:")
        if not result.citations:
            print("<none>")
        else:
            for index, citation in enumerate(result.citations, start=1):
                print(f"{index}. {citation}")
        print()
        print("PAYLOAD JSON:")
        print(result.answer_payload_json)
        print("=" * 80)
    finally:
        await runtime.dispose()


def main() -> None:
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
        help="Question preset: documents, documents_epgu, deadline, deadline_review, procedure, refusal",
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


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    raise SystemExit(main())