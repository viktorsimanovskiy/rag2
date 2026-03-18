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
from app.runtime.app_runtime import AppRuntime, AppRuntimeConfig
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


def _parse_intent(value: str) -> QuestionIntentEnum:
    normalized = value.strip().lower()
    for key, (_, intent) in QUESTION_PRESETS.items():
        if key == normalized:
            return intent
    for item in QuestionIntentEnum:
        if item.value == normalized:
            return item
    raise ValueError(f"Unknown intent: {value}")


def _resolve_question(raw_question: str | None, preset: str | None) -> str:
    if raw_question and raw_question.strip():
        return raw_question.strip()
    if preset and preset in QUESTION_PRESETS:
        return QUESTION_PRESETS[preset][0]
    raise ValueError("question is required if preset is not used")


async def run(question_text: str, intent: QuestionIntentEnum) -> int:
    settings = load_settings()

    runtime = AppRuntime(
        AppRuntimeConfig(
            database=settings.database,
        )
    )
    await runtime.startup()

    try:
        async with runtime.session_scope() as session:
            factory = runtime.build_service_factory(session)
            runtime_answer_service = factory.get_runtime_answer_service()

            payload = RuntimeAnswerInput(
                session_id=uuid4(),
                question_event_id=uuid4(),
                channel_code=ChannelTypeEnum.TELEGRAM,
                question_text_raw=question_text,
                question_text_normalized=question_text,
                language_code="ru",
                intent_type=intent,
                measure_code="edv" if "едв" in question_text.lower() else None,
                subject_category_code=None,
                routing_payload_json={},
                query_constraints_json={},
                request_metadata_json={},
                query_terms=[],
                top_k_facts=12,
                top_k_tables=12,
                top_k_rows=16,
                top_k_blocks=12,
                final_top_k=12,
            )

            result = await runtime_answer_service.build_answer(payload)

            print(json.dumps({
                "answer_mode": result.generation_result.answer_mode.value if result.generation_result.answer_mode else None,
                "answer_text": result.generation_result.answer_text,
                "citations_json": result.generation_result.citations_json,
                "answer_payload_json": result.generation_result.answer_payload_json,
                "reuse_decision_payload_json": result.generation_result.reuse_decision_payload_json,
                "runtime_payload_json": result.runtime_payload_json,
            }, ensure_ascii=False, indent=2, default=str))

        return 0
    finally:
        await runtime.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full runtime answer smoke test.")
    parser.add_argument("--question", default=None)
    parser.add_argument("--preset", default=None, choices=list(QUESTION_PRESETS.keys()))
    parser.add_argument("--intent", default="documents")
    args = parser.parse_args()

    intent = _parse_intent(args.intent)
    question_text = _resolve_question(args.question, args.preset)
    return asyncio.run(run(question_text, intent))


if __name__ == "__main__":
    raise SystemExit(main())