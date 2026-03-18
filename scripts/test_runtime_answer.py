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
from app.services.answers.answer_orchestrator import AnswerOrchestrator
from app.services.generation.generation_pipeline import GenerationRequest
from app.services.retrieval.retrieval_orchestrator import RetrievalInput


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
    manager = DatabaseSessionManager(settings.database)
    manager.initialize()
    await manager.check_connection()

    try:
        async with manager.session_scope() as session:
            orchestrator = AnswerOrchestrator(session)

            retrieval_input = RetrievalInput(
                question_event_id=uuid4(),
                question_text_raw=question_text,
                question_text_normalized=question_text,
                intent_type=intent,
                measure_code="edv" if "едв" in question_text.lower() else None,
                query_terms=[],
                constraints_json={},
                top_k_facts=12,
                top_k_tables=12,
                top_k_rows=16,
                top_k_blocks=12,
                final_top_k=12,
            )

            generation_request = GenerationRequest(
                session_id=uuid4(),
                question_event_id=retrieval_input.question_event_id,
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
            )

            result = await orchestrator.run_runtime_answer(
                retrieval_input=retrieval_input,
                generation_request=generation_request,
            )

            print(json.dumps({
                "answer_mode": getattr(result, "answer_mode", None).value if getattr(result, "answer_mode", None) else None,
                "answer_text": getattr(result, "answer_text", None),
                "citations_json": getattr(result, "citations_json", None),
                "answer_payload_json": getattr(result, "answer_payload_json", None),
                "reuse_decision_payload_json": getattr(result, "reuse_decision_payload_json", None),
            }, ensure_ascii=False, indent=2, default=str))

        return 0
    finally:
        await manager.dispose()


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