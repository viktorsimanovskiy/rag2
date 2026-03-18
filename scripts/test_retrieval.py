from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Iterable
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import load_settings
from app.db.models.enums import QuestionIntentEnum
from app.db.session import DatabaseSessionManager
from app.services.retrieval.retrieval_orchestrator import (
    RetrievalInput,
    RetrievalOrchestrator,
)


QUESTION_PRESETS: dict[str, tuple[str, QuestionIntentEnum]] = {
    "eligibility": (
        "кто имеет право на получение едв",
        QuestionIntentEnum.ELIGIBILITY_QUESTION,
    ),
    "categories": (
        "перечень категорий для получения едв",
        QuestionIntentEnum.ELIGIBILITY_QUESTION,
    ),
    "documents": (
        "список документов для едв",
        QuestionIntentEnum.DOCUMENTS_QUESTION,
    ),
    "rejection": (
        "причины отказа заявителям едв",
        QuestionIntentEnum.REJECTION_QUESTION,
    ),
    "deadline": (
        "срок принятия решения по едв",
        QuestionIntentEnum.DEADLINE_QUESTION,
    ),
    "procedure": (
        "порядок назначения едв",
        QuestionIntentEnum.PROCEDURE_QUESTION,
    ),
}


def _parse_intent(value: str) -> QuestionIntentEnum:
    normalized = value.strip().lower()

    preset = QUESTION_PRESETS.get(normalized)
    if preset is not None:
        return preset[1]

    for item in QuestionIntentEnum:
        if item.value == normalized:
            return item

    raise ValueError(
        "Unknown intent. Use one of: "
        + ", ".join([x.value for x in QuestionIntentEnum])
        + " or one of presets: "
        + ", ".join(QUESTION_PRESETS.keys())
    )


def _resolve_question(raw_question: str | None, intent_value: str) -> str:
    normalized = intent_value.strip().lower()
    preset = QUESTION_PRESETS.get(normalized)

    if raw_question and raw_question.strip():
        return raw_question.strip()

    if preset is not None:
        return preset[0]

    raise ValueError(
        "Question text is required when intent is not one of the presets: "
        + ", ".join(QUESTION_PRESETS.keys())
    )


def _effective_score(candidate: object) -> float:
    rerank_score = getattr(candidate, "rerank_score", None)
    if rerank_score is not None:
        return float(rerank_score)
    return float(getattr(candidate, "score", 0.0) or 0.0)


def _shorten_text(value: str | None, limit: int = 700) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _looks_like_form_noise(text: str | None) -> bool:
    if not text:
        return False

    normalized = text.lower()

    markers = (
        "на основании доверенности",
        "серия, номер",
        "кем выдан",
        "дата выдачи",
        "срок действия полномочий",
        "почтовый адрес места жительства",
        "телефон в федеральном формате",
    )

    return any(marker in normalized for marker in markers)


def _filter_candidates(
    candidates: Iterable[object],
    *,
    document_id: UUID | None,
    exclude_form_noise: bool,
) -> list[object]:
    filtered: list[object] = []

    for candidate in candidates:
        if document_id is not None and getattr(candidate, "document_id", None) != document_id:
            continue

        if exclude_form_noise:
            source_type = getattr(candidate, "source_type", "")
            snippet = getattr(candidate, "snippet", None)
            title = getattr(candidate, "title", None)

            if source_type in {"table", "table_row"} and (
                _looks_like_form_noise(snippet) or _looks_like_form_noise(title)
            ):
                continue

        filtered.append(candidate)

    return filtered


def _render_candidate(candidate: object) -> dict[str, object]:
    return {
        "source_type": getattr(candidate, "source_type", None),
        "source_id": str(getattr(candidate, "source_id", "")),
        "document_id": str(getattr(candidate, "document_id", "")),
        "score": round(float(getattr(candidate, "score", 0.0) or 0.0), 4),
        "effective_score": round(_effective_score(candidate), 4),
        "document_name": getattr(candidate, "document_name", None),
        "doc_uid_base": getattr(candidate, "doc_uid_base", None),
        "revision_date": getattr(candidate, "revision_date", None),
        "title": getattr(candidate, "title", None),
        "snippet": _shorten_text(getattr(candidate, "snippet", None), limit=900),
        "citation_json": getattr(candidate, "citation_json", {}) or {},
        "metadata_json": getattr(candidate, "metadata_json", {}) or {},
    }


async def run(
    *,
    question_text: str,
    intent: QuestionIntentEnum,
    document_id: UUID | None,
    top_k: int,
    exclude_form_noise: bool,
) -> int:
    settings = load_settings()

    manager = DatabaseSessionManager(settings.database)
    manager.initialize()
    await manager.check_connection()

    try:
        async with manager.session_scope() as session:
            orchestrator = RetrievalOrchestrator(session)

            payload = RetrievalInput(
                question_event_id=uuid4(),
                question_text_raw=question_text,
                question_text_normalized=question_text,
                intent_type=intent,
                measure_code="edv" if "едв" in question_text.lower() else None,
                query_terms=[],
                constraints_json={},
                top_k_facts=max(8, top_k),
                top_k_tables=max(8, top_k),
                top_k_rows=max(12, top_k),
                top_k_blocks=max(12, top_k),
                final_top_k=max(12, top_k),
            )

            evidence = await orchestrator.retrieve(payload)

            filtered_candidates = _filter_candidates(
                evidence.selected_candidates,
                document_id=document_id,
                exclude_form_noise=exclude_form_noise,
            )

            output = {
                "question_text": question_text,
                "intent_type": intent.value,
                "document_filter": str(document_id) if document_id else None,
                "exclude_form_noise": exclude_form_noise,
                "strategy_code": evidence.strategy_code,
                "selected_document_ids": [str(x) for x in evidence.selected_document_ids],
                "selected_fact_ids": [str(x) for x in evidence.selected_fact_ids],
                "selected_table_ids": [str(x) for x in evidence.selected_table_ids],
                "selected_row_ids": [str(x) for x in evidence.selected_row_ids],
                "selected_block_ids": [str(x) for x in evidence.selected_block_ids],
                "metrics_json": evidence.metrics_json,
                "debug_payload_json": evidence.debug_payload_json,
                "candidates": [
                    _render_candidate(candidate)
                    for candidate in filtered_candidates[:top_k]
                ],
            }

            print(
                json.dumps(
                    output,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )

        return 0
    finally:
        await manager.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic retrieval smoke test for one question."
    )
    parser.add_argument(
        "--question",
        default=None,
        help="User question text. Optional for preset intents.",
    )
    parser.add_argument(
        "--intent",
        required=True,
        help=(
            "Question intent enum value or preset name: "
            + ", ".join(QUESTION_PRESETS.keys())
        ),
    )
    parser.add_argument(
        "--document-id",
        default=None,
        help="Optional document_id filter to keep only one document in output.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many candidates to print.",
    )
    parser.add_argument(
        "--include-form-noise",
        action="store_true",
        help="Do not filter form-like table noise from the printed output.",
    )

    args = parser.parse_args()

    intent = _parse_intent(args.intent)
    question_text = _resolve_question(args.question, args.intent)

    document_id: UUID | None = None
    if args.document_id:
        document_id = UUID(args.document_id)

    return asyncio.run(
        run(
            question_text=question_text,
            intent=intent,
            document_id=document_id,
            top_k=args.top_k,
            exclude_form_noise=not args.include_form_noise,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())