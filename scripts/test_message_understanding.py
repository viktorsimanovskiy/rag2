# ============================================================
# File: scripts/test_message_understanding.py
# Purpose:
#   Manual smoke-test for optional LLM message understanding.
#
# Version:
#   second_step_05_message_understanding_smoke_and_debug_v1
#
# Usage:
#   APP_MESSAGE_UNDERSTANDING_ENABLED=true \
#   APP_MESSAGE_UNDERSTANDING_MODE=shadow \
#   python scripts/test_message_understanding.py \
#     --question "Документы для получения выплаты для почетных доноров" \
#     --json-report /home/logs/second_step_04/understanding_donors.json
# ============================================================

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import load_settings
from app.integrations.openai.client_factory import OpenAIClientFactory
from app.services.answers.intent_classifier import RuleBasedIntentClassifier
from app.services.answers.message_understanding import (
    LLMMessageUnderstandingService,
    MessageUnderstandingConfig,
)


async def main() -> None:
    args = _parse_args()
    settings = load_settings()

    client = OpenAIClientFactory(settings.openai).create_async_client()
    service = LLMMessageUnderstandingService(
        client,
        config=MessageUnderstandingConfig(
            enabled=settings.message_understanding.enabled,
            mode=settings.message_understanding.mode,
            model_name=settings.message_understanding.model_name,
            temperature=settings.message_understanding.temperature,
            max_output_tokens=settings.message_understanding.max_output_tokens,
            min_confidence_to_apply=settings.message_understanding.min_confidence_to_apply,
            request_timeout_seconds=settings.message_understanding.request_timeout_seconds,
        ),
    )

    classifier = RuleBasedIntentClassifier()
    deterministic = await classifier.classify(args.question)
    understanding = await service.understand(
        args.question,
        deterministic_classification=deterministic,
        channel_code=args.channel,
    )

    report = {
        "ok": understanding.provider_status == "ok",
        "question": args.question,
        "channel": args.channel,
        "deterministic_classification": _jsonable(deterministic),
        "message_understanding": understanding.to_payload(),
    }

    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text)

    if args.json_report:
        path = Path(args.json_report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--channel", default="test_console")
    parser.add_argument("--json-report", default="")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


if __name__ == "__main__":
    asyncio.run(main())
