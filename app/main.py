# ============================================================
# File: app/main.py
# Purpose:
#   Application entrypoint for running the RAG system.
#
# Responsibilities:
#   - load validated application settings
#   - build runtime dependencies
#   - assemble OpenAI-based query embedding service
#   - start application runtime
#   - run current Telegram loop path
#
# Important:
#   - this is the CURRENT runtime entrypoint from the archive state
#   - Telegram path is preserved as the current operational path
#   - future target architecture still moves external channel orchestration to n8n
# ============================================================

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from app.channels.telegram_bot import (
    TelegramBotHandler,
    TelegramCallbackUpdate,
    TelegramMessageUpdate,
    TelegramOutgoingAck,
    TelegramOutgoingMessage,
)
from app.config.constants import EMBEDDING_MODEL_NAME
from app.config.settings import AppSettings, load_settings
from app.integrations.openai.client_factory import OpenAIClientFactory
from app.runtime.app_runtime import AppRuntime, AppRuntimeConfig
from app.services.embedding.openai_embedding_provider import OpenAIEmbeddingProvider
from app.services.retrieval.query_embedding_service import QueryEmbeddingService

logger = logging.getLogger(__name__)


# ============================================================
# Example transport interface
# ============================================================

class TelegramTransport:
    """
    Abstract Telegram transport layer.

    Current archive state keeps Telegram as the runtime path here.
    Later, when the project moves to the target n8n-centered architecture,
    this transport/runtime loop will likely be replaced by API entrypoints.
    """

    async def receive_update(self) -> Dict[str, Any]:
        raise NotImplementedError

    async def send_message(self, message: TelegramOutgoingMessage) -> None:
        raise NotImplementedError

    async def answer_callback(self, ack: TelegramOutgoingAck) -> None:
        raise NotImplementedError


# ============================================================
# Update parsing
# ============================================================

def parse_message_update(update: Dict[str, Any]) -> TelegramMessageUpdate:
    message = update["message"]
    chat = message["chat"]
    user = message["from"]

    return TelegramMessageUpdate(
        update_id=str(update["update_id"]),
        message_id=message["message_id"],
        text=message.get("text", ""),
        chat={
            "telegram_chat_id": str(chat["id"]),
            "chat_type": chat.get("type"),
            "title": chat.get("title"),
        },
        user={
            "telegram_user_id": str(user["id"]),
            "username": user.get("username"),
            "first_name": user.get("first_name"),
            "last_name": user.get("last_name"),
            "language_code": user.get("language_code"),
        },
    )


def parse_callback_update(update: Dict[str, Any]) -> TelegramCallbackUpdate:
    callback = update["callback_query"]
    user = callback["from"]
    message = callback["message"]
    chat = message["chat"]

    return TelegramCallbackUpdate(
        update_id=str(update["update_id"]),
        callback_query_id=callback["id"],
        callback_data=callback["data"],
        message_id=message.get("message_id"),
        chat={
            "telegram_chat_id": str(chat["id"]),
            "chat_type": chat.get("type"),
        },
        user={
            "telegram_user_id": str(user["id"]),
            "username": user.get("username"),
            "first_name": user.get("first_name"),
            "last_name": user.get("last_name"),
            "language_code": user.get("language_code"),
        },
    )


# ============================================================
# Bootstrap helpers
# ============================================================

def build_query_embedding_service(settings: AppSettings) -> QueryEmbeddingService:
    """
    Build query embedding service through the canonical OpenAI client factory path.

    Why here:
    - AppRuntime expects question_embedding_service to be injected
    - ServiceFactory forwards it into AnswerOrchestrator
    - this keeps OpenAI wiring in the composition root, not inside services
    """
    openai_client = OpenAIClientFactory(settings.openai).create_async_client()
    embedding_provider = OpenAIEmbeddingProvider(openai_client)

    return QueryEmbeddingService(
        provider=embedding_provider,
        model_name=EMBEDDING_MODEL_NAME,
    )


def build_runtime(settings: AppSettings) -> AppRuntime:
    """
    Build application runtime with all currently required dependencies.
    """
    query_embedding_service = build_query_embedding_service(settings)

    runtime_config = AppRuntimeConfig(
        database=settings.database,
        question_embedding_service=query_embedding_service,
    )

    return AppRuntime(runtime_config)


# ============================================================
# Runtime loop
# ============================================================

async def run_telegram_loop(runtime: AppRuntime, transport: TelegramTransport) -> None:
    logger.info("Telegram update loop started")

    while True:
        update = await transport.receive_update()

        try:
            async with runtime.session_scope() as session:
                handler: TelegramBotHandler = runtime.build_telegram_bot_handler(session)

                if "message" in update:
                    msg_update = parse_message_update(update)
                    await handler.handle_message_update(
                        msg_update,
                        send_message=transport.send_message,
                    )

                elif "callback_query" in update:
                    cb_update = parse_callback_update(update)
                    await handler.handle_callback_update(
                        cb_update,
                        ack_callback=transport.answer_callback,
                    )

                else:
                    logger.warning(
                        "Unsupported Telegram update payload received",
                        extra={
                            "update_keys": sorted(update.keys()),
                        },
                    )

        except Exception:
            logger.exception("Failed to process Telegram update")


# ============================================================
# Entry point
# ============================================================

async def async_main() -> None:
    settings = load_settings()
    runtime = build_runtime(settings)

    await runtime.startup()

    try:
        transport = TelegramTransport()  # must be implemented by the actual transport layer
        await run_telegram_loop(runtime, transport)
    finally:
        await runtime.shutdown()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()