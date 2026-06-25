# ============================================================
# File: app/db/models/__init__.py
# Purpose:
#   Canonical ORM model registry for the project.
# ============================================================

from app.db.models.documents import (
    IngestionJob,
    DocumentRegistry,
    DocumentBlock,
    DocumentTable,
    DocumentTableRow,
    LegalFact,
)

from app.db.models.services import ServiceRegistry

from app.db.models.feedback import (
    Channel,
    ConversationSession,
    QuestionEvent,
    AnswerEvent,
    AnswerEvidenceItem,
    AnswerFeedback,
    AnswerReuseCandidate,
    QualityAggregateDaily,
)

__all__ = [
    "IngestionJob",
    "DocumentRegistry",
    "DocumentBlock",
    "DocumentTable",
    "DocumentTableRow",
    "LegalFact",
    "ServiceRegistry",
    "Channel",
    "ConversationSession",
    "QuestionEvent",
    "AnswerEvent",
    "AnswerEvidenceItem",
    "AnswerFeedback",
    "AnswerReuseCandidate",
    "QualityAggregateDaily",
]