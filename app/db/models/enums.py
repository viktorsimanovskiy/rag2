# ============================================================
# File: app/db/models/enums.py
# Purpose:
#   Centralized application enums for ORM models.
#
# Notes:
#   - Python enum values are aligned with PostgreSQL enum labels
#   - string enums are used for stable serialization
# ============================================================

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """
    Base enum that behaves like both str and Enum.
    """

    def __str__(self) -> str:
        return str(self.value)


class ChannelTypeEnum(StrEnum):
    TELEGRAM = "telegram"
    MAX = "max"
    WEB = "web"
    TEST_CONSOLE = "test_console"
    UNKNOWN = "unknown"


class QuestionIntentEnum(StrEnum):
    ELIGIBILITY_QUESTION = "eligibility_question"
    DOCUMENTS_QUESTION = "documents_question"
    REJECTION_QUESTION = "rejection_question"
    FORM_QUESTION = "form_question"
    DEADLINE_QUESTION = "deadline_question"
    PAYMENT_TIMING_QUESTION = "payment_timing_question"
    AMOUNT_QUESTION = "amount_question"
    PROCEDURE_QUESTION = "procedure_question"
    APPEAL_QUESTION = "appeal_question"
    MIXED_QUESTION = "mixed_question"
    AMBIGUOUS_QUESTION = "ambiguous_question"
    NO_MEASURE_DETECTED = "no_measure_detected"
    OTHER = "other"


class AnswerModeEnum(StrEnum):
    DIRECT_STRUCTURED = "direct_structured"
    GROUNDED_NARRATIVE = "grounded_narrative"
    SAFE_NO_ANSWER = "safe_no_answer"
    REUSED_ANSWER = "reused_answer"


class ValidationStatusEnum(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    NOT_RUN = "not_run"


class EvidenceItemTypeEnum(StrEnum):
    DOCUMENT = "document"
    BLOCK = "block"
    TABLE = "table"
    TABLE_ROW = "table_row"
    LEGAL_FACT = "legal_fact"


class FeedbackReasonCodeEnum(StrEnum):
    CORRECT_AND_HELPFUL = "correct_and_helpful"
    PARTIALLY_HELPFUL = "partially_helpful"
    NOT_RELEVANT = "not_relevant"
    INCORRECT_FACTS = "incorrect_facts"
    OUTDATED_INFORMATION = "outdated_information"
    UNCLEAR_ANSWER = "unclear_answer"
    TOO_LONG = "too_long"
    TOO_SHORT = "too_short"
    MISSING_DETAILS = "missing_details"
    WRONG_DOCUMENTS_USED = "wrong_documents_used"
    OTHER = "other"


class ReuseStatusEnum(StrEnum):
    ELIGIBLE = "eligible"
    TEMPORARILY_BLOCKED = "temporarily_blocked"
    BLOCKED = "blocked"
    NEEDS_REVALIDATION = "needs_revalidation"