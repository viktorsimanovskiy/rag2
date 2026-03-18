# ============================================================
# File: app/config/constants.py
# Purpose:
#   Centralized architectural constants for AI models,
#   retrieval, generation, validation and document processing.
#
# Important:
#   - This file stores project-level constants.
#   - Secrets, URLs and tokens belong in settings.py / env vars.
# ============================================================

from __future__ import annotations


# ============================================================
# System metadata
# ============================================================

SYSTEM_NAME: str = "normative-rag-system"
SYSTEM_VERSION: str = "0.1.0"


# ============================================================
# LLM configuration
# ============================================================

# Main answer generation model
LLM_MODEL_NAME: str = "gpt-4.1-mini"

# Low temperature is preferable for grounded legal / normative answers
LLM_TEMPERATURE: float = 0.1

# Conservative maximum answer size
LLM_MAX_OUTPUT_TOKENS: int = 1200

# Prompt / pipeline versioning
GENERATION_PROMPT_VERSION: str = "rag_answer_prompt_v1"
GENERATION_PIPELINE_VERSION: str = "generation_pipeline_v1"


# ============================================================
# Embedding configuration
# ============================================================

# Embedding model chosen for the current project stage
EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"

# text-embedding-3-small commonly uses 1536 dimensions when not shortened
EMBEDDING_DIMENSION: int = 1536

# Keep source text moderate before embedding.
# This is an application-level truncation safeguard, not the model token limit.
EMBEDDING_MAX_TEXT_LENGTH: int = 2000

# Versioning
EMBEDDING_PIPELINE_VERSION: str = "embedding_pipeline_v1"


# ============================================================
# Vector search configuration
# ============================================================

# How many vector candidates to request from the index
VECTOR_SEARCH_TOP_K: int = 20

# Similarity floor for candidate acceptance
MIN_VECTOR_SIMILARITY: float = 0.30

# Versioning
RETRIEVAL_PIPELINE_VERSION: str = "retrieval_pipeline_v1"


# ============================================================
# Retrieval / evidence configuration
# ============================================================

# Max number of evidence blocks sent to generation
MAX_EVIDENCE_BLOCKS: int = 8

# Max number of candidates preserved before optional rerank / merge
MAX_RETRIEVAL_CANDIDATES: int = 20

# If reranker is introduced later, this is a safe early default
RERANK_TOP_K: int = 10


# ============================================================
# Validation / safety configuration
# ============================================================

# Minimum retrieval confidence for attempting a grounded answer
MIN_RETRIEVAL_CONFIDENCE: float = 0.35

# Minimum final confidence for allowing answer reuse
MIN_REUSE_CONFIDENCE: float = 0.75

# Conservative SAFE_NO_ANSWER mode is preferred for normative QA
ENABLE_SAFE_NO_ANSWER: bool = True


# ============================================================
# Document processing configuration
# ============================================================

# Conservative block sizing for normative text
DOCUMENT_BLOCK_MAX_CHARS: int = 1200
DOCUMENT_BLOCK_OVERLAP: int = 200


# ============================================================
# Feedback / quality configuration
# ============================================================

# Early-production thresholds for reuse eligibility
MIN_FEEDBACK_COUNT_FOR_REUSE: int = 3
MIN_AVG_FEEDBACK_SCORE_FOR_REUSE: float = 4.2
MAX_NEGATIVE_FEEDBACK_RATIO_FOR_REUSE: float = 0.20
