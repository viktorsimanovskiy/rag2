-- =========================================================
-- File: db/migrations/2026_03_08_001_feedback_and_reuse.sql
-- Purpose:
--   Добавляет в систему:
--   - каналы и сессии
--   - журнал вопросов
--   - журнал ответов
--   - evidence по ответам
--   - обратную связь
--   - кандидатов на safe reuse
--   - агрегаты качества
-- =========================================================

BEGIN;

-- ---------------------------------------------------------
-- 1. Extensions
-- ---------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------
-- 2. Reference enums
-- ---------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'channel_type_enum') THEN
        CREATE TYPE channel_type_enum AS ENUM (
            'telegram',
            'max',
            'web',
            'test_console',
            'unknown'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'question_intent_enum') THEN
        CREATE TYPE question_intent_enum AS ENUM (
            'eligibility_question',
            'documents_question',
            'rejection_question',
            'form_question',
            'deadline_question',
            'payment_timing_question',
            'amount_question',
            'procedure_question',
            'appeal_question',
            'mixed_question',
            'ambiguous_question',
            'no_measure_detected',
            'other'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'answer_mode_enum') THEN
        CREATE TYPE answer_mode_enum AS ENUM (
            'direct_structured',
            'grounded_narrative',
            'safe_no_answer',
            'reused_answer'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'validation_status_enum') THEN
        CREATE TYPE validation_status_enum AS ENUM (
            'passed',
            'failed',
            'partial',
            'not_run'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'evidence_item_type_enum') THEN
        CREATE TYPE evidence_item_type_enum AS ENUM (
            'document',
            'block',
            'table',
            'table_row',
            'legal_fact'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'feedback_reason_code_enum') THEN
        CREATE TYPE feedback_reason_code_enum AS ENUM (
            'correct_and_helpful',
            'partially_helpful',
            'not_relevant',
            'incorrect_facts',
            'outdated_information',
            'unclear_answer',
            'too_long',
            'too_short',
            'missing_details',
            'wrong_documents_used',
            'other'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'reuse_status_enum') THEN
        CREATE TYPE reuse_status_enum AS ENUM (
            'eligible',
            'temporarily_blocked',
            'blocked',
            'needs_revalidation'
        );
    END IF;
END $$;

-- ---------------------------------------------------------
-- 3. Channels
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS channels (
    channel_id              BIGSERIAL PRIMARY KEY,
    channel_code            channel_type_enum NOT NULL UNIQUE,
    channel_name            TEXT NOT NULL,
    is_active               BOOLEAN NOT NULL DEFAULT TRUE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO channels (channel_code, channel_name)
VALUES
    ('telegram', 'Telegram'),
    ('max', 'MAX'),
    ('web', 'Web'),
    ('test_console', 'Test Console'),
    ('unknown', 'Unknown')
ON CONFLICT (channel_code) DO NOTHING;

-- ---------------------------------------------------------
-- 4. Conversation sessions
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_id                      BIGINT NOT NULL REFERENCES channels(channel_id),
    external_session_id             TEXT NOT NULL,
    external_user_id                TEXT,
    external_chat_id                TEXT,
    user_platform_name              TEXT,
    session_started_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_last_activity_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata_json                   JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT uq_conversation_sessions_unique
        UNIQUE (channel_id, external_session_id)
);

CREATE INDEX IF NOT EXISTS idx_conversation_sessions_channel
    ON conversation_sessions(channel_id);

CREATE INDEX IF NOT EXISTS idx_conversation_sessions_last_activity
    ON conversation_sessions(session_last_activity_at DESC);

-- ---------------------------------------------------------
-- 5. Question events
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS question_events (
    question_event_id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id                      UUID NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,

    question_text_raw               TEXT NOT NULL,
    question_text_normalized        TEXT,
    question_language_code          TEXT NOT NULL DEFAULT 'ru',

    intent_type                     question_intent_enum NOT NULL DEFAULT 'other',
    measure_code                    TEXT,
    subject_category_code           TEXT,
    query_constraints_json          JSONB NOT NULL DEFAULT '{}'::jsonb,

    routing_payload_json            JSONB NOT NULL DEFAULT '{}'::jsonb,
    classifier_version              TEXT,
    embedding_model_name            TEXT,

    question_embedding              vector(1536),

    created_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_question_events_session
    ON question_events(session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_question_events_intent
    ON question_events(intent_type);

CREATE INDEX IF NOT EXISTS idx_question_events_measure_code
    ON question_events(measure_code);

CREATE INDEX IF NOT EXISTS idx_question_events_created_at
    ON question_events(created_at DESC);

-- Vector index: подстрой размерность под фактическую модель, если она будет другой.
CREATE INDEX IF NOT EXISTS idx_question_events_embedding_ivfflat
    ON question_events
    USING ivfflat (question_embedding vector_cosine_ops)
    WITH (lists = 100);

-- ---------------------------------------------------------
-- 6. Answer events
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS answer_events (
    answer_event_id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question_event_id                UUID NOT NULL REFERENCES question_events(question_event_id) ON DELETE CASCADE,

    answer_mode                      answer_mode_enum NOT NULL,
    answer_text                      TEXT NOT NULL,
    answer_text_short                TEXT,
    answer_language_code             TEXT NOT NULL DEFAULT 'ru',

    confidence_score                 NUMERIC(5,4),
    trust_score_at_generation        NUMERIC(5,4),

    validation_status                validation_status_enum NOT NULL DEFAULT 'not_run',
    deterministic_validation_passed  BOOLEAN NOT NULL DEFAULT FALSE,
    semantic_validation_passed       BOOLEAN NOT NULL DEFAULT FALSE,

    reuse_allowed                    BOOLEAN NOT NULL DEFAULT FALSE,
    reused_from_answer_event_id      UUID REFERENCES answer_events(answer_event_id),
    reuse_policy_version             TEXT,
    reuse_decision_payload_json      JSONB NOT NULL DEFAULT '{}'::jsonb,

    citations_json                   JSONB NOT NULL DEFAULT '[]'::jsonb,
    answer_payload_json              JSONB NOT NULL DEFAULT '{}'::jsonb,

    document_set_hash                TEXT,
    evidence_hash                    TEXT,
    answer_payload_hash              TEXT,
    answer_text_hash                 TEXT,

    generation_model_name            TEXT,
    generation_prompt_version        TEXT,
    pipeline_version                 TEXT,

    created_at                       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_answer_events_confidence_range
        CHECK (
            confidence_score IS NULL
            OR (confidence_score >= 0 AND confidence_score <= 1)
        ),

    CONSTRAINT chk_answer_events_trust_range
        CHECK (
            trust_score_at_generation IS NULL
            OR (trust_score_at_generation >= 0 AND trust_score_at_generation <= 1)
        )
);

CREATE INDEX IF NOT EXISTS idx_answer_events_question
    ON answer_events(question_event_id);

CREATE INDEX IF NOT EXISTS idx_answer_events_created_at
    ON answer_events(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_answer_events_reuse_allowed
    ON answer_events(reuse_allowed);

CREATE INDEX IF NOT EXISTS idx_answer_events_reused_from
    ON answer_events(reused_from_answer_event_id);

CREATE INDEX IF NOT EXISTS idx_answer_events_validation_status
    ON answer_events(validation_status);

CREATE INDEX IF NOT EXISTS idx_answer_events_document_set_hash
    ON answer_events(document_set_hash);

CREATE INDEX IF NOT EXISTS idx_answer_events_evidence_hash
    ON answer_events(evidence_hash);

-- ---------------------------------------------------------
-- 7. Answer evidence items
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS answer_evidence_items (
    answer_evidence_item_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    answer_event_id                  UUID NOT NULL REFERENCES answer_events(answer_event_id) ON DELETE CASCADE,

    evidence_order                   INTEGER NOT NULL,
    evidence_item_type               evidence_item_type_enum NOT NULL,

    document_id                      UUID,
    block_id                         UUID,
    table_id                         UUID,
    table_row_id                     UUID,
    legal_fact_id                    UUID,

    citation_json                    JSONB NOT NULL DEFAULT '{}'::jsonb,

    document_file_hash               TEXT,
    document_content_hash            TEXT,

    role_code                        TEXT,
    created_at                       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_answer_evidence_items_order
        UNIQUE (answer_event_id, evidence_order),

    CONSTRAINT chk_answer_evidence_exactly_one_pointer
        CHECK (
            ((document_id IS NOT NULL)::int +
             (block_id IS NOT NULL)::int +
             (table_id IS NOT NULL)::int +
             (table_row_id IS NOT NULL)::int +
             (legal_fact_id IS NOT NULL)::int) = 1
        )
);

CREATE INDEX IF NOT EXISTS idx_answer_evidence_items_answer
    ON answer_evidence_items(answer_event_id);

CREATE INDEX IF NOT EXISTS idx_answer_evidence_items_document
    ON answer_evidence_items(document_id);

CREATE INDEX IF NOT EXISTS idx_answer_evidence_items_table
    ON answer_evidence_items(table_id);

CREATE INDEX IF NOT EXISTS idx_answer_evidence_items_fact
    ON answer_evidence_items(legal_fact_id);

CREATE INDEX IF NOT EXISTS idx_answer_evidence_items_doc_content_hash
    ON answer_evidence_items(document_content_hash);

-- ---------------------------------------------------------
-- 8. Answer feedback
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS answer_feedback (
    feedback_id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    answer_event_id                  UUID NOT NULL REFERENCES answer_events(answer_event_id) ON DELETE CASCADE,
    session_id                       UUID NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,

    score                            SMALLINT NOT NULL,
    reason_code                      feedback_reason_code_enum,
    comment_text                     TEXT,

    feedback_channel_id              BIGINT REFERENCES channels(channel_id),
    submitted_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    is_sampled_request               BOOLEAN NOT NULL DEFAULT FALSE,
    sampling_policy_version          TEXT,

    is_resolved_for_analytics        BOOLEAN NOT NULL DEFAULT TRUE,
    metadata_json                    JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT chk_answer_feedback_score_range
        CHECK (score >= 1 AND score <= 5)
);

CREATE INDEX IF NOT EXISTS idx_answer_feedback_answer
    ON answer_feedback(answer_event_id);

CREATE INDEX IF NOT EXISTS idx_answer_feedback_session
    ON answer_feedback(session_id);

CREATE INDEX IF NOT EXISTS idx_answer_feedback_submitted_at
    ON answer_feedback(submitted_at DESC);

CREATE INDEX IF NOT EXISTS idx_answer_feedback_score
    ON answer_feedback(score);

-- Один пользователь в рамках одной сессии не должен многократно голосовать за один и тот же ответ.
CREATE UNIQUE INDEX IF NOT EXISTS uq_answer_feedback_one_vote_per_session_answer
    ON answer_feedback(session_id, answer_event_id);

-- ---------------------------------------------------------
-- 9. Reuse candidates
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS answer_reuse_candidates (
    reuse_candidate_id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_answer_event_id           UUID NOT NULL UNIQUE REFERENCES answer_events(answer_event_id) ON DELETE CASCADE,

    reuse_status                     reuse_status_enum NOT NULL DEFAULT 'needs_revalidation',
    reuse_allowed_effective          BOOLEAN NOT NULL DEFAULT FALSE,

    avg_feedback_score               NUMERIC(5,4),
    feedback_count                   INTEGER NOT NULL DEFAULT 0,
    negative_feedback_count          INTEGER NOT NULL DEFAULT 0,

    reuse_score                      NUMERIC(5,4),
    last_revalidated_at              TIMESTAMPTZ,
    revalidation_payload_json        JSONB NOT NULL DEFAULT '{}'::jsonb,

    block_reason_code                TEXT,
    created_at                       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_answer_reuse_candidates_avg_feedback_range
        CHECK (
            avg_feedback_score IS NULL
            OR (avg_feedback_score >= 1 AND avg_feedback_score <= 5)
        ),

    CONSTRAINT chk_answer_reuse_candidates_reuse_score_range
        CHECK (
            reuse_score IS NULL
            OR (reuse_score >= 0 AND reuse_score <= 1)
        )
);

CREATE INDEX IF NOT EXISTS idx_answer_reuse_candidates_status
    ON answer_reuse_candidates(reuse_status);

CREATE INDEX IF NOT EXISTS idx_answer_reuse_candidates_allowed
    ON answer_reuse_candidates(reuse_allowed_effective);

CREATE INDEX IF NOT EXISTS idx_answer_reuse_candidates_reuse_score
    ON answer_reuse_candidates(reuse_score DESC NULLS LAST);

-- ---------------------------------------------------------
-- 10. Daily quality aggregates
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS quality_aggregates_daily (
    quality_aggregate_id             BIGSERIAL PRIMARY KEY,
    aggregate_date                   DATE NOT NULL,

    channel_code                     channel_type_enum NOT NULL DEFAULT 'unknown',
    intent_type                      question_intent_enum NOT NULL DEFAULT 'other',
    measure_code                     TEXT,

    total_answers                    INTEGER NOT NULL DEFAULT 0,
    total_feedback                   INTEGER NOT NULL DEFAULT 0,
    avg_feedback_score               NUMERIC(5,4),

    reused_answers_count             INTEGER NOT NULL DEFAULT 0,
    low_rated_answers_count          INTEGER NOT NULL DEFAULT 0,
    failed_validation_count          INTEGER NOT NULL DEFAULT 0,

    created_at                       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_quality_aggregates_daily
        UNIQUE (aggregate_date, channel_code, intent_type, measure_code)
);

CREATE INDEX IF NOT EXISTS idx_quality_aggregates_daily_date
    ON quality_aggregates_daily(aggregate_date DESC);

CREATE INDEX IF NOT EXISTS idx_quality_aggregates_daily_measure
    ON quality_aggregates_daily(measure_code);

-- ---------------------------------------------------------
-- 11. Helpful views
-- ---------------------------------------------------------
CREATE OR REPLACE VIEW vw_answer_feedback_summary AS
SELECT
    ae.answer_event_id,
    COUNT(af.feedback_id) AS feedback_count,
    AVG(af.score::numeric) AS avg_score,
    SUM(CASE WHEN af.score <= 2 THEN 1 ELSE 0 END) AS low_score_count,
    MIN(af.submitted_at) AS first_feedback_at,
    MAX(af.submitted_at) AS last_feedback_at
FROM answer_events ae
LEFT JOIN answer_feedback af
    ON af.answer_event_id = ae.answer_event_id
GROUP BY ae.answer_event_id;

CREATE OR REPLACE VIEW vw_reuse_ready_answers AS
SELECT
    ae.answer_event_id,
    ae.question_event_id,
    ae.answer_mode,
    ae.answer_text,
    ae.document_set_hash,
    ae.evidence_hash,
    ae.validation_status,
    rc.reuse_status,
    rc.reuse_allowed_effective,
    rc.avg_feedback_score,
    rc.feedback_count,
    rc.reuse_score,
    rc.last_revalidated_at
FROM answer_events ae
JOIN answer_reuse_candidates rc
    ON rc.source_answer_event_id = ae.answer_event_id
WHERE
    ae.reuse_allowed = TRUE
    AND ae.validation_status = 'passed'
    AND rc.reuse_allowed_effective = TRUE
    AND rc.reuse_status = 'eligible';

-- ---------------------------------------------------------
-- 12. Updated_at trigger
-- ---------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at_now()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_answer_reuse_candidates_updated_at ON answer_reuse_candidates;

CREATE TRIGGER trg_answer_reuse_candidates_updated_at
BEFORE UPDATE ON answer_reuse_candidates
FOR EACH ROW
EXECUTE FUNCTION set_updated_at_now();

COMMIT;