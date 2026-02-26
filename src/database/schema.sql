-- VoiceBridge PostgreSQL — 2 tables: users (customizations), conversations (user ↔ agent).
-- Encrypt sensitive data in app (M13) before storing.

-- 1. Users + per-person customizations (preferences, consent in JSONB)
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    preferences JSONB NOT NULL DEFAULT '{}',   -- theme, verbosity, asr_model_id, etc.
    consent     JSONB NOT NULL DEFAULT '{}',   -- consent flags and timestamps
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2. Conversations (user ↔ agent messages)
CREATE TABLE conversations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    messages    JSONB NOT NULL DEFAULT '[]',   -- [{ "role": "user"|"agent", "content": "...", "at": "..." }, ...]
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_updated_at ON conversations(updated_at);
