-- Migration: Create Revoked Tokens Table
-- Description: Creates a table to store revoked JWT IDs for blacklisting.

CREATE TABLE IF NOT EXISTS public.revoked_tokens (
    id SERIAL PRIMARY KEY,
    jti TEXT UNIQUE NOT NULL,
    user_id UUID REFERENCES auth.users ON DELETE CASCADE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for faster lookup and cleanup
CREATE INDEX IF NOT EXISTS idx_revoked_tokens_jti ON public.revoked_tokens(jti);
CREATE INDEX IF NOT EXISTS idx_revoked_tokens_expires_at ON public.revoked_tokens(expires_at);
