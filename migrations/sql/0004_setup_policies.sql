-- Migration: Setup RLS Policies
-- Description: Enables RLS and defines access policies for profiles and revoked_tokens.

-- 1. Enable RLS on profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- 2. Profiles Policies
DROP POLICY IF EXISTS "Users can view their own profile" ON public.profiles;
CREATE POLICY "Users can view their own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users can update their own profile" ON public.profiles;
CREATE POLICY "Users can update their own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id);

DROP POLICY IF EXISTS "Admins can view all profiles" ON public.profiles;
CREATE POLICY "Admins can view all profiles"
    ON public.profiles FOR SELECT
    USING (public.is_admin(auth.uid()));

DROP POLICY IF EXISTS "Admins can update all profiles" ON public.profiles;
CREATE POLICY "Admins can update all profiles"
    ON public.profiles FOR UPDATE
    USING (public.is_admin(auth.uid()));

-- 3. Enable RLS on revoked_tokens
ALTER TABLE public.revoked_tokens ENABLE ROW LEVEL SECURITY;

-- 4. Revoked Tokens Policies
DROP POLICY IF EXISTS "Users can view their own revoked tokens" ON public.revoked_tokens;
CREATE POLICY "Users can view their own revoked tokens"
    ON public.revoked_tokens FOR SELECT
    USING (auth.uid() = user_id);
