-- Migration: Add User Roles
-- Description: Defines user roles and adds the role column to the profiles table.

-- 1. Create user_role enum type
DO $$ BEGIN
    CREATE TYPE user_role AS ENUM ('user', 'moderator', 'admin', 'super_admin');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 2. Add role column to profiles
ALTER TABLE public.profiles
ADD COLUMN IF NOT EXISTS role user_role DEFAULT 'user';

-- 3. Create a helper function to check if a user is an admin
CREATE OR REPLACE FUNCTION public.is_admin(user_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM public.profiles
        WHERE id = user_id AND role IN ('admin', 'super_admin')
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
