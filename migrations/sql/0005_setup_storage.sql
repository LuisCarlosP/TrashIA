-- Migration: Setup Storage for Avatars
-- Description: Creates the avatars bucket and defines RLS policies for storage objects.

INSERT INTO storage.buckets (id, name, public)
VALUES ('avatars', 'avatars', true)
ON CONFLICT (id) DO NOTHING;

DROP POLICY IF EXISTS "Avatar pictures are publicly accessible" ON storage.objects;
CREATE POLICY "Avatar pictures are publicly accessible"
    ON storage.objects FOR SELECT
    USING (bucket_id = 'avatars');

-- Policy: Allow authenticated users to upload their own avatar
DROP POLICY IF EXISTS "Users can upload their own avatar" ON storage.objects;
CREATE POLICY "Users can upload their own avatar"
    ON storage.objects FOR INSERT
    TO authenticated
    WITH CHECK (
        bucket_id = 'avatars' AND 
        (storage.foldername(name))[1] = 'profile-pictures' AND
        (auth.uid())::text = split_part((storage.foldername(name))[2], '_', 1)
    );

-- Policy: Allow users to update their own avatar
DROP POLICY IF EXISTS "Users can update their own avatar" ON storage.objects;
CREATE POLICY "Users can update their own avatar"
    ON storage.objects FOR UPDATE
    TO authenticated
    USING (
        bucket_id = 'avatars' AND 
        (auth.uid())::text = split_part((storage.foldername(name))[2], '_', 1)
    );

-- Policy: Allow users to delete their own avatar
DROP POLICY IF EXISTS "Users can delete their own avatar" ON storage.objects;
CREATE POLICY "Users can delete their own avatar"
    ON storage.objects FOR DELETE
    TO authenticated
    USING (
        bucket_id = 'avatars' AND 
        (auth.uid())::text = split_part((storage.foldername(name))[2], '_', 1)
    );
