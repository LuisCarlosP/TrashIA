"""
Authentication service using Supabase Auth.
"""
import logging
from typing import Optional
from uuid import UUID

from gotrue.errors import AuthApiError

from infrastructure.supabase_client import SupabaseClient
from models.user_models import (
    UserRegisterRequest,
    UserLoginRequest,
    RefreshTokenRequest,
    ResendVerificationRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    UserResponse,
    UserProfile,
    TokenResponse,
    AuthResponse,
    MessageResponse,
    ProfilePictureResponse
)
from exceptions.auth_exceptions import (
    AuthenticationError,
    UserAlreadyExistsError,
    UserNotFoundError,
    InvalidTokenError,
    InactiveUserError
)
from config.settings import FRONTEND_URL

logger = logging.getLogger(__name__)

# Remove trailing slash from FRONTEND_URL to avoid double slashes in redirects
_FRONTEND_URL = FRONTEND_URL.rstrip('/')

class SupabaseAuthService:
    """
    Authentication service using Supabase Auth.
    Handles registration, login, logout, and email verification.
    """
    
    def __init__(self):
        self.client = SupabaseClient.get_client()
    
    async def register(self, user_data: UserRegisterRequest) -> AuthResponse:
        """
        Register a new user with Supabase Auth.
        Sends verification email automatically.
        
        Args:
            user_data: User registration data
            
        Returns:
            AuthResponse: User data and authentication tokens
        """
        try:
            # Register with Supabase Auth
            # Metadata will be used by trigger to create profile
            response = self.client.auth.sign_up({
                "email": user_data.email,
                "password": user_data.password,
                "options": {
                    "data": {
                        "name": user_data.name,
                        "last_name": user_data.last_name,
                        "telephone": user_data.telephone,
                        "profile_picture": user_data.profile_picture
                    },
                    "email_redirect_to": f"{_FRONTEND_URL}/auth/callback"
                }
            })
            
            if response.user is None:
                raise AuthenticationError("Registration failed")
            
            user = response.user
            session = response.session
            
            logger.info(f"New user registered: {user.email}")
            
            # Get profile data
            profile = await self._get_profile(user.id)
            
            # Build response
            user_response = UserResponse(
                id=UUID(user.id),
                email=user.email,
                email_verified=user.email_confirmed_at is not None,
                profile=profile
            )
            
            # If session exists (auto-confirm disabled), return tokens
            if session:
                tokens = TokenResponse(
                    access_token=session.access_token,
                    refresh_token=session.refresh_token,
                    expires_in=session.expires_in or 3600,
                    expires_at=session.expires_at
                )
            else:
                # No session means email confirmation required
                tokens = TokenResponse(
                    access_token="",
                    refresh_token="",
                    expires_in=0
                )
            
            return AuthResponse(user=user_response, tokens=tokens)
            
        except AuthApiError as e:
            logger.error(f"Supabase auth error during registration: {e}")
            if "already registered" in str(e).lower():
                raise UserAlreadyExistsError(user_data.email)
            raise AuthenticationError(str(e))
        except Exception as e:
            logger.error(f"Registration error: {e}")
            raise
    
    async def login(self, credentials: UserLoginRequest) -> AuthResponse:
        """
        Authenticate user with email and password.
        
        Args:
            credentials: Login credentials
            
        Returns:
            AuthResponse: User data and authentication tokens
        """
        try:
            response = self.client.auth.sign_in_with_password({
                "email": credentials.email,
                "password": credentials.password
            })
            
            if response.user is None or response.session is None:
                raise AuthenticationError("Invalid email or password")
            
            user = response.user
            session = response.session
            
            logger.info(f"User logged in: {user.email}")
            
            # Get profile data
            profile = await self._get_profile(user.id)
            
            user_response = UserResponse(
                id=UUID(user.id),
                email=user.email,
                email_verified=user.email_confirmed_at is not None,
                profile=profile
            )
            
            tokens = TokenResponse(
                access_token=session.access_token,
                refresh_token=session.refresh_token,
                expires_in=session.expires_in or 3600,
                expires_at=session.expires_at
            )
            
            return AuthResponse(user=user_response, tokens=tokens)
            
        except AuthApiError as e:
            logger.warning(f"Login failed: {e}")
            raise AuthenticationError("Invalid email or password")
        except Exception as e:
            logger.error(f"Login error: {e}")
            raise
    
    async def logout(self, access_token: str) -> MessageResponse:
        """
        Logout user by invalidating the session.
        
        Args:
            access_token: Current access token
        """
        try:
            # Set the session to use this token
            self.client.auth.sign_out()
            logger.info("User logged out")
            return MessageResponse(message="Successfully logged out")
        except Exception as e:
            logger.error(f"Logout error: {e}")
            # Still return success - token will expire anyway
            return MessageResponse(message="Logged out")
    
    async def refresh_tokens(self, refresh_token: str) -> TokenResponse:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Current refresh token
            
        Returns:
            TokenResponse: New token pair
        """
        try:
            response = self.client.auth.refresh_session(refresh_token)
            
            if response.session is None:
                raise InvalidTokenError("Invalid refresh token")
            
            session = response.session
            
            return TokenResponse(
                access_token=session.access_token,
                refresh_token=session.refresh_token,
                expires_in=session.expires_in or 3600,
                expires_at=session.expires_at
            )
            
        except AuthApiError as e:
            logger.warning(f"Token refresh failed: {e}")
            raise InvalidTokenError("Invalid or expired refresh token")
        except Exception as e:
            logger.error(f"Refresh error: {e}")
            raise
    
    async def get_current_user(self, access_token: str) -> UserResponse:
        """
        Get the currently authenticated user from access token.
        
        Args:
            access_token: JWT access token
            
        Returns:
            UserResponse: Current user data
        """
        try:
            # Get user from token
            response = self.client.auth.get_user(access_token)
            
            if response.user is None:
                raise InvalidTokenError("Invalid token")
            
            user = response.user
            
            # Get profile data
            profile = await self._get_profile(user.id)
            
            return UserResponse(
                id=UUID(user.id),
                email=user.email,
                email_verified=user.email_confirmed_at is not None,
                profile=profile
            )
            
        except AuthApiError as e:
            logger.warning(f"Get user failed: {e}")
            raise InvalidTokenError("Invalid or expired token")
        except Exception as e:
            logger.error(f"Get user error: {e}")
            raise
    
    async def resend_verification_email(
        self, 
        request: ResendVerificationRequest
    ) -> MessageResponse:
        """
        Resend verification email.
        
        Args:
            request: Contains email address
            
        Returns:
            MessageResponse: Generic success message
        """
        try:
            self.client.auth.resend({
                "type": "signup",
                "email": request.email,
                "options": {
                    "email_redirect_to": f"{_FRONTEND_URL}/auth/callback"
                }
            })
            
            # Always return success to prevent email enumeration
            return MessageResponse(
                message="If the email exists, a verification link was sent"
            )
            
        except Exception as e:
            logger.error(f"Resend verification error: {e}")
            # Still return success to prevent enumeration
            return MessageResponse(
                message="If the email exists, a verification link was sent"
            )
    
    async def forgot_password(
        self, 
        request: ForgotPasswordRequest
    ) -> MessageResponse:
        """
        Send password reset email.
        
        Args:
            request: Contains email address
            
        Returns:
            MessageResponse: Generic success message
        """
        try:
            self.client.auth.reset_password_email(
                request.email,
                options={
                    "redirect_to": f"{_FRONTEND_URL}/auth/reset-password"
                }
            )
            
            return MessageResponse(
                message="If the email exists, a password reset link was sent"
            )
            
        except Exception as e:
            logger.error(f"Forgot password error: {e}")
            return MessageResponse(
                message="If the email exists, a password reset link was sent"
            )
    
    async def reset_password(
        self, 
        request: ResetPasswordRequest
    ) -> MessageResponse:
        """
        Reset password using token from email.
        
        Args:
            request: Contains access_token and new password
            
        Returns:
            MessageResponse: Success message
        """
        try:
            # The access_token from the reset email URL is a JWT session token
            # We need to set the session with it, then update the password
            # Supabase sends: #access_token=...&refresh_token=...&type=recovery
            
            # Set session using the access token from the recovery email
            # This authenticates the user with the recovery token
            self.client.auth.set_session(
                access_token=request.access_token,
                refresh_token=request.refresh_token if hasattr(request, 'refresh_token') and request.refresh_token else request.access_token
            )
            
            # Now update the password for the authenticated user
            self.client.auth.update_user({
                "password": request.new_password
            })
            
            # Sign out after password change for security
            self.client.auth.sign_out()
            
            return MessageResponse(message="Password updated successfully")
            
        except AuthApiError as e:
            logger.warning(f"Reset password failed: {e}")
            raise InvalidTokenError("Invalid or expired reset token")
        except Exception as e:
            logger.error(f"Reset password error: {e}")
            raise
    
    async def update_profile(
        self, 
        user_id: UUID, 
        **kwargs
    ) -> UserProfile:
        """
        Update user profile data.
        
        Args:
            user_id: User's UUID
            **kwargs: Fields to update (name, last_name, telephone, profile_picture)
            
        Returns:
            UserProfile: Updated profile
        """
        try:
            # Filter valid fields
            valid_fields = {'name', 'last_name', 'telephone', 'profile_picture'}
            update_data = {}
            
            for k, v in kwargs.items():
                if k in valid_fields:
                    # Handle empty string as None (for removing profile picture)
                    if v == '':
                        update_data[k] = None
                    else:
                        update_data[k] = v
            
            if not update_data:
                return await self._get_profile(str(user_id))
            
            result = self.client.table('profiles')\
                .update(update_data)\
                .eq('id', str(user_id))\
                .execute()
            
            if result.data:
                return UserProfile(**result.data[0])
            
            raise UserNotFoundError()
            
        except Exception as e:
            logger.error(f"Update profile error: {e}")
            raise
    
    async def _get_profile(self, user_id: str) -> UserProfile:
        """Get user profile from profiles table."""
        try:
            # Use .execute() without .maybe_single() to avoid 406 errors
            # when the profile doesn't exist yet
            result = self.client.table('profiles')\
                .select('*')\
                .eq('id', user_id)\
                .execute()
            
            if result.data and len(result.data) > 0:
                return UserProfile(**result.data[0])
            
            # Profile doesn't exist yet (trigger may not have run)
            # Return empty profile
            return UserProfile(
                id=UUID(user_id),
                name="",
                last_name="",
                telephone=None,
                profile_picture=None,
                created_at=None
            )
            
        except Exception as e:
            logger.warning(f"Get profile error: {e}")
            return UserProfile(
                id=UUID(user_id),
                name="",
                last_name="",
                telephone=None,
                profile_picture=None,
                created_at=None
            )

    async def upload_profile_picture(
        self,
        user_id: UUID,
        file_content: bytes,
        content_type: str
    ) -> str:
        """
        Upload profile picture to Supabase Storage.
        
        Args:
            user_id: User's UUID
            file_content: Image file bytes
            content_type: MIME type of the image
            
        Returns:
            str: Public URL of the uploaded image
        """
        try:
            # Determine file extension from content type
            extensions = {
                'image/jpeg': 'jpg',
                'image/png': 'png',
                'image/gif': 'gif',
                'image/webp': 'webp'
            }
            extension = extensions.get(content_type, 'jpg')
            
            # Create unique filename
            import time
            filename = f"{user_id}_{int(time.time())}.{extension}"
            file_path = f"profile-pictures/{filename}"
            
            # Upload to Supabase Storage
            result = self.client.storage.from_('avatars').upload(
                path=file_path,
                file=file_content,
                file_options={"content-type": content_type, "upsert": "true"}
            )
            
            # Get public URL
            public_url = self.client.storage.from_('avatars').get_public_url(file_path)
            
            # Update profile with new picture URL
            await self.update_profile(user_id, profile_picture=public_url)
            
            logger.info(f"Profile picture uploaded for user {user_id}")
            return public_url
            
        except Exception as e:
            logger.error(f"Upload profile picture error: {e}")
            raise

    async def change_password(
        self,
        email: str,
        current_password: str,
        new_password: str
    ) -> MessageResponse:
        """
        Change user password after verifying current password.
        
        Args:
            email: User's email address
            current_password: Current password for verification
            new_password: New password to set
            
        Returns:
            MessageResponse: Success message
        """
        try:
            # First, verify the current password by attempting to sign in
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": current_password
            })
            
            if response.user is None or response.session is None:
                raise AuthenticationError("Current password is incorrect")
            
            # Now update the password for the authenticated user
            self.client.auth.update_user({
                "password": new_password
            })
            
            logger.info(f"Password changed for user: {email}")
            
            return MessageResponse(message="Password changed successfully")
            
        except AuthApiError as e:
            logger.warning(f"Change password failed: {e}")
            if "invalid" in str(e).lower() or "password" in str(e).lower():
                raise AuthenticationError("Current password is incorrect")
            raise AuthenticationError(str(e))
        except Exception as e:
            logger.error(f"Change password error: {e}")
            raise
