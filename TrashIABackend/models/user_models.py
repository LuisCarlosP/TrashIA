"""
User models for authentication with Supabase Auth.
"""
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator
import re


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class UserRegisterRequest(BaseModel):
    """Schema for user registration."""
    name: str = Field(..., min_length=2, max_length=100, description="User's first name")
    last_name: str = Field(..., min_length=2, max_length=100, description="User's last name")
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=8, max_length=128, description="User's password")
    telephone: Optional[str] = Field(None, max_length=20, description="User's phone number (optional)")
    profile_picture: Optional[str] = Field(None, description="URL to profile picture (optional)")
    
    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password meets security requirements."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v
    
    @field_validator('telephone')
    @classmethod
    def validate_telephone(cls, v: Optional[str]) -> Optional[str]:
        """Validate phone number format if provided."""
        if v is None:
            return v
        # Remove spaces and dashes for validation
        cleaned = re.sub(r'[\s\-\(\)]', '', v)
        if not re.match(r'^\+?\d{8,15}$', cleaned):
            raise ValueError('Invalid phone number format')
        return v


class UserLoginRequest(BaseModel):
    """Schema for user login."""
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")


class RefreshTokenRequest(BaseModel):
    """Schema for token refresh."""
    refresh_token: str = Field(..., description="Refresh token")


class LogoutRequest(BaseModel):
    """Schema for logout (optional, can use headers instead)."""
    refresh_token: Optional[str] = Field(None, description="Refresh token to revoke")


class ResendVerificationRequest(BaseModel):
    """Schema for resending verification email."""
    email: EmailStr = Field(..., description="Email to resend verification to")


class ForgotPasswordRequest(BaseModel):
    """Schema for forgot password request."""
    email: EmailStr = Field(..., description="Email to send reset link to")


class ResetPasswordRequest(BaseModel):
    """Schema for password reset."""
    access_token: str = Field(..., description="Reset token from email link")
    refresh_token: Optional[str] = Field(None, description="Refresh token from email link")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")
    
    @field_validator('new_password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password meets security requirements."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v


class UpdateProfileRequest(BaseModel):
    """Schema for profile updates."""
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    last_name: Optional[str] = Field(None, min_length=2, max_length=100)
    telephone: Optional[str] = Field(None, max_length=20)
    profile_picture: Optional[str] = Field(None)


class ChangePasswordRequest(BaseModel):
    """Schema for changing password with current password verification."""
    current_password: str = Field(..., description="Current password for verification")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")
    
    @field_validator('new_password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password meets security requirements."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class UserProfile(BaseModel):
    """User profile data from profiles table."""
    id: UUID
    name: str
    last_name: str
    telephone: Optional[str] = None
    profile_picture: Optional[str] = None
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserResponse(BaseModel):
    """User data response combining auth.users and profiles."""
    id: UUID
    email: str
    email_verified: bool = False
    profile: Optional[UserProfile] = None
    
    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Token pair response from Supabase Auth."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(default=3600, description="Access token expiry in seconds")
    expires_at: Optional[int] = Field(None, description="Token expiry timestamp")


class AuthResponse(BaseModel):
    """Complete authentication response with user and tokens."""
    user: UserResponse
    tokens: TokenResponse
    message: Optional[str] = None


class MessageResponse(BaseModel):
    """Simple message response."""
    message: str
    success: bool = True


class ProfilePictureResponse(BaseModel):
    """Response for profile picture upload."""
    url: str
    message: str


# =============================================================================
# INTERNAL MODELS
# =============================================================================

class UserInDB(BaseModel):
    """Internal user model including password hash."""
    id: UUID
    name: str
    last_name: str
    email: str
    password_hash: str
    telephone: Optional[str] = None
    profile_picture: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    """Model for creating a user in the database."""
    name: str
    last_name: str
    email: str
    password_hash: str
    telephone: Optional[str] = None
    profile_picture: Optional[str] = None
