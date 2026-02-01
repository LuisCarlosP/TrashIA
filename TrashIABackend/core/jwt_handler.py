"""
JWT token handling utilities.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from uuid import uuid4

import jwt
from pydantic import BaseModel

from config.settings import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_REFRESH_TOKEN_EXPIRE_DAYS
)


class TokenPayload(BaseModel):
    """JWT token payload structure."""
    sub: str  # user_id
    exp: datetime
    iat: datetime
    jti: str  # unique token id for revocation
    type: str  # "access" or "refresh"


class TokenData(BaseModel):
    """Decoded token data for internal use."""
    user_id: str
    jti: str
    token_type: str
    expires_at: datetime


def create_access_token(user_id: str) -> Tuple[str, str]:
    """
    Create a JWT access token.
    
    Args:
        user_id: User's unique identifier
        
    Returns:
        Tuple[str, str]: (token, jti)
    """
    jti = str(uuid4())
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    payload = {
        "sub": user_id,
        "exp": expires,
        "iat": now,
        "jti": jti,
        "type": "access"
    }
    
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token, jti


def create_refresh_token(user_id: str) -> Tuple[str, str, datetime]:
    """
    Create a JWT refresh token.
    
    Args:
        user_id: User's unique identifier
        
    Returns:
        Tuple[str, str, datetime]: (token, jti, expires_at)
    """
    jti = str(uuid4())
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    
    payload = {
        "sub": user_id,
        "exp": expires,
        "iat": now,
        "jti": jti,
        "type": "refresh"
    }
    
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token, jti, expires


def decode_token(token: str) -> Optional[TokenData]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Optional[TokenData]: Decoded token data or None if invalid
    """
    try:
        payload = jwt.decode(
            token, 
            JWT_SECRET_KEY, 
            algorithms=[JWT_ALGORITHM]
        )
        
        return TokenData(
            user_id=payload["sub"],
            jti=payload["jti"],
            token_type=payload["type"],
            expires_at=datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        )
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
    except Exception:
        return None


def get_token_expiry_seconds() -> int:
    """Get access token expiry time in seconds."""
    return JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
