"""
User repository for database operations.
"""
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from infrastructure.supabase_client import SupabaseClient
from models.user_models import UserInDB, UserRegisterRequest, UserCreate

logger = logging.getLogger(__name__)


class UserRepository:
    """Repository for user CRUD operations."""
    
    TABLE_NAME = "users"
    
    def __init__(self):
        self.client = SupabaseClient.get_client()
    
    async def create(self, user_data: UserRegisterRequest, password_hash: str) -> UserInDB:
        """
        Create a new user in the database.
        
        Args:
            user_data: User registration data
            password_hash: Hashed password
            
        Returns:
            UserInDB: Created user
        """
        data = {
            "name": user_data.name,
            "last_name": user_data.last_name,
            "email": user_data.email.lower(),  # Normalize email
            "password_hash": password_hash,
            "telephone": user_data.telephone,
            "profile_picture": user_data.profile_picture,
        }
        
        try:
            result = self.client.table(self.TABLE_NAME).insert(data).execute()
            
            if result.data:
                logger.info(f"User created successfully: {user_data.email}")
                return UserInDB(**result.data[0])
            
            raise Exception("Failed to create user: No data returned")
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise
    
    async def get_by_email(self, email: str) -> Optional[UserInDB]:
        """
        Find a user by email address.
        
        Args:
            email: User's email address
            
        Returns:
            Optional[UserInDB]: User if found, None otherwise
        """
        try:
            result = self.client.table(self.TABLE_NAME)\
                .select("*")\
                .eq("email", email.lower())\
                .execute()
            
            if result.data:
                return UserInDB(**result.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Error fetching user by email: {e}")
            raise
    
    async def get_by_id(self, user_id: UUID) -> Optional[UserInDB]:
        """
        Find a user by ID.
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            Optional[UserInDB]: User if found, None otherwise
        """
        try:
            result = self.client.table(self.TABLE_NAME)\
                .select("*")\
                .eq("id", str(user_id))\
                .execute()
            
            if result.data:
                return UserInDB(**result.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Error fetching user by ID: {e}")
            raise
    
    async def email_exists(self, email: str) -> bool:
        """
        Check if an email is already registered.
        
        Args:
            email: Email address to check
            
        Returns:
            bool: True if email exists, False otherwise
        """
        try:
            result = self.client.table(self.TABLE_NAME)\
                .select("id")\
                .eq("email", email.lower())\
                .execute()
            
            return len(result.data) > 0
            
        except Exception as e:
            logger.error(f"Error checking email existence: {e}")
            raise
    
    async def update(self, user_id: UUID, **kwargs) -> Optional[UserInDB]:
        """
        Update user fields.
        
        Args:
            user_id: User's unique identifier
            **kwargs: Fields to update
            
        Returns:
            Optional[UserInDB]: Updated user if found
        """
        try:
            result = self.client.table(self.TABLE_NAME)\
                .update(kwargs)\
                .eq("id", str(user_id))\
                .execute()
            
            if result.data:
                return UserInDB(**result.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            raise
    
    async def deactivate(self, user_id: UUID) -> bool:
        """
        Deactivate a user account.
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            bool: True if successful
        """
        try:
            result = self.client.table(self.TABLE_NAME)\
                .update({"is_active": False})\
                .eq("id", str(user_id))\
                .execute()
            
            return len(result.data) > 0
            
        except Exception as e:
            logger.error(f"Error deactivating user: {e}")
            raise


class TokenBlacklistRepository:
    """Repository for managing revoked tokens."""
    
    TABLE_NAME = "revoked_tokens"
    
    def __init__(self):
        self.client = SupabaseClient.get_client()
    
    async def add(self, jti: str, user_id: UUID, expires_at: datetime) -> None:
        """
        Add a token to the blacklist.
        
        Args:
            jti: JWT ID (unique token identifier)
            user_id: User who owns the token
            expires_at: Token's original expiration time
        """
        data = {
            "jti": jti,
            "user_id": str(user_id),
            "expires_at": expires_at.isoformat()
        }
        
        try:
            self.client.table(self.TABLE_NAME).insert(data).execute()
            logger.info(f"Token revoked: {jti[:8]}...")
            
        except Exception as e:
            # Ignore duplicate key errors (token already revoked)
            if "duplicate key" not in str(e).lower():
                logger.error(f"Error revoking token: {e}")
                raise
    
    async def is_revoked(self, jti: str) -> bool:
        """
        Check if a token has been revoked.
        
        Args:
            jti: JWT ID to check
            
        Returns:
            bool: True if token is revoked
        """
        try:
            result = self.client.table(self.TABLE_NAME)\
                .select("id")\
                .eq("jti", jti)\
                .execute()
            
            return len(result.data) > 0
            
        except Exception as e:
            logger.error(f"Error checking token revocation: {e}")
            # Fail secure: if we can't check, assume revoked
            return True
    
    async def cleanup_expired(self) -> int:
        """
        Remove expired tokens from the blacklist.
        
        Returns:
            int: Number of tokens removed
        """
        try:
            result = self.client.table(self.TABLE_NAME)\
                .delete()\
                .lt("expires_at", datetime.utcnow().isoformat())\
                .execute()
            
            count = len(result.data) if result.data else 0
            if count > 0:
                logger.info(f"Cleaned up {count} expired tokens")
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning up tokens: {e}")
            return 0
