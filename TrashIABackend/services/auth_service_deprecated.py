"""
Authentication service - Business logic for user authentication.
"""
import logging
from typing import Optional
from uuid import UUID

from models.user_models import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
    AuthResponse
)
from repositories.user_repository import UserRepository, TokenBlacklistRepository
from core.password import hash_password, verify_password
from core.jwt_handler import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_token_expiry_seconds
)
from exceptions.auth_exceptions import (
    AuthenticationError,
    UserAlreadyExistsError,
    UserNotFoundError,
    InvalidTokenError,
    TokenRevokedError,
    InactiveUserError
)

logger = logging.getLogger(__name__)


class AuthService:
    """
    Authentication service implementing user registration, login, and token management.
    """
    
    def __init__(
        self,
        user_repository: Optional[UserRepository] = None,
        token_blacklist: Optional[TokenBlacklistRepository] = None
    ):
        """
        Initialize the auth service with repositories.
        
        Args:
            user_repository: Repository for user operations
            token_blacklist: Repository for token blacklist operations
        """
        self.user_repo = user_repository or UserRepository()
        self.token_blacklist = token_blacklist or TokenBlacklistRepository()
    
    async def register(self, user_data: UserRegisterRequest) -> AuthResponse:
        """
        Register a new user.
        
        Args:
            user_data: User registration data
            
        Returns:
            AuthResponse: User data and authentication tokens
            
        Raises:
            UserAlreadyExistsError: If email is already registered
        """
        # Check if email already exists
        if await self.user_repo.email_exists(user_data.email):
            logger.warning(f"Registration attempt with existing email: {user_data.email}")
            raise UserAlreadyExistsError(user_data.email)
        
        # Hash the password
        password_hash = hash_password(user_data.password)
        
        # Create the user
        user = await self.user_repo.create(user_data, password_hash)
        logger.info(f"New user registered: {user.email}")
        
        # Generate tokens
        tokens = self._generate_tokens(str(user.id))
        
        return AuthResponse(
            user=UserResponse(
                id=user.id,
                name=user.name,
                last_name=user.last_name,
                email=user.email,
                telephone=user.telephone,
                profile_picture=user.profile_picture,
                is_active=user.is_active,
                created_at=user.created_at
            ),
            tokens=tokens
        )
    
    async def login(self, credentials: UserLoginRequest) -> AuthResponse:
        """
        Authenticate a user with email and password.
        
        Args:
            credentials: Login credentials
            
        Returns:
            AuthResponse: User data and authentication tokens
            
        Raises:
            AuthenticationError: If credentials are invalid
            InactiveUserError: If user account is deactivated
        """
        # Find user by email
        user = await self.user_repo.get_by_email(credentials.email)
        
        if not user:
            logger.warning(f"Login attempt for non-existent user: {credentials.email}")
            raise AuthenticationError("Invalid email or password")
        
        # Verify password
        if not verify_password(credentials.password, user.password_hash):
            logger.warning(f"Failed login attempt for user: {credentials.email}")
            raise AuthenticationError("Invalid email or password")
        
        # Check if account is active
        if not user.is_active:
            logger.warning(f"Login attempt for inactive account: {credentials.email}")
            raise InactiveUserError()
        
        logger.info(f"User logged in: {user.email}")
        
        # Generate tokens
        tokens = self._generate_tokens(str(user.id))
        
        return AuthResponse(
            user=UserResponse(
                id=user.id,
                name=user.name,
                last_name=user.last_name,
                email=user.email,
                telephone=user.telephone,
                profile_picture=user.profile_picture,
                is_active=user.is_active,
                created_at=user.created_at
            ),
            tokens=tokens
        )
    
    async def logout(
        self, 
        access_token: str, 
        refresh_token: Optional[str] = None
    ) -> None:
        """
        Logout user by revoking tokens.
        
        Args:
            access_token: Current access token
            refresh_token: Current refresh token (optional)
        """
        # Revoke access token
        access_data = decode_token(access_token)
        if access_data:
            await self.token_blacklist.add(
                jti=access_data.jti,
                user_id=UUID(access_data.user_id),
                expires_at=access_data.expires_at
            )
            logger.info(f"Access token revoked for user: {access_data.user_id}")
        
        # Revoke refresh token if provided
        if refresh_token:
            refresh_data = decode_token(refresh_token)
            if refresh_data:
                await self.token_blacklist.add(
                    jti=refresh_data.jti,
                    user_id=UUID(refresh_data.user_id),
                    expires_at=refresh_data.expires_at
                )
                logger.info(f"Refresh token revoked for user: {refresh_data.user_id}")
    
    async def refresh_tokens(self, refresh_token: str) -> TokenResponse:
        """
        Generate new tokens using a valid refresh token.
        
        Args:
            refresh_token: Current refresh token
            
        Returns:
            TokenResponse: New token pair
            
        Raises:
            InvalidTokenError: If refresh token is invalid
            TokenRevokedError: If refresh token has been revoked
        """
        # Decode refresh token
        token_data = decode_token(refresh_token)
        
        if not token_data:
            raise InvalidTokenError("Invalid or expired refresh token")
        
        if token_data.token_type != "refresh":
            raise InvalidTokenError("Invalid token type")
        
        # Check if token is revoked
        if await self.token_blacklist.is_revoked(token_data.jti):
            raise TokenRevokedError()
        
        # Verify user exists and is active
        user = await self.user_repo.get_by_id(UUID(token_data.user_id))
        if not user:
            raise InvalidTokenError("User not found")
        
        if not user.is_active:
            raise InactiveUserError()
        
        # Revoke the old refresh token
        await self.token_blacklist.add(
            jti=token_data.jti,
            user_id=UUID(token_data.user_id),
            expires_at=token_data.expires_at
        )
        
        # Generate new tokens
        logger.info(f"Tokens refreshed for user: {user.email}")
        return self._generate_tokens(token_data.user_id)
    
    async def get_current_user(self, token: str) -> UserResponse:
        """
        Get the currently authenticated user from a token.
        
        Args:
            token: Access token
            
        Returns:
            UserResponse: Current user data
            
        Raises:
            InvalidTokenError: If token is invalid
            TokenRevokedError: If token has been revoked
            UserNotFoundError: If user doesn't exist
        """
        # Decode token
        token_data = decode_token(token)
        
        if not token_data:
            raise InvalidTokenError()
        
        if token_data.token_type != "access":
            raise InvalidTokenError("Invalid token type")
        
        # Check if token is revoked
        if await self.token_blacklist.is_revoked(token_data.jti):
            raise TokenRevokedError()
        
        # Get user
        user = await self.user_repo.get_by_id(UUID(token_data.user_id))
        
        if not user:
            raise UserNotFoundError()
        
        if not user.is_active:
            raise InactiveUserError()
        
        return UserResponse(
            id=user.id,
            name=user.name,
            last_name=user.last_name,
            email=user.email,
            telephone=user.telephone,
            profile_picture=user.profile_picture,
            is_active=user.is_active,
            created_at=user.created_at
        )
    
    async def validate_token(self, token: str) -> bool:
        """
        Validate an access token without returning user data.
        
        Args:
            token: Access token to validate
            
        Returns:
            bool: True if token is valid
        """
        token_data = decode_token(token)
        
        if not token_data or token_data.token_type != "access":
            return False
        
        if await self.token_blacklist.is_revoked(token_data.jti):
            return False
        
        return True
    
    def _generate_tokens(self, user_id: str) -> TokenResponse:
        """
        Generate a new access/refresh token pair.
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            TokenResponse: Token pair
        """
        access_token, _ = create_access_token(user_id)
        refresh_token, _, _ = create_refresh_token(user_id)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=get_token_expiry_seconds()
        )
