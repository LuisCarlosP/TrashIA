"""
Authentication exceptions.
"""
from exceptions.base_exception import TrashIAException


class AuthenticationError(TrashIAException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Invalid credentials"):
        super().__init__(
            message=message,
            code=401,
            error_type="AUTH_ERROR"
        )


class UserAlreadyExistsError(TrashIAException):
    """Raised when trying to register with an existing email."""
    
    def __init__(self, email: str):
        super().__init__(
            message=f"User with email '{email}' already exists",
            code=409,
            error_type="USER_EXISTS"
        )


class UserNotFoundError(TrashIAException):
    """Raised when user is not found."""
    
    def __init__(self, identifier: str = ""):
        message = f"User not found: {identifier}" if identifier else "User not found"
        super().__init__(
            message=message,
            code=404,
            error_type="USER_NOT_FOUND"
        )


class InvalidTokenError(TrashIAException):
    """Raised when token is invalid or expired."""
    
    def __init__(self, message: str = "Invalid or expired token"):
        super().__init__(
            message=message,
            code=401,
            error_type="INVALID_TOKEN"
        )


class TokenRevokedError(TrashIAException):
    """Raised when token has been revoked."""
    
    def __init__(self):
        super().__init__(
            message="Token has been revoked",
            code=401,
            error_type="TOKEN_REVOKED"
        )


class InactiveUserError(TrashIAException):
    """Raised when user account is deactivated."""
    
    def __init__(self):
        super().__init__(
            message="User account is deactivated",
            code=403,
            error_type="USER_INACTIVE"
        )


class WeakPasswordError(TrashIAException):
    """Raised when password doesn't meet requirements."""
    
    def __init__(self, message: str = "Password does not meet security requirements"):
        super().__init__(
            message=message,
            code=400,
            error_type="WEAK_PASSWORD"
        )
