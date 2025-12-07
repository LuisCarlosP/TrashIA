import logging
from fastapi import Security, HTTPException, status, Request
from fastapi.security import APIKeyHeader
from config.settings import API_KEY, ENVIRONMENT

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def validate_security_config() -> None:
    """Validate security configuration at startup.
    
    Raises:
        RuntimeError: If API_KEY is not set in production environment.
    """
    if ENVIRONMENT == "production" and not API_KEY:
        raise RuntimeError(
            "API_KEY environment variable must be set in production. "
            "Set ENVIRONMENT=development to disable this check."
        )
    if not API_KEY:
        logger.warning(
            "API_KEY not set - authentication is disabled. "
            "This is only acceptable in development environments."
        )

async def get_api_key(request: Request, api_key_header: str = Security(api_key_header)):
    # Allow OPTIONS requests (CORS preflight) without authentication
    if request.method == "OPTIONS":
        return None
    
    if not API_KEY:  # Already validated at startup; dev mode only
        return None

    if api_key_header == API_KEY:
        return api_key_header
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials"
    )
