from fastapi import Security, HTTPException, status, Request
from fastapi.security import APIKeyHeader
from config.settings import API_KEY

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(request: Request, api_key_header: str = Security(api_key_header)):
    # Allow OPTIONS requests (CORS preflight) without authentication
    if request.method == "OPTIONS":
        return None
    
    if not API_KEY:
        return None

    if api_key_header == API_KEY:
        return api_key_header
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials"
    )
