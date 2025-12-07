import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from exceptions.base_exception import TrashIAException

logger = logging.getLogger(__name__)


def generate_correlation_id() -> str:
    return str(uuid.uuid4())


def create_error_response(
    message: str,
    code: int,
    error_type: str = "Error",
    correlation_id: str = None,
    details: Dict[str, Any] = None
) -> Dict[str, Any]:
    response = {
        "error": True,
        "code": code,
        "message": message,
        "error_type": error_type,
        "correlation_id": correlation_id or generate_correlation_id(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    if details:
        response["details"] = details
        
    return response


async def trashia_exception_handler(request: Request, exc: TrashIAException) -> JSONResponse:
    correlation_id = generate_correlation_id()
    exc.correlation_id = correlation_id
    
    logger.error(
        f"[{correlation_id}] {exc.error_type}: {exc.message}",
        extra={
            "correlation_id": correlation_id,
            "error_type": exc.error_type,
            "code": exc.code,
            "path": request.url.path,
            "details": exc.details
        }
    )
    
    response_data = exc.to_dict()
    response_data["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    return JSONResponse(status_code=exc.code, content=response_data)


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    correlation_id = generate_correlation_id()
    
    logger.warning(
        f"[{correlation_id}] HTTPException: {exc.detail}",
        extra={
            "correlation_id": correlation_id,
            "code": exc.status_code,
            "path": request.url.path
        }
    )
    
    response_data = create_error_response(
        message=str(exc.detail),
        code=exc.status_code,
        error_type="HTTPException",
        correlation_id=correlation_id
    )
    
    return JSONResponse(status_code=exc.status_code, content=response_data)


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    correlation_id = generate_correlation_id()
    
    logger.exception(
        f"[{correlation_id}] Unhandled exception: {exc}",
        extra={
            "correlation_id": correlation_id,
            "path": request.url.path
        }
    )
    
    response_data = create_error_response(
        message="An unexpected error occurred. Please try again later.",
        code=500,
        error_type="InternalServerError",
        correlation_id=correlation_id
    )
    
    return JSONResponse(status_code=500, content=response_data)


def register_exception_handlers(app):
    from starlette.exceptions import HTTPException as StarletteHTTPException
    
    app.add_exception_handler(TrashIAException, trashia_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    
    logger.info("Exception handlers registered")
