from typing import Optional, Dict, Any
from .base_exception import TrashIAException


class ExternalAPIError(TrashIAException):
    
    def __init__(
        self, 
        message: str = "External API error",
        code: int = 503,
        service: Optional[str] = None,
        original_error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.service = service
        self.original_error = original_error
        
        combined_details = details or {}
        if service:
            combined_details["service"] = service
        if original_error:
            combined_details["original_error"] = original_error
            
        super().__init__(message, code, details=combined_details)


class GroqAPIError(ExternalAPIError):
    
    def __init__(
        self, 
        message: str = "Groq API error",
        original_error: Optional[str] = None,
        code: int = 503
    ):
        super().__init__(
            message=message, 
            code=code, 
            service="groq",
            original_error=original_error
        )


class OpenFoodFactsError(ExternalAPIError):
    
    def __init__(
        self, 
        message: str = "Open Food Facts API error",
        original_error: Optional[str] = None,
        code: int = 503
    ):
        super().__init__(
            message=message, 
            code=code, 
            service="openfoodfacts",
            original_error=original_error
        )


class UPCItemDBError(ExternalAPIError):
    
    def __init__(
        self, 
        message: str = "UPCitemdb API error",
        original_error: Optional[str] = None,
        code: int = 503
    ):
        super().__init__(
            message=message, 
            code=code, 
            service="upcitemdb",
            original_error=original_error
        )


class OpenStreetMapError(ExternalAPIError):
    
    def __init__(
        self, 
        message: str = "OpenStreetMap API error",
        original_error: Optional[str] = None,
        code: int = 503
    ):
        super().__init__(
            message=message, 
            code=code, 
            service="openstreetmap",
            original_error=original_error
        )


class CircuitBreakerOpenError(ExternalAPIError):
    
    def __init__(
        self, 
        service: str,
        message: Optional[str] = None
    ):
        default_message = f"Service temporarily unavailable: {service}. Please try again later."
        super().__init__(
            message=message or default_message, 
            code=503, 
            service=service
        )
