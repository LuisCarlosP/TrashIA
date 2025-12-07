from typing import Any, Dict, Optional


class TrashIAException(Exception):
    
    def __init__(
        self, 
        message: str, 
        code: int = 500, 
        error_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.error_type = error_type or self.__class__.__name__
        self.details = details or {}
        self.correlation_id: Optional[str] = None
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        response = {
            "error": True,
            "code": self.code,
            "message": self.message,
            "error_type": self.error_type,
        }
        
        if self.correlation_id:
            response["correlation_id"] = self.correlation_id
        
        if self.details:
            response["details"] = self.details
            
        return response
