from typing import Protocol, Dict, Any, runtime_checkable
from abc import abstractmethod


@runtime_checkable
class ResponseFormatterProtocol(Protocol):
    @abstractmethod
    def format_success(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ...
    
    @abstractmethod
    def format_error(self, message: str, code: int = 500) -> Dict[str, Any]:
        ...
