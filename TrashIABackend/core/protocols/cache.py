from typing import Protocol, Any, Optional, runtime_checkable
from abc import abstractmethod


@runtime_checkable
class CacheProtocol(Protocol):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        ...
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        ...
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        ...
    
    @abstractmethod
    def clear(self) -> None:
        ...
