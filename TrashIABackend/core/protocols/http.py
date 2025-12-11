from typing import Protocol, Dict, Any, runtime_checkable
from abc import abstractmethod


@runtime_checkable
class HttpClientProtocol(Protocol):
    @abstractmethod
    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        ...
    
    @abstractmethod
    async def post(self, url: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        ...
