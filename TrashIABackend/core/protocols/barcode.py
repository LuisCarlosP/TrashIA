from typing import Protocol, Dict, Any, Optional, runtime_checkable
from abc import abstractmethod


@runtime_checkable
class BarcodeProviderProtocol(Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        ...
    
    @abstractmethod
    async def fetch_product(self, barcode: str) -> Optional[Dict[str, Any]]:
        ...
