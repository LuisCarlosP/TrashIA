from typing import Protocol, Optional, runtime_checkable
from abc import abstractmethod


@runtime_checkable
class FileValidatorProtocol(Protocol):
    @abstractmethod
    def validate(self, file_bytes: bytes, content_type: Optional[str] = None) -> None:
        ...
