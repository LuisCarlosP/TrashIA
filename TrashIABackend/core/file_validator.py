import magic
from typing import Optional, Set
from fastapi import HTTPException
from config.settings import MAX_FILE_SIZE, ALLOWED_MIME_TYPES
from exceptions import ValidationError


class FileSizeExceededError(Exception):
    def __init__(self, message: str, max_mb: int):
        self.message = message
        self.max_mb = max_mb
        super().__init__(message)


class FileValidator:
    def __init__(
        self,
        max_size: int = MAX_FILE_SIZE,
        allowed_mime_types: Set[str] = None
    ):
        self._max_size = max_size
        self._allowed_mime_types = allowed_mime_types or set(ALLOWED_MIME_TYPES)

    def validate(self, file_bytes: bytes, content_type: Optional[str] = None) -> None:
        self._validate_size(file_bytes)
        self._validate_mime_type(file_bytes)
        if content_type:
            self._validate_content_type(content_type)

    def _validate_size(self, file_bytes: bytes) -> None:
        if len(file_bytes) > self._max_size:
            max_mb = self._max_size // (1024 * 1024)
            raise FileSizeExceededError(
                f"File exceeds maximum allowed size of {max_mb}MB",
                max_mb
            )

    def _validate_mime_type(self, file_bytes: bytes) -> None:
        mime = magic.from_buffer(file_bytes, mime=True)
        if mime not in self._allowed_mime_types:
            raise ValidationError(
                f"File type not allowed. Only accepted: {', '.join(self._allowed_mime_types)}"
            )

    def _validate_content_type(self, content_type: str) -> None:
        if not content_type.startswith("image/"):
            raise ValidationError("File must be a valid image")
