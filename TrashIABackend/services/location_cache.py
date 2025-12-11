from typing import Any, Optional
import time


class InMemoryCache:
    def __init__(self, default_ttl_seconds: int = 1800):
        self._cache: dict = {}
        self._default_ttl = default_ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if expiry and time.time() > expiry:
            del self._cache[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expiry = time.time() + ttl if ttl else None
        self._cache[key] = (value, expiry)

    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()

    def generate_key(self, *args) -> str:
        return ":".join(str(arg) for arg in args)
