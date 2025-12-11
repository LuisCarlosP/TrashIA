from typing import Dict, Any, Optional


class InMemoryChatSessionRepository:
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    def save(self, session_id: str, session_data: Dict[str, Any]) -> None:
        self._sessions[session_id] = session_data

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def clear(self) -> None:
        self._sessions.clear()
