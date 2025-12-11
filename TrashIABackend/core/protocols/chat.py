from typing import Protocol, Dict, Any, Optional, runtime_checkable
from abc import abstractmethod


@runtime_checkable
class ChatProviderProtocol(Protocol):
    @abstractmethod
    def start_chat(self, system_context: str) -> Any:
        ...
    
    @abstractmethod
    def send_message(self, chat_session: Any, message: str) -> str:
        ...


@runtime_checkable
class ChatSessionRepositoryProtocol(Protocol):
    @abstractmethod
    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        ...
    
    @abstractmethod
    def save(self, session_id: str, session_data: Dict[str, Any]) -> None:
        ...
    
    @abstractmethod
    def delete(self, session_id: str) -> bool:
        ...
    
    @abstractmethod
    def exists(self, session_id: str) -> bool:
        ...
