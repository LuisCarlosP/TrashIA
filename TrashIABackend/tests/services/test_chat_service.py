import pytest
from unittest.mock import MagicMock
from services.chat_service import ChatService, MaterialContext
from exceptions import ValidationError


class MockChatProvider:
    """Mock implementation of ChatProviderProtocol for testing."""
    
    def __init__(self):
        self.mock_chat = MagicMock()
    
    def start_chat(self, system_context: str = None):
        return self.mock_chat
    
    def send_message(self, chat_session, message: str):
        return "AI Response"


class MockSessionRepository:
    """Mock implementation of ChatSessionRepositoryProtocol for testing."""
    
    def __init__(self):
        self._sessions = {}
    
    def get(self, session_id: str):
        return self._sessions.get(session_id)
    
    def save(self, session_id: str, session_data):
        self._sessions[session_id] = session_data
    
    def delete(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def exists(self, session_id: str):
        return session_id in self._sessions


@pytest.fixture
def mock_provider():
    return MockChatProvider()


@pytest.fixture
def mock_repository():
    return MockSessionRepository()


@pytest.fixture
def chat_service(mock_provider, mock_repository):
    return ChatService(chat_provider=mock_provider, session_repository=mock_repository)


def test_create_chat_session(chat_service, mock_repository):
    result = chat_service.create_chat_session("test-session", language="en")
    
    assert result["session_id"] == "test-session"
    assert mock_repository.exists("test-session")


def test_create_chat_session_with_material(chat_service):
    context = MaterialContext("plastic", True, "bottle")
    result = chat_service.create_chat_session("test-session", material_context=context)
    
    assert "plastic" in result["message"] or "Plastic" in result["message"]


def test_send_message(chat_service):
    chat_service.create_chat_session("test-session")
    result = chat_service.send_message("test-session", "Hello")
    
    assert result["response"] == "AI Response"
    history = chat_service.get_chat_history("test-session")
    assert len(history) == 2


def test_send_message_no_session(chat_service):
    result = chat_service.send_message("invalid-session", "Hello")
    
    assert "error" in result or "response" in result
    assert result.get("session_id") == "invalid-session"


def test_get_chat_history(chat_service):
    chat_service.create_chat_session("test-session")
    chat_service.send_message("test-session", "Hello")
    
    history = chat_service.get_chat_history("test-session")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_delete_chat_session(chat_service, mock_repository):
    chat_service.create_chat_session("test-session")
    assert mock_repository.exists("test-session")
    
    result = chat_service.delete_chat_session("test-session")
    assert result is True
    assert not mock_repository.exists("test-session")


def test_delete_chat_session_not_found(chat_service):
    result = chat_service.delete_chat_session("nonexistent")
    assert result is False
