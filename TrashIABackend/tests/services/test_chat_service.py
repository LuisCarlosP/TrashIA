import pytest
from unittest.mock import MagicMock, patch
from services.chat_service import ChatService, MaterialContext
from exceptions import ValidationError

@pytest.fixture
def mock_genai():
    with patch("google.generativeai.GenerativeModel") as mock:
        yield mock

def test_create_chat_session(mock_genai):
    service = ChatService()
    mock_chat = MagicMock()
    service.model.start_chat.return_value = mock_chat
    
    result = service.create_chat_session("test-session", language="en")
    
    assert result["session_id"] == "test-session"
    assert "test-session" in service.chat_sessions

def test_create_chat_session_with_material(mock_genai):
    service = ChatService()
    service.model.start_chat.return_value = MagicMock()
    
    context = MaterialContext("plastic", True, "bottle")
    result = service.create_chat_session("test-session", material_context=context)
    
    assert "plastic" in result["message"] or "Plastic" in result["message"]

def test_send_message(mock_genai):
    service = ChatService()
    mock_chat = MagicMock()
    mock_chat.send_message.return_value.text = "AI Response"
    service.model.start_chat.return_value = mock_chat
    
    service.create_chat_session("test-session")
    result = service.send_message("test-session", "Hello")
    
    assert result["response"] == "AI Response"
    assert len(service.chat_sessions["test-session"]["history"]) == 2

def test_send_message_no_session(mock_genai):
    service = ChatService()
    result = service.send_message("invalid-session", "Hello")
    
    assert "error" in result or "response" in result
    assert result.get("session_id") == "invalid-session"
