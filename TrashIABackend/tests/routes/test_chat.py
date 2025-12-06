import pytest
from unittest.mock import MagicMock
from app import app
from routes.chat import get_chat_service

@pytest.fixture
def mock_chat_service():
    mock = MagicMock()
    app.dependency_overrides[get_chat_service] = lambda: mock
    yield mock

def test_create_session(client, mock_chat_service):
    mock_chat_service.create_chat_session.return_value = {
        "session_id": "test-session",
        "message": "Welcome",
        "language": "en"
    }
    
    response = client.post(
        "/chat/session", 
        json={"language": "en"}, 
        headers={"X-API-Key": "test-api-key"}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["session_id"] == "test-session"
    assert data["message"] == "Welcome"

def test_send_message(client, mock_chat_service):
    mock_chat_service.send_message.return_value = {
        "response": "This is a response",
        "session_id": "test-session",
        "on_topic": True
    }
    
    response = client.post(
        "/chat/message",
        json={"session_id": "test-session", "message": "Hello"},
        headers={"X-API-Key": "test-api-key"}
    )
    
    assert response.status_code == 200
    assert response.json()["response"] == "This is a response"

def test_get_history(client, mock_chat_service):
    mock_chat_service.get_chat_history.return_value = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"}
    ]
    
    response = client.get(
        "/chat/history/test-session",
        headers={"X-API-Key": "test-api-key"}
    )
    
    assert response.status_code == 200
    assert len(response.json()["history"]) == 2

def test_delete_session(client, mock_chat_service):
    mock_chat_service.delete_chat_session.return_value = True
    
    response = client.delete(
        "/chat/session/test-session",
        headers={"X-API-Key": "test-api-key"}
    )
    
    assert response.status_code == 200
    assert response.json()["message"] == "Session deleted successfully"

def test_delete_session_not_found(client, mock_chat_service):
    mock_chat_service.delete_chat_session.return_value = False
    
    response = client.delete(
        "/chat/session/invalid-session",
        headers={"X-API-Key": "test-api-key"}
    )
    
    assert response.status_code == 404
