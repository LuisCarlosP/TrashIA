import pytest
from unittest.mock import MagicMock
from app import app
from routes.chat import get_chat_service
from tests.factories import ChatDataFactory


@pytest.fixture
def mock_chat_service():
    mock = MagicMock()
    app.dependency_overrides[get_chat_service] = lambda: mock
    yield mock


def test_create_session(client, mock_chat_service):
    # Use factory for session response
    session_response = ChatDataFactory.create_session_response(
        session_id="test-session"
    )
    mock_chat_service.create_chat_session.return_value = session_response
    
    # Use factory for session request
    session_request = ChatDataFactory.create_session_request(language="en")
    
    response = client.post(
        "/chat/session", 
        json=session_request, 
        headers={"X-API-Key": "test-api-key"}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["session_id"] == "test-session"
    assert "message" in data


def test_send_message(client, mock_chat_service):
    # Use factory for message response
    message_response = ChatDataFactory.create_message_response(
        session_id="test-session",
        response="This is a response"
    )
    mock_chat_service.send_message.return_value = message_response
    
    # Use factory for message request
    message_request = ChatDataFactory.create_message_request(
        session_id="test-session",
        message="Hello"
    )
    
    response = client.post(
        "/chat/message",
        json=message_request,
        headers={"X-API-Key": "test-api-key"}
    )
    
    assert response.status_code == 200
    assert response.json()["response"] == "This is a response"


def test_get_history(client, mock_chat_service):
    # Use factory for chat history
    history = ChatDataFactory.create_chat_history(message_count=2)
    mock_chat_service.get_chat_history.return_value = history
    
    session_id = ChatDataFactory.create_session_id()
    
    response = client.get(
        f"/chat/history/{session_id}",
        headers={"X-API-Key": "test-api-key"}
    )
    
    assert response.status_code == 200
    assert len(response.json()["history"]) == 2


def test_delete_session(client, mock_chat_service):
    mock_chat_service.delete_chat_session.return_value = True
    
    session_id = ChatDataFactory.create_session_id()
    
    response = client.delete(
        f"/chat/session/{session_id}",
        headers={"X-API-Key": "test-api-key"}
    )
    
    assert response.status_code == 200
    assert response.json()["message"] == "Session deleted successfully"


def test_delete_session_not_found(client, mock_chat_service):
    mock_chat_service.delete_chat_session.return_value = False
    
    session_id = ChatDataFactory.create_session_id()
    
    response = client.delete(
        f"/chat/session/{session_id}",
        headers={"X-API-Key": "test-api-key"}
    )
    
    assert response.status_code == 404
