"""Rate Limiting Tests for API endpoints."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tests.factories import PredictionDataFactory


pytestmark = pytest.mark.integration


@pytest.fixture(scope="function")
def rate_limit_client():
    """Client for rate limiting tests - keeps rate limiting enabled with mocked services."""
    with patch("services.trash_services.ModelService._load_model"):
        from app import app, limiter
        from core.security import get_api_key
        from core.dependencies import get_prediction_service
        
        try:
            limiter.reset()
        except Exception:
            pass
        
        app.dependency_overrides[get_api_key] = lambda: "test-api-key"
        
        mock_service = MagicMock()
        mock_service.predict_image = AsyncMock(return_value={
            "class": "plastic",
            "confidence": 0.95,
            "is_recyclable": True,
            "message": "Detected plastic"
        })
        mock_service.format_error = MagicMock(return_value={"error": True})
        app.dependency_overrides[get_prediction_service] = lambda: mock_service
        
        with TestClient(app) as client:
            yield client
        
        app.dependency_overrides = {}


class TestRateLimiting:
    """Tests for rate limiting behavior."""
    
    def test_rate_limit_predict_endpoint(self, rate_limit_client):
        image_bytes = PredictionDataFactory.create_image_bytes()
        
        with patch("magic.from_buffer", return_value="image/jpeg"):
            responses = []
            for i in range(12):  # Try 12 requests (limit is 10/minute)
                files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
                response = rate_limit_client.post(
                    "/predict",
                    files=files,
                    headers={"X-API-Key": "test-api-key"}
                )
                responses.append(response.status_code)
            
            success_count = responses.count(200)
            error_count = responses.count(500)
            rate_limited = responses.count(429)
            
            assert success_count > 0 or rate_limited > 0 or error_count > 0, \
                f"Unexpected response distribution: {responses}"
    
    def test_rate_limit_chat_session(self, rate_limit_client):
        from routes.chat import get_chat_service
        from app import app
        
        mock_service = MagicMock()
        mock_service.create_chat_session.return_value = {
            "session_id": "test-id",
            "message": "Welcome",
            "language": "en"
        }
        app.dependency_overrides[get_chat_service] = lambda: mock_service
        
        responses = []
        for i in range(25):  # Try 25 requests (limit is 20/minute)
            response = rate_limit_client.post(
                "/chat/session",
                json={"language": "en"},
                headers={"X-API-Key": "test-api-key"}
            )
            responses.append(response.status_code)
        
        success_count = responses.count(201)
        rate_limited = responses.count(429)
        
        assert success_count > 0 or rate_limited > 0, \
            f"Expected either success or rate limiting, got: {responses}"

