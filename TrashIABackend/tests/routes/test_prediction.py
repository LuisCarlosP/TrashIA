import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import UploadFile
from app import app
from core.dependencies import get_prediction_service
from tests.factories import PredictionDataFactory


def test_predict_success(client):
    # Use factory for prediction response
    prediction_response = PredictionDataFactory.create_prediction_response(
        material="plastic",
        confidence=0.95,
        is_recyclable=True
    )
    
    mock_service = MagicMock()
    mock_service.predict_image = AsyncMock(return_value=prediction_response)
    
    app.dependency_overrides[get_prediction_service] = lambda: mock_service

    with patch("magic.from_buffer", return_value="image/jpeg"):
        # Use factory for image bytes
        image_bytes = PredictionDataFactory.create_image_bytes()
        files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
        
        response = client.post("/predict", files=files, headers={"X-API-Key": "test-api-key"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["class"] == "plastic"
        assert data["is_recyclable"] is True


def test_predict_invalid_mime_type(client):
    with patch("magic.from_buffer", return_value="text/plain"):
        invalid_bytes = PredictionDataFactory.create_invalid_file_bytes()
        files = {"file": ("test.txt", invalid_bytes, "text/plain")}
        
        response = client.post("/predict", files=files, headers={"X-API-Key": "test-api-key"})
        
        assert response.status_code == 400
        data = response.json()
        assert "File type not allowed" in data.get("detail", data.get("message", ""))


def test_predict_service_error(client):
    mock_service = MagicMock()
    from exceptions import PredictionError
    mock_service.predict_image = AsyncMock(side_effect=PredictionError("Model failed", "Test error"))
    mock_service.format_error.return_value = {"error": True, "message": "Model failed", "code": 500}
    
    app.dependency_overrides[get_prediction_service] = lambda: mock_service

    with patch("magic.from_buffer", return_value="image/jpeg"):
        image_bytes = PredictionDataFactory.create_image_bytes()
        files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
        response = client.post("/predict", files=files, headers={"X-API-Key": "test-api-key"})
        
        assert response.status_code == 500
        assert "error" in response.json() or "message" in response.json()
