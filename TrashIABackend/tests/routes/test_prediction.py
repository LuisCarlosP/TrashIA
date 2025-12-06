import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import UploadFile
from app import app
from core.dependencies import get_prediction_service

def test_predict_success(client):
    mock_service = MagicMock()
    mock_service.predict_image = AsyncMock(return_value={
        "clase": "plastic",
        "confianza": 0.95,
        "es_reciclable": True,
        "mensaje": "Test message"
    })
    
    app.dependency_overrides[get_prediction_service] = lambda: mock_service

    with patch("magic.from_buffer", return_value="image/jpeg"):
        file_content = b"fake image content"
        files = {"file": ("test.jpg", file_content, "image/jpeg")}
        
        response = client.post("/predict", files=files, headers={"X-API-Key": "test-api-key"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["clase"] == "plastic"
        assert data["es_reciclable"] is True

def test_predict_invalid_mime_type(client):
    with patch("magic.from_buffer", return_value="text/plain"):
        file_content = b"text content"
        files = {"file": ("test.txt", file_content, "text/plain")}
        
        response = client.post("/predict", files=files, headers={"X-API-Key": "test-api-key"})
        
        assert response.status_code == 400
        assert "Tipo de archivo no permitido" in response.json()["detail"]

def test_predict_service_error(client):
    mock_service = MagicMock()
    from exceptions import PredictionError
    mock_service.predict_image = AsyncMock(side_effect=PredictionError("Model failed", "Test error"))
    mock_service.format_error.return_value = {"error": True, "mensaje": "Model failed", "codigo": 500}
    
    app.dependency_overrides[get_prediction_service] = lambda: mock_service

    with patch("magic.from_buffer", return_value="image/jpeg"):
        files = {"file": ("test.jpg", b"content", "image/jpeg")}
        response = client.post("/predict", files=files, headers={"X-API-Key": "test-api-key"})
        
        assert response.status_code == 500
        assert "error" in response.json() or "mensaje" in response.json()
