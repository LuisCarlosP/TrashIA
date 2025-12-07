import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tests.factories import PredictionDataFactory


@pytest.fixture(scope="function")
def validation_client():
    """Client with prediction service properly mocked via dependency overrides."""
    with patch("services.trash_services.ModelService._load_model"):
        from app import app
        from core.security import get_api_key
        from core.dependencies import get_prediction_service
        from routes.prediction import limiter
        
        limiter.enabled = False
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
        
        limiter.enabled = True
        app.dependency_overrides = {}


class TestFileSizeLimits:
    
    def test_file_size_within_limit_accepted(self, validation_client):
        image_bytes = PredictionDataFactory.create_image_bytes()
        
        with patch("magic.from_buffer", return_value="image/jpeg"):
            files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
            response = validation_client.post(
                "/predict",
                files=files,
                headers={"X-API-Key": "test-api-key"}
            )
            
            assert response.status_code == 200
    
    def test_file_size_limit_exceeded_rejected(self, validation_client):
        oversized_bytes = PredictionDataFactory.create_oversized_file(size_mb=6)
        
        with patch("magic.from_buffer", return_value="image/jpeg"):
            files = {"file": ("large.jpg", oversized_bytes, "image/jpeg")}
            response = validation_client.post(
                "/predict",
                files=files,
                headers={"X-API-Key": "test-api-key"}
            )
            
            assert response.status_code == 413
            response_data = response.json()
            error_msg = response_data.get("message", "") or response_data.get("detail", "")
            assert "exceeds" in error_msg.lower() or "5" in error_msg


class TestMIMETypeEnforcement:
    
    def test_mime_type_jpeg_accepted(self, validation_client):
        image_bytes = PredictionDataFactory.create_image_bytes(format='JPEG')
        
        with patch("magic.from_buffer", return_value="image/jpeg"):
            files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
            response = validation_client.post(
                "/predict",
                files=files,
                headers={"X-API-Key": "test-api-key"}
            )
            
            assert response.status_code == 200
    
    def test_mime_type_png_accepted(self, validation_client):
        image_bytes = PredictionDataFactory.create_image_bytes(format='PNG')
        
        with patch("magic.from_buffer", return_value="image/png"):
            files = {"file": ("test.png", image_bytes, "image/png")}
            response = validation_client.post(
                "/predict",
                files=files,
                headers={"X-API-Key": "test-api-key"}
            )
            
            assert response.status_code == 200
    
    def test_mime_type_pdf_rejected(self, validation_client):
        pdf_bytes = b'%PDF-1.4 fake pdf content'
        
        with patch("magic.from_buffer", return_value="application/pdf"):
            files = {"file": ("document.pdf", pdf_bytes, "application/pdf")}
            response = validation_client.post(
                "/predict",
                files=files,
                headers={"X-API-Key": "test-api-key"}
            )
            
            assert response.status_code == 400
            response_data = response.json()
            error_msg = response_data.get("message", "") or response_data.get("detail", "")
            assert "not allowed" in error_msg.lower()
    
    def test_mime_type_text_rejected(self, validation_client):
        text_bytes = b'This is plain text content'
        
        with patch("magic.from_buffer", return_value="text/plain"):
            files = {"file": ("readme.txt", text_bytes, "text/plain")}
            response = validation_client.post(
                "/predict",
                files=files,
                headers={"X-API-Key": "test-api-key"}
            )
            
            assert response.status_code == 400
    
    def test_mime_type_spoofed_extension_rejected(self, validation_client):
        fake_image = b'This is not actually an image file'
        
        with patch("magic.from_buffer", return_value="text/plain"):
            files = {"file": ("fake.jpg", fake_image, "image/jpeg")}
            response = validation_client.post(
                "/predict",
                files=files,
                headers={"X-API-Key": "test-api-key"}
            )
            
            assert response.status_code == 400
    
    def test_mime_type_gif_rejected(self, validation_client):
        gif_bytes = b'GIF89a' + b'\x00' * 100
        
        with patch("magic.from_buffer", return_value="image/gif"):
            files = {"file": ("animation.gif", gif_bytes, "image/gif")}
            response = validation_client.post(
                "/predict",
                files=files,
                headers={"X-API-Key": "test-api-key"}
            )
            
            assert response.status_code == 400
    
    def test_mime_type_webp_rejected(self, validation_client):
        webp_bytes = b'RIFF' + b'\x00' * 4 + b'WEBP' + b'\x00' * 100
        
        with patch("magic.from_buffer", return_value="image/webp"):
            files = {"file": ("photo.webp", webp_bytes, "image/webp")}
            response = validation_client.post(
                "/predict",
                files=files,
                headers={"X-API-Key": "test-api-key"}
            )
            
            assert response.status_code == 400
