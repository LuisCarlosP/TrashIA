import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["TESTING"] = "true"

import services.trash_services
with patch("services.trash_services.ModelService._load_model"):
    from app import app
    from core.dependencies import get_prediction_service
    from core.security import get_api_key
    from routes.prediction import limiter as prediction_limiter
    from routes.chat import limiter as chat_limiter
    from routes.location import limiter as location_limiter
    from routes.barcode import limiter as barcode_limiter

@pytest.fixture(autouse=True)
def disable_rate_limiting():
    """Disable all rate limiters for tests."""
    prediction_limiter.enabled = False
    chat_limiter.enabled = False
    location_limiter.enabled = False
    barcode_limiter.enabled = False
    yield
    prediction_limiter.enabled = True
    chat_limiter.enabled = True
    location_limiter.enabled = True
    barcode_limiter.enabled = True

@pytest.fixture
def client():
    app.dependency_overrides[get_api_key] = lambda: "test-api-key"
    
    mock_service = MagicMock()
    app.dependency_overrides[get_prediction_service] = lambda: mock_service
    
    with patch("services.trash_services.ModelService._load_model"):
        with TestClient(app) as client:
            yield client
    
    app.dependency_overrides = {}

