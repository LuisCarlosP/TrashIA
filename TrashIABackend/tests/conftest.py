import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import services.trash_services
with patch("services.trash_services.ModelService._load_model"):
    from app import app
    from core.dependencies import get_prediction_service
    from core.security import get_api_key

@pytest.fixture
def client():
    app.dependency_overrides[get_api_key] = lambda: "test-api-key"
    
    mock_service = MagicMock()
    app.dependency_overrides[get_prediction_service] = lambda: mock_service
    
    with patch("services.trash_services.ModelService._load_model"):
        with TestClient(app) as client:
            yield client
    
    app.dependency_overrides = {}
