from fastapi.testclient import TestClient
from app import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "trashia-api"
    assert "timestamp" in data


def test_dependencies_all_healthy():
    response = client.get("/health/dependencies")
    assert response.status_code == 200
    data = response.json()
    assert "dependencies" in data
    assert "timestamp" in data
    assert "status" in data


def test_health_endpoint_no_auth_required():
    response = client.get("/health")
    assert response.status_code == 200


def test_gemini_health_endpoint():
    response = client.get("/health/gemini")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "gemini"
    assert "status" in data


def test_osm_health_endpoint():
    response = client.get("/health/osm")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "openstreetmap"
    assert "status" in data


def test_openfoodfacts_health_endpoint():
    response = client.get("/health/openfoodfacts")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "openfoodfacts"
    assert "status" in data


def test_model_health_endpoint():
    """Test the ML model health check endpoint."""
    response = client.get("/health/model")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ml_model"
    assert "status" in data
    assert "model_loaded" in data
    assert "last_check" in data


def test_dependencies_includes_model():
    """Test that dependencies endpoint includes ML model status."""
    response = client.get("/health/dependencies")
    assert response.status_code == 200
    data = response.json()
    assert "ml_model" in data["dependencies"]
    assert "status" in data["dependencies"]["ml_model"]

