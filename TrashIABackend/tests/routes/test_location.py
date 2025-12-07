import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from models.location_models import RecyclingPoint
from tests.factories import LocationDataFactory


def test_get_recycling_points_success(client):
    # Use factory to create recycling point data
    point_data = LocationDataFactory.create_recycling_point(
        id="node_1",
        name="Point 1",
        latitude=40.0,
        longitude=-3.0,
        distance=100.0,
        types=["plastic"]
    )
    
    mock_points = [RecyclingPoint(**point_data)]
    
    mock_service = MagicMock()
    mock_service.get_recycling_points = AsyncMock(return_value=mock_points)
    
    coords = LocationDataFactory.create_coordinates(latitude=40.0, longitude=-3.0)
    
    with patch("routes.location.get_location_service", return_value=mock_service):
        response = client.get(
            f"/location/recycling-points?latitude={coords['latitude']}&longitude={coords['longitude']}",
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["points"]) == 1
        assert data["points"][0]["name"] == "Point 1"


def test_get_recycling_points_invalid_coords(client):
    # Invalid latitude (> 90)
    response = client.get(
        "/location/recycling-points?latitude=100.0&longitude=-3.0",
        headers={"X-API-Key": "test-api-key"}
    )
    assert response.status_code == 422


def test_search_recycling_points_post(client):
    mock_points = []
    mock_service = MagicMock()
    mock_service.get_recycling_points = AsyncMock(return_value=mock_points)
    
    coords = LocationDataFactory.create_coordinates(location_name="madrid")
    
    with patch("routes.location.get_location_service", return_value=mock_service):
        response = client.post(
            "/location/recycling-points/search",
            json={
                "latitude": coords["latitude"],
                "longitude": coords["longitude"],
                "radius": 1000
            },
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == 200
        assert response.json()["count"] == 0
