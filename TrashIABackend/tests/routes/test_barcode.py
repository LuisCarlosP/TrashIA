import pytest
from unittest.mock import patch, AsyncMock

def test_get_product_found(client):
    mock_product = {
        "found": True,
        "barcode": "12345678",
        "name": "Test Product",
        "brand": "Test Brand",
        "recycling_info": [
            {
                "material": "Plastic",
                "recyclable": True,
                "bin": "Yellow Bin",
                "bin_type": "yellow"
            }
        ]
    }
    
    with patch("routes.barcode.fetch_product_by_barcode", new_callable=AsyncMock, return_value=mock_product):
        response = client.get("/barcode/12345678", headers={"X-API-Key": "test-api-key"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Product"
        assert data["found"] is True

def test_get_product_not_found(client):
    with patch("routes.barcode.fetch_product_by_barcode", new_callable=AsyncMock, return_value=None):
        response = client.get("/barcode/87654321", headers={"X-API-Key": "test-api-key"})
        
        assert response.status_code == 404
        assert "Producto no encontrado" in response.json()["detail"]

def test_get_product_invalid_barcode(client):
    response = client.get("/barcode/123", headers={"X-API-Key": "test-api-key"})
    assert response.status_code == 400
