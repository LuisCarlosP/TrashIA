import pytest
from unittest.mock import patch, AsyncMock
from tests.factories import BarcodeDataFactory


def test_get_product_found(client):
    # Use factory for product data
    mock_product = BarcodeDataFactory.create_product_data(
        barcode="12345678",
        name="Test Product",
        brand="Test Brand"
    )
    
    with patch("routes.barcode.fetch_product_by_barcode", new_callable=AsyncMock, return_value=mock_product):
        response = client.get("/barcode/12345678", headers={"X-API-Key": "test-api-key"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Product"
        assert data["found"] is True


def test_get_product_not_found(client):
    with patch("routes.barcode.fetch_product_by_barcode", new_callable=AsyncMock, return_value=None):
        # Use factory-generated barcode
        barcode = BarcodeDataFactory.create_barcode()
        response = client.get(f"/barcode/{barcode}", headers={"X-API-Key": "test-api-key"})
        
        assert response.status_code == 404
        assert "Product not found" in response.json()["message"]


def test_get_product_invalid_barcode(client):
    # Use factory for invalid barcode
    invalid_barcode = BarcodeDataFactory.create_invalid_barcode()
    response = client.get(f"/barcode/{invalid_barcode}", headers={"X-API-Key": "test-api-key"})
    assert response.status_code == 400
