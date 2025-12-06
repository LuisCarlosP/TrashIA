import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from services.barcode_service import fetch_product_by_barcode, analyze_packaging

@pytest.mark.asyncio
async def test_fetch_product_openfoodfacts_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": 1,
        "product": {
            "product_name_en": "Test Product",
            "brands": "Test Brand",
            "packaging": "plastic bottle",
            "categories": "beverages"
        }
    }
    
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_response
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        with patch("services.barcode_service.off_breaker", lambda f: f):
            result = await fetch_product_by_barcode("12345678")
            
            assert result["found"] is True
            assert result["name"] == "Test Product"
            assert result["source"] == "openfoodfacts"
            assert len(result["recycling_info"]) > 0

@pytest.mark.asyncio
async def test_fetch_product_not_found():
    mock_response = MagicMock()
    mock_response.status_code = 404
    
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_response
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        with patch("services.barcode_service.off_breaker", lambda f: f):
            result = await fetch_product_by_barcode("12345678")
            assert result is None

def test_analyze_packaging():
    results = analyze_packaging("plastic bottle")
    assert len(results) > 0
    assert results[0]["bin_type"] == "yellow"
    
    results = analyze_packaging("cardboard box")
    assert len(results) > 0
    assert results[0]["bin_type"] == "blue"
    
    results = analyze_packaging("unknown material")
    assert len(results) == 0
