import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any, Optional

from services.barcode_service import (
    BarcodeService,
    PackagingAnalyzer,
    fetch_product_by_barcode,
    analyze_packaging
)


class MockBarcodeProvider:
    """Mock implementation of BarcodeProviderProtocol for testing."""
    
    def __init__(self, product_data: Optional[Dict[str, Any]] = None):
        self._product_data = product_data
        self._name = "mock_provider"
    
    @property
    def name(self) -> str:
        return self._name
    
    async def fetch_product(self, barcode: str) -> Optional[Dict[str, Any]]:
        return self._product_data


@pytest.mark.asyncio
async def test_barcode_service_with_mock_provider():
    """Test BarcodeService with an injected mock provider."""
    mock_product = {
        "found": True,
        "barcode": "12345678",
        "name": "Test Product",
        "brand": "Test Brand",
        "packaging": "plastic bottle",
        "categories": "beverages",
        "source": "mock_provider"
    }
    
    mock_provider = MockBarcodeProvider(mock_product)
    service = BarcodeService(providers=[mock_provider])
    
    result = await service.fetch_product("12345678")
    
    assert result["found"] is True
    assert result["name"] == "Test Product"
    assert len(result["recycling_info"]) > 0


@pytest.mark.asyncio
async def test_barcode_service_product_not_found():
    """Test BarcodeService when no provider finds the product."""
    mock_provider = MockBarcodeProvider(None)
    service = BarcodeService(providers=[mock_provider])
    
    result = await service.fetch_product("12345678")
    assert result is None


@pytest.mark.asyncio
async def test_barcode_service_fallback_providers():
    """Test that BarcodeService tries multiple providers."""
    mock_provider1 = MockBarcodeProvider(None)  # First fails
    mock_product = {
        "found": True,
        "barcode": "12345678",
        "name": "Fallback Product",
        "brand": "Brand",
        "packaging": "glass bottle",
        "categories": "drinks",
        "source": "fallback"
    }
    mock_provider2 = MockBarcodeProvider(mock_product)  # Second succeeds
    
    service = BarcodeService(providers=[mock_provider1, mock_provider2])
    result = await service.fetch_product("12345678")
    
    assert result is not None
    assert result["name"] == "Fallback Product"


@pytest.mark.asyncio
async def test_fetch_product_by_barcode_wrapper():
    """Test the convenience wrapper function."""
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
        result = await fetch_product_by_barcode("12345678")
        
        assert result["found"] is True
        assert result["name"] == "Test Product"
        assert result["source"] == "openfoodfacts"


def test_analyze_packaging():
    """Test the PackagingAnalyzer."""
    results = analyze_packaging("plastic bottle")
    assert len(results) > 0
    assert results[0]["bin_type"] == "yellow"
    
    results = analyze_packaging("cardboard box")
    assert len(results) > 0
    assert results[0]["bin_type"] == "blue"
    
    results = analyze_packaging("unknown material")
    assert len(results) == 0


def test_packaging_analyzer_class():
    """Test PackagingAnalyzer class directly."""
    analyzer = PackagingAnalyzer()
    
    # Test with glass
    results = analyzer.analyze("glass jar")
    assert any(r["bin_type"] == "green" for r in results)
    
    # Test with metal
    results = analyzer.analyze("aluminum can")
    assert any(r["bin_type"] == "yellow" for r in results)
    
    # Test default info
    default = analyzer.get_default_info()
    assert len(default) == 1
    assert default[0]["bin_type"] == "unknown"
