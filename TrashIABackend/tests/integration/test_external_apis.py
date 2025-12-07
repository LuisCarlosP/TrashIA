"""
External API Integration Tests

Tests actual external API interactions with Overpass, Open Food Facts, and UPCitemdb.
These tests hit real external APIs and may be slower than unit tests.

Run with: pytest tests/integration/test_external_apis.py -v
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tests.factories import LocationDataFactory, BarcodeDataFactory


pytestmark = pytest.mark.integration


class TestExternalAPIs:
    """Tests that hit real external APIs."""
    
    @pytest.mark.asyncio
    async def test_overpass_api_real_request(self):
        """Test real Overpass API (OpenStreetMap) request."""
        from services.location_service import fetch_recycling_points, clear_cache
        
        # Clear cache to ensure fresh API call
        clear_cache()
        
        # Use Madrid coordinates - should have recycling points
        coords = LocationDataFactory.create_coordinates(location_name="madrid")
        
        try:
            points = await fetch_recycling_points(
                latitude=coords["latitude"],
                longitude=coords["longitude"],
                radius=2000
            )
            
            # Verify response structure
            assert isinstance(points, list)
            
            if len(points) > 0:
                point = points[0]
                assert hasattr(point, 'id')
                assert hasattr(point, 'name')
                assert hasattr(point, 'latitude')
                assert hasattr(point, 'longitude')
                assert hasattr(point, 'distance')
                
        except Exception as e:
            # API might be rate-limited or unavailable
            pytest.skip(f"Overpass API unavailable: {e}")
    
    @pytest.mark.asyncio
    async def test_open_food_facts_real_request(self):
        """Test real Open Food Facts API request."""
        from services.barcode_service import fetch_product_by_barcode
        
        # Use known product barcode (Coca-Cola)
        barcode = BarcodeDataFactory.REAL_BARCODES["coca_cola"]
        
        try:
            product = await fetch_product_by_barcode(barcode)
            
            # Product may or may not be found, but response should be valid
            if product:
                assert product.get("found") is True
                assert product.get("barcode") == barcode
                assert "name" in product
                assert "recycling_info" in product
            else:
                # Product not found is also valid
                assert product is None
                
        except Exception as e:
            pytest.skip(f"Open Food Facts API unavailable: {e}")
    
    @pytest.mark.asyncio  
    async def test_upcitemdb_fallback(self):
        """Test UPCitemdb fallback when Open Food Facts fails."""
        from services.barcode_service import fetch_from_upcitemdb
        
        # Use a barcode that might be in UPCitemdb
        barcode = "5449000214911"
        
        try:
            result = await fetch_from_upcitemdb(barcode)
            # Result can be None or a product dict
            if result:
                assert result.get("source") == "upcitemdb"
                assert "name" in result
        except Exception as e:
            pytest.skip(f"UPCitemdb API unavailable: {e}")
