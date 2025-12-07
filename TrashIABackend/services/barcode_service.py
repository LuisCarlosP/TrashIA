import logging
from typing import Optional, Dict, Any
import httpx
import pybreaker
from config.settings import (
    CIRCUIT_BREAKER_FAIL_MAX,
    CIRCUIT_BREAKER_RESET_TIMEOUT,
    HTTP_TIMEOUT_BARCODE,
    OPEN_FOOD_FACTS_URL,
    UPCITEMDB_URL
)

# Circuit Breakers
off_breaker = pybreaker.CircuitBreaker(
    fail_max=CIRCUIT_BREAKER_FAIL_MAX,
    reset_timeout=CIRCUIT_BREAKER_RESET_TIMEOUT
)
upc_breaker = pybreaker.CircuitBreaker(
    fail_max=CIRCUIT_BREAKER_FAIL_MAX,
    reset_timeout=CIRCUIT_BREAKER_RESET_TIMEOUT
)

logger = logging.getLogger(__name__)

PACKAGING_RECYCLABILITY = {
    "plastic": {"recyclable": True, "bin_type": "yellow"},
    "plastico": {"recyclable": True, "bin_type": "yellow"},
    "pet": {"recyclable": True, "bin_type": "yellow"},
    "hdpe": {"recyclable": True, "bin_type": "yellow"},
    "bottle": {"recyclable": True, "bin_type": "yellow"},
    "botella": {"recyclable": True, "bin_type": "yellow"},
    "bag": {"recyclable": True, "bin_type": "yellow"},
    "bolsa": {"recyclable": True, "bin_type": "yellow"},
    "wrapper": {"recyclable": True, "bin_type": "yellow"},
    "envoltorio": {"recyclable": True, "bin_type": "yellow"},
    "pouch": {"recyclable": True, "bin_type": "yellow"},
    "glass": {"recyclable": True, "bin_type": "green"},
    "vidrio": {"recyclable": True, "bin_type": "green"},
    "jar": {"recyclable": True, "bin_type": "green"},
    "frasco": {"recyclable": True, "bin_type": "green"},
    "cardboard": {"recyclable": True, "bin_type": "blue"},
    "carton": {"recyclable": True, "bin_type": "blue"},
    "paper": {"recyclable": True, "bin_type": "blue"},
    "papel": {"recyclable": True, "bin_type": "blue"},
    "box": {"recyclable": True, "bin_type": "blue"},
    "caja": {"recyclable": True, "bin_type": "blue"},
    "metal": {"recyclable": True, "bin_type": "yellow"},
    "aluminium": {"recyclable": True, "bin_type": "yellow"},
    "aluminio": {"recyclable": True, "bin_type": "yellow"},
    "can": {"recyclable": True, "bin_type": "yellow"},
    "lata": {"recyclable": True, "bin_type": "yellow"},
    "tin": {"recyclable": True, "bin_type": "yellow"},
    "tetra": {"recyclable": True, "bin_type": "yellow"},
    "brik": {"recyclable": True, "bin_type": "yellow"},
}

BIN_INFO = {
    "yellow": {
        "bin": "Yellow Bin",
        "tip": "Clean/Rinse the container. Plastics, cans and briks."
    },
    "green": {
        "bin": "Green Bin",
        "tip": "Remove caps and corks. Glass only."
    },
    "blue": {
        "bin": "Blue Bin",
        "tip": "Flatten to save space. Paper and cardboard."
    },
    "unknown": {
        "bin": "Check Packaging",
        "tip": "Material not detected automatically. Check the packaging."
    }
}

def analyze_packaging(text_to_analyze: str) -> list:
    if not text_to_analyze:
        return []

    text_lower = text_to_analyze.lower()
    detected_bins = set()
    results = []

    for material_key, info in PACKAGING_RECYCLABILITY.items():
        if material_key in text_lower:
            detected_bins.add(info["bin_type"])

    for bin_type in detected_bins:
        bin_data = BIN_INFO.get(bin_type, BIN_INFO["unknown"])

        results.append({
            "material": "Recyclable Material",
            "recyclable": True,
            "bin": bin_data["bin"],
            "tip": bin_data["tip"],
            "bin_type": bin_type
        })

    return results

async def fetch_from_upcitemdb(barcode: str) -> Optional[Dict[str, Any]]:
    """Queries the trial API of UPCitemdb."""
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_BARCODE) as client:
            @upc_breaker
            async def make_upc_request():
                return await client.get(f"{UPCITEMDB_URL}?upc={barcode}")
            
            response = await make_upc_request()
            response.raise_for_status()
            data = response.json()

            if data.get("code") == "OK" and data.get("total", 0) > 0:
                item = data["items"][0]
                return {
                    "found": True,
                    "barcode": barcode,
                    "name": item.get("title", "Product without name"),
                    "brand": item.get("brand", ""),
                    "image_url": item.get("images", [None])[0],
                    "packaging": item.get("description", ""),
                    "categories": item.get("category", ""),
                    "source": "upcitemdb"
                }

    except pybreaker.CircuitBreakerError:
        logger.warning(f"Circuit breaker open for UPCitemdb {barcode}")
    except Exception as e:
        logger.warning(f"UPCitemdb failed for {barcode}: {e}")

    return None

async def fetch_product_by_barcode(barcode: str) -> Optional[Dict[str, Any]]:
    product_data = None

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_BARCODE) as client:
            @off_breaker
            async def make_off_request():
                return await client.get(f"{OPEN_FOOD_FACTS_URL}/{barcode}.json")
            
            response = await make_off_request()
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == 1:
                    product = data.get("product", {})
                    product_data = {
                        "found": True,
                        "barcode": barcode,
                        "name": product.get("product_name_en", product.get("product_name", "Product without name")),
                        "brand": product.get("brands", ""),
                        "image_url": product.get("image_url") or product.get("image_front_url"),
                        "packaging": product.get("packaging", "") or product.get("packaging_text", ""),
                        "categories": product.get("categories", ""),
                        "source": "openfoodfacts"
                    }

    except pybreaker.CircuitBreakerError:
        logger.warning(f"Circuit breaker open for OpenFoodFacts {barcode}")
    except Exception as e:
        logger.error(f"Error OpenFoodFacts {barcode}: {e}")

    if not product_data:
        logger.info(f"Product {barcode} not found in OFF, trying UPCitemdb...")
        product_data = await fetch_from_upcitemdb(barcode)

    if product_data:
        analysis_text = f"{product_data.get('packaging', '')} {product_data.get('categories', '')}"
        recycling_info = analyze_packaging(analysis_text)

        if not recycling_info:
             default_bin = BIN_INFO["unknown"]
             recycling_info = [{
                "material": "Unknown",
                "recyclable": None,
                "bin": None,
                "tip": default_bin["tip"],
                "bin_type": "unknown"
            }]

        product_data["recycling_info"] = recycling_info
        return product_data

    logger.info(f"Product not found in any database: {barcode}")
    return None
