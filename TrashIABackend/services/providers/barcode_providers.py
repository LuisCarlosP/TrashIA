import logging
from typing import Dict, Any, Optional
import httpx
import pybreaker
from config.settings import (
    CIRCUIT_BREAKER_FAIL_MAX,
    CIRCUIT_BREAKER_RESET_TIMEOUT,
    HTTP_TIMEOUT_BARCODE,
    OPEN_FOOD_FACTS_URL,
    UPCITEMDB_URL
)

logger = logging.getLogger(__name__)


class OpenFoodFactsProvider:
    def __init__(
        self,
        base_url: str = OPEN_FOOD_FACTS_URL,
        timeout: float = HTTP_TIMEOUT_BARCODE
    ):
        self._base_url = base_url
        self._timeout = timeout
        self._breaker = pybreaker.CircuitBreaker(
            fail_max=CIRCUIT_BREAKER_FAIL_MAX,
            reset_timeout=CIRCUIT_BREAKER_RESET_TIMEOUT
        )

    @property
    def name(self) -> str:
        return "openfoodfacts"

    async def fetch_product(self, barcode: str) -> Optional[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                @self._breaker
                async def make_request():
                    return await client.get(f"{self._base_url}/{barcode}.json")
                
                response = await make_request()
                if response.status_code != 200:
                    return None
                    
                data = response.json()
                if data.get("status") != 1:
                    return None
                    
                product = data.get("product", {})
                return {
                    "found": True,
                    "barcode": barcode,
                    "name": product.get("product_name_en", product.get("product_name", "Product without name")),
                    "brand": product.get("brands", ""),
                    "image_url": product.get("image_url") or product.get("image_front_url"),
                    "packaging": product.get("packaging", "") or product.get("packaging_text", ""),
                    "categories": product.get("categories", ""),
                    "source": self.name
                }
                
        except pybreaker.CircuitBreakerError:
            logger.warning(f"Circuit breaker open for {self.name} {barcode}")
        except Exception as e:
            logger.error(f"Error {self.name} {barcode}: {e}")
        
        return None


class UPCItemDBProvider:
    def __init__(
        self,
        base_url: str = UPCITEMDB_URL,
        timeout: float = HTTP_TIMEOUT_BARCODE
    ):
        self._base_url = base_url
        self._timeout = timeout
        self._breaker = pybreaker.CircuitBreaker(
            fail_max=CIRCUIT_BREAKER_FAIL_MAX,
            reset_timeout=CIRCUIT_BREAKER_RESET_TIMEOUT
        )

    @property
    def name(self) -> str:
        return "upcitemdb"

    async def fetch_product(self, barcode: str) -> Optional[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                @self._breaker
                async def make_request():
                    return await client.get(f"{self._base_url}?upc={barcode}")
                
                response = await make_request()
                response.raise_for_status()
                data = response.json()

                if data.get("code") != "OK" or data.get("total", 0) <= 0:
                    return None
                    
                item = data["items"][0]
                return {
                    "found": True,
                    "barcode": barcode,
                    "name": item.get("title", "Product without name"),
                    "brand": item.get("brand", ""),
                    "image_url": item.get("images", [None])[0],
                    "packaging": item.get("description", ""),
                    "categories": item.get("category", ""),
                    "source": self.name
                }

        except pybreaker.CircuitBreakerError:
            logger.warning(f"Circuit breaker open for {self.name} {barcode}")
        except Exception as e:
            logger.warning(f"{self.name} failed for {barcode}: {e}")

        return None
