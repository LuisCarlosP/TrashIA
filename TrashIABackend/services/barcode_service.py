import logging
from typing import List, Dict, Any, Optional
import pybreaker
from config.settings import CIRCUIT_BREAKER_FAIL_MAX, CIRCUIT_BREAKER_RESET_TIMEOUT
from core.protocols.barcode import BarcodeProviderProtocol

logger = logging.getLogger(__name__)

off_breaker = pybreaker.CircuitBreaker(
    fail_max=CIRCUIT_BREAKER_FAIL_MAX,
    reset_timeout=CIRCUIT_BREAKER_RESET_TIMEOUT
)
upc_breaker = pybreaker.CircuitBreaker(
    fail_max=CIRCUIT_BREAKER_FAIL_MAX,
    reset_timeout=CIRCUIT_BREAKER_RESET_TIMEOUT
)

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


class PackagingAnalyzer:
    def analyze(self, text_to_analyze: str) -> List[Dict[str, Any]]:
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

    def get_default_info(self) -> List[Dict[str, Any]]:
        default_bin = BIN_INFO["unknown"]
        return [{
            "material": "Unknown",
            "recyclable": None,
            "bin": None,
            "tip": default_bin["tip"],
            "bin_type": "unknown"
        }]


class BarcodeService:
    def __init__(
        self,
        providers: List[BarcodeProviderProtocol],
        analyzer: PackagingAnalyzer = None
    ):
        self._providers = providers
        self._analyzer = analyzer or PackagingAnalyzer()
        logger.info(f"BarcodeService initialized with {len(providers)} providers")

    async def fetch_product(self, barcode: str) -> Optional[Dict[str, Any]]:
        product_data = None
        
        for provider in self._providers:
            product_data = await provider.fetch_product(barcode)
            if product_data:
                logger.info(f"Product {barcode} found in {provider.name}")
                break
        
        if not product_data:
            logger.info(f"Product not found in any database: {barcode}")
            return None
        
        analysis_text = f"{product_data.get('packaging', '')} {product_data.get('categories', '')}"
        recycling_info = self._analyzer.analyze(analysis_text)
        
        if not recycling_info:
            recycling_info = self._analyzer.get_default_info()
        
        product_data["recycling_info"] = recycling_info
        return product_data


async def fetch_product_by_barcode(barcode: str) -> Optional[Dict[str, Any]]:
    from services.providers.barcode_providers import OpenFoodFactsProvider, UPCItemDBProvider
    
    providers = [OpenFoodFactsProvider(), UPCItemDBProvider()]
    service = BarcodeService(providers)
    return await service.fetch_product(barcode)


async def fetch_from_upcitemdb(barcode: str) -> Optional[Dict[str, Any]]:
    from services.providers.barcode_providers import UPCItemDBProvider
    
    provider = UPCItemDBProvider()
    return await provider.fetch_product(barcode)


def analyze_packaging(text_to_analyze: str) -> List[Dict[str, Any]]:
    analyzer = PackagingAnalyzer()
    return analyzer.analyze(text_to_analyze)
