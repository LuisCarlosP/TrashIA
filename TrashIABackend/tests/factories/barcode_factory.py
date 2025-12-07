import random
from typing import Dict, Any, Optional


class BarcodeDataFactory:
    
    REAL_BARCODES = {
        "coca_cola": "5449000000996",
        "nutella": "3017620422003",
        "oreo": "7622210449283",
    }
    
    BIN_TYPES = ["yellow", "green", "blue", "unknown"]
    
    @staticmethod
    def create_barcode(length: int = 13) -> str:
        return ''.join([str(random.randint(0, 9)) for _ in range(length)])
    
    @staticmethod
    def create_invalid_barcode() -> str:
        return "123"
    
    @staticmethod
    def create_product_data(
        barcode: Optional[str] = None,
        name: Optional[str] = None,
        brand: Optional[str] = None,
        found: bool = True
    ) -> Dict[str, Any]:
        barcode = barcode or BarcodeDataFactory.create_barcode()
        
        return {
            "found": found,
            "barcode": barcode,
            "name": name or f"Test Product {random.randint(1, 100)}",
            "brand": brand or "Test Brand",
            "image_url": "https://example.com/product.jpg",
            "packaging": "plastic bottle",
            "categories": "beverages",
            "source": "openfoodfacts",
            "recycling_info": [
                {
                    "material": "Plastic",
                    "recyclable": True,
                    "bin": "Yellow Bin",
                    "tip": "Clean and rinse before recycling",
                    "bin_type": "yellow"
                }
            ]
        }
