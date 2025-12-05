import logging
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger(__name__)

OPEN_FOOD_FACTS_URL = "https://world.openfoodfacts.org/api/v2/product"

PACKAGING_RECYCLABILITY = {
    "plastic": {
        "recyclable": True,
        "bin": "amarillo",
        "tip_es": "Enjuaga el envase antes de reciclarlo. Deposita en el contenedor amarillo.",
        "tip_en": "Rinse the container before recycling. Place in the yellow bin."
    },
    "glass": {
        "recyclable": True,
        "bin": "verde",
        "tip_es": "Retira tapas y corchos. Deposita solo el vidrio en el contenedor verde.",
        "tip_en": "Remove caps and corks. Place only glass in the green bin."
    },
    "cardboard": {
        "recyclable": True,
        "bin": "azul",
        "tip_es": "Aplana las cajas para ahorrar espacio. Deposita en el contenedor azul.",
        "tip_en": "Flatten boxes to save space. Place in the blue bin."
    },
    "paper": {
        "recyclable": True,
        "bin": "azul",
        "tip_es": "Asegúrate de que esté limpio y seco. Deposita en el contenedor azul.",
        "tip_en": "Make sure it's clean and dry. Place in the blue bin."
    },
    "metal": {
        "recyclable": True,
        "bin": "amarillo",
        "tip_es": "Limpia las latas antes de reciclar. Deposita en el contenedor amarillo.",
        "tip_en": "Clean cans before recycling. Place in the yellow bin."
    },
    "aluminium": {
        "recyclable": True,
        "bin": "amarillo",
        "tip_es": "El aluminio es 100% reciclable. Deposita en el contenedor amarillo.",
        "tip_en": "Aluminum is 100% recyclable. Place in the yellow bin."
    },
    "tetra": {
        "recyclable": True,
        "bin": "amarillo",
        "tip_es": "Aplana el envase y deposita en el contenedor amarillo.",
        "tip_en": "Flatten the container and place in the yellow bin."
    },
    "can": {
        "recyclable": True,
        "bin": "amarillo",
        "tip_es": "Enjuaga la lata antes de reciclar. Deposita en el contenedor amarillo.",
        "tip_en": "Rinse the can before recycling. Place in the yellow bin."
    }
}


def analyze_packaging(packaging_text: str, lang: str = "es") -> list:
    if not packaging_text:
        return []
    
    packaging_lower = packaging_text.lower()
    results = []
    
    for material, info in PACKAGING_RECYCLABILITY.items():
        if material in packaging_lower:
            tip = info["tip_es"] if lang == "es" else info["tip_en"]
            results.append({
                "material": material,
                "recyclable": info["recyclable"],
                "bin": info["bin"],
                "tip": tip
            })
    
    return results


async def fetch_product_by_barcode(barcode: str, lang: str = "es") -> Optional[Dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{OPEN_FOOD_FACTS_URL}/{barcode}.json")
            response.raise_for_status()
            data = response.json()
        
        if data.get("status") != 1:
            logger.info(f"Producto no encontrado para código: {barcode}")
            return None
        
        product = data.get("product", {})
        
        packaging = product.get("packaging", "") or product.get("packaging_text", "")
        recycling_info = analyze_packaging(packaging, lang)
        
        if not recycling_info and packaging:
            default_tip = "Revisa el empaque para instrucciones de reciclaje." if lang == "es" else "Check the packaging for recycling instructions."
            recycling_info = [{
                "material": "unknown",
                "recyclable": None,
                "bin": None,
                "tip": default_tip
            }]
        
        return {
            "found": True,
            "barcode": barcode,
            "name": product.get("product_name") or product.get("product_name_es") or "Producto sin nombre",
            "brand": product.get("brands", ""),
            "image_url": product.get("image_url") or product.get("image_front_url"),
            "packaging": packaging,
            "categories": product.get("categories", ""),
            "recycling_info": recycling_info
        }
        
    except httpx.TimeoutException:
        logger.error(f"Timeout al consultar Open Food Facts para {barcode}")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"Error HTTP {e.response.status_code} para {barcode}")
        return None
    except Exception as e:
        logger.error(f"Error consultando producto {barcode}: {e}")
        return None
