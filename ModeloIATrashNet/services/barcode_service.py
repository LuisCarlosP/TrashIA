import logging
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger(__name__)

OPEN_FOOD_FACTS_URL = "https://world.openfoodfacts.org/api/v2/product"
UPCITEMDB_URL = "https://api.upcitemdb.com/prod/trial/lookup"

# Definición de materiales y sus propiedades de reciclaje
PACKAGING_RECYCLABILITY = {
    "plastic": {"recyclable": True, "bin_type": "yellow"},
    "plastico": {"recyclable": True, "bin_type": "yellow"},
    "glass": {"recyclable": True, "bin_type": "green"},
    "vidrio": {"recyclable": True, "bin_type": "green"},
    "cardboard": {"recyclable": True, "bin_type": "blue"},
    "carton": {"recyclable": True, "bin_type": "blue"},
    "paper": {"recyclable": True, "bin_type": "blue"},
    "papel": {"recyclable": True, "bin_type": "blue"},
    "metal": {"recyclable": True, "bin_type": "yellow"},
    "aluminium": {"recyclable": True, "bin_type": "yellow"},
    "aluminio": {"recyclable": True, "bin_type": "yellow"},
    "tetra": {"recyclable": True, "bin_type": "yellow"},
    "can": {"recyclable": True, "bin_type": "yellow"},
    "lata": {"recyclable": True, "bin_type": "yellow"},
    "bottle": {"recyclable": True, "bin_type": "yellow"}, # Asumimos plástico por defecto si no especifica vidrio
    "botella": {"recyclable": True, "bin_type": "yellow"},
    "box": {"recyclable": True, "bin_type": "blue"},
    "caja": {"recyclable": True, "bin_type": "blue"},
}

# Traducciones de contenedores y consejos
BIN_INFO = {
    "yellow": {
        "es": "Contenedor Amarillo",
        "en": "Yellow Bin",
        "tip_es": "Limpia/Enjuaga el envase. Deposita en el contenedor amarillo.",
        "tip_en": "Clean/Rinse the container. Place in the yellow bin."
    },
    "green": {
        "es": "Contenedor Verde",
        "en": "Green Bin",
        "tip_es": "Retira tapas y corchos. Solo vidrio.",
        "tip_en": "Remove caps and corks. Glass only."
    },
    "blue": {
        "es": "Contenedor Azul",
        "en": "Blue Bin",
        "tip_es": "Aplana para ahorrar espacio. Papel y cartón.",
        "tip_en": "Flatten to save space. Paper and cardboard."
    },
    "unknown": {
        "es": "Consultar Empaque",
        "en": "Check Packaging",
        "tip_es": "Revisa el empaque para instrucciones específicas.",
        "tip_en": "Check the packaging for specific instructions."
    }
}

def analyze_packaging(packaging_text: str, lang: str = "es") -> list:
    if not packaging_text:
        return []
    
    packaging_lower = packaging_text.lower()
    detected_bins = set()
    results = []
    
    # Buscar materiales en el texto
    for material_key, info in PACKAGING_RECYCLABILITY.items():
        if material_key in packaging_lower:
            detected_bins.add(info["bin_type"])

    # Generar resultados únicos por tipo de contenedor para evitar redundancia
    # (Ej: no mostrar "Lata -> Amarillo" y "Aluminio -> Amarillo" por separado)
    for bin_type in detected_bins:
        bin_data = BIN_INFO.get(bin_type, BIN_INFO["unknown"])
        
        results.append({
            "material": "Material Reciclable" if lang == "es" else "Recyclable Material", # Nombre genérico ya que agrupamos por contenedor
            "recyclable": True,
            "bin": bin_data["es"] if lang == "es" else bin_data["en"],
            "tip": bin_data["tip_es"] if lang == "es" else bin_data["tip_en"]
        })
    
    return results

async def fetch_from_upcitemdb(barcode: str) -> Optional[Dict[str, Any]]:
    """Consulta la API de prueba de UPCitemdb."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{UPCITEMDB_URL}?upc={barcode}")
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") == "OK" and data.get("total", 0) > 0:
                item = data["items"][0]
                return {
                    "found": True,
                    "barcode": barcode,
                    "name": item.get("title", "Producto sin nombre"),
                    "brand": item.get("brand", ""),
                    "image_url": item.get("images", [None])[0],
                    "packaging": item.get("description", ""), # UPCitemdb no tiene campo packaging específico, usamos descripción
                    "categories": item.get("category", ""),
                    "source": "upcitemdb"
                }
    except Exception as e:
        logger.warning(f"UPCitemdb falló para {barcode}: {e}")
    
    return None

async def fetch_product_by_barcode(barcode: str, lang: str = "es") -> Optional[Dict[str, Any]]:
    product_data = None
    
    # 1. Intentar OpenFoodFacts
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OPEN_FOOD_FACTS_URL}/{barcode}.json")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == 1:
                    product = data.get("product", {})
                    product_data = {
                        "found": True,
                        "barcode": barcode,
                        "name": product.get(f"product_name_{lang}", product.get("product_name", "Producto sin nombre")),
                        "brand": product.get("brands", ""),
                        "image_url": product.get("image_url") or product.get("image_front_url"),
                        "packaging": product.get("packaging", "") or product.get("packaging_text", ""),
                        "categories": product.get("categories", ""),
                        "source": "openfoodfacts"
                    }
    except Exception as e:
        logger.error(f"Error OpenFoodFacts {barcode}: {e}")

    # 2. Si no se encuentra, intentar UPCitemdb
    if not product_data:
        logger.info(f"Producto {barcode} no encontrado en OFF, intentando UPCitemdb...")
        product_data = await fetch_from_upcitemdb(barcode)

    # 3. Procesar resultados si encontramos algo
    if product_data:
        recycling_info = analyze_packaging(product_data.get("packaging", ""), lang)
        
        # Si no hay info de reciclaje detectada pero tenemos el producto, dar consejo genérico
        if not recycling_info:
             default_bin = BIN_INFO["unknown"]
             recycling_info = [{
                "material": "unknown",
                "recyclable": None,
                "bin": default_bin["es"] if lang == "es" else default_bin["en"],
                "tip": default_bin["tip_es"] if lang == "es" else default_bin["tip_en"]
            }]
        
        product_data["recycling_info"] = recycling_info
        return product_data

    logger.info(f"Producto no encontrado en ninguna base de datos: {barcode}")
    return None
