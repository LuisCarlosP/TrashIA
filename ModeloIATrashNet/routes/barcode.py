import logging
from fastapi import APIRouter, HTTPException, Request, Query
from slowapi import Limiter
from slowapi.util import get_remote_address

from services.barcode_service import fetch_product_by_barcode

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/barcode", tags=["Barcode"])
limiter = Limiter(key_func=get_remote_address)


@router.get(
    "/{barcode}",
    summary="Buscar producto por código de barras",
    description="Consulta Open Food Facts para obtener información del producto y su reciclabilidad"
)
@limiter.limit("30/minute")
async def get_product_by_barcode(
    request: Request,
    barcode: str,
    lang: str = Query("es", description="Idioma para los consejos (es/en)")
):
    if not barcode or len(barcode) < 8:
        raise HTTPException(status_code=400, detail="Código de barras inválido")
    
    product = await fetch_product_by_barcode(barcode, lang)
    
    if not product:
        raise HTTPException(
            status_code=404,
            detail="Producto no encontrado. Intenta con otro código."
        )
    
    logger.info(f"Producto encontrado: {product.get('name')} ({barcode})")
    return product


@router.get(
    "/health",
    summary="Estado del servicio de códigos de barras"
)
async def barcode_health():
    return {
        "status": "healthy",
        "service": "barcode",
        "message": "Servicio de códigos de barras funcionando"
    }
