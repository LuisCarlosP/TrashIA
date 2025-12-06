import logging
from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from services.barcode_service import fetch_product_by_barcode

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/barcode", tags=["Barcode"])
limiter = Limiter(key_func=get_remote_address)


@router.get(
    "/{barcode}",
    summary="Search product by barcode",
    description="Queries Open Food Facts to get product information and recyclability"
)
@limiter.limit("30/minute")
async def get_product_by_barcode(
    request: Request,
    barcode: str
):
    if not barcode or len(barcode) < 8:
        raise HTTPException(status_code=400, detail="Invalid barcode")
    
    product = await fetch_product_by_barcode(barcode)
    
    if not product:
        raise HTTPException(
            status_code=404,
            detail="Product not found. Try another code."
        )
    
    logger.info(f"Product found: {product.get('name')} ({barcode})")
    return product


@router.get(
    "/health",
    summary="Barcode service status"
)
async def barcode_health():
    return {
        "status": "healthy",
        "service": "barcode",
        "message": "Barcode service working"
    }
