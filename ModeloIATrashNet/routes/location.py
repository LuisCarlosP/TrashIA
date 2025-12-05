"""
Rutas para funcionalidades de ubicación y puntos de reciclaje.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from models.location_models import (
    RecyclingPoint,
    RecyclingPointsRequest,
    RecyclingPointsResponse,
    Coordinates
)
from services.location_service import get_location_service, LocationService
from exceptions.location_exceptions import (
    LocationError,
    InvalidCoordinatesError,
    InvalidRadiusError
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/location", tags=["Location"])
limiter = Limiter(key_func=get_remote_address)


def validate_coordinates(latitude: float, longitude: float) -> None:
    """Valida que las coordenadas estén en rangos válidos."""
    if not (-90 <= latitude <= 90):
        raise InvalidCoordinatesError(f"Latitud {latitude} fuera de rango (-90 a 90)")
    if not (-180 <= longitude <= 180):
        raise InvalidCoordinatesError(f"Longitud {longitude} fuera de rango (-180 a 180)")


def validate_radius(radius: int) -> None:
    """Valida que el radio esté en el rango permitido."""
    if not (100 <= radius <= 50000):
        raise InvalidRadiusError(f"Radio {radius} fuera de rango (100 a 50000 metros)")


@router.get(
    "/recycling-points",
    response_model=RecyclingPointsResponse,
    summary="Buscar puntos de reciclaje cercanos",
    description="Busca puntos de reciclaje en un radio específico desde las coordenadas dadas"
)
@limiter.limit("30/minute")
async def get_recycling_points(
    request: Request,
    latitude: float = Query(..., ge=-90, le=90, description="Latitud del centro de búsqueda"),
    longitude: float = Query(..., ge=-180, le=180, description="Longitud del centro de búsqueda"),
    radius: int = Query(2000, ge=100, le=50000, description="Radio de búsqueda en metros"),
    types: Optional[str] = Query(
        None,
        description="Tipos de materiales separados por coma (plastic,glass,paper,metal,cardboard,electronics,batteries)"
    )
):
    """
    Busca puntos de reciclaje cercanos a una ubicación.
    
    - **latitude**: Latitud del centro de búsqueda (-90 a 90)
    - **longitude**: Longitud del centro de búsqueda (-180 a 180)
    - **radius**: Radio de búsqueda en metros (100 a 50000)
    - **types**: Filtrar por tipos de materiales (opcional)
    """
    try:
        validate_coordinates(latitude, longitude)
        validate_radius(radius)
        
        # Parsear tipos si se proporcionan
        types_filter = None
        if types:
            types_filter = [t.strip().lower() for t in types.split(',')]
        
        location_service = get_location_service()
        points = await location_service.get_recycling_points(
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            types_filter=types_filter
        )
        
        logger.info(f"Encontrados {len(points)} puntos de reciclaje para lat={latitude}, lon={longitude}")
        
        return RecyclingPointsResponse(
            success=True,
            count=len(points),
            radius=radius,
            center=Coordinates(latitude=latitude, longitude=longitude),
            points=points
        )
        
    except InvalidCoordinatesError as e:
        logger.warning(f"Coordenadas inválidas: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except InvalidRadiusError as e:
        logger.warning(f"Radio inválido: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except LocationError as e:
        logger.error(f"Error de ubicación: {e}")
        raise HTTPException(status_code=e.code, detail=e.message)
    except Exception as e:
        logger.error(f"Error inesperado buscando puntos de reciclaje: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error interno al buscar puntos de reciclaje"
        )


@router.post(
    "/recycling-points/search",
    response_model=RecyclingPointsResponse,
    summary="Buscar puntos de reciclaje (POST)",
    description="Alternativa POST para buscar puntos de reciclaje"
)
@limiter.limit("30/minute")
async def search_recycling_points(
    request: Request,
    search_request: RecyclingPointsRequest
):
    """
    Busca puntos de reciclaje usando un body JSON.
    Útil para aplicaciones que prefieren enviar datos por POST.
    """
    try:
        validate_coordinates(search_request.latitude, search_request.longitude)
        validate_radius(search_request.radius)
        
        location_service = get_location_service()
        points = await location_service.get_recycling_points(
            latitude=search_request.latitude,
            longitude=search_request.longitude,
            radius=search_request.radius,
            types_filter=search_request.types
        )
        
        return RecyclingPointsResponse(
            success=True,
            count=len(points),
            radius=search_request.radius,
            center=Coordinates(
                latitude=search_request.latitude,
                longitude=search_request.longitude
            ),
            points=points
        )
        
    except InvalidCoordinatesError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except InvalidRadiusError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LocationError as e:
        raise HTTPException(status_code=e.code, detail=e.message)
    except Exception as e:
        logger.error(f"Error en búsqueda POST: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error interno al buscar puntos de reciclaje"
        )


@router.get(
    "/health",
    summary="Estado del servicio de ubicación",
    description="Verifica que el servicio de ubicación esté funcionando"
)
async def location_health():
    """Endpoint de salud para el servicio de ubicación."""
    return {
        "status": "healthy",
        "service": "location",
        "message": "Servicio de ubicación funcionando correctamente"
    }
