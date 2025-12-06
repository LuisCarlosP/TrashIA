"""
Routes for location functionality and recycling points.
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
    """Validates that coordinates are in valid ranges."""
    if not (-90 <= latitude <= 90):
        raise InvalidCoordinatesError(f"Latitude {latitude} out of range (-90 to 90)")
    if not (-180 <= longitude <= 180):
        raise InvalidCoordinatesError(f"Longitude {longitude} out of range (-180 to 180)")


def validate_radius(radius: int) -> None:
    """Validates that the radius is in the allowed range."""
    if not (100 <= radius <= 50000):
        raise InvalidRadiusError(f"Radius {radius} out of range (100 to 50000 meters)")


@router.get(
    "/recycling-points",
    response_model=RecyclingPointsResponse,
    summary="Search nearby recycling points",
    description="Searches for recycling points within a specific radius from given coordinates"
)
@limiter.limit("30/minute")
async def get_recycling_points(
    request: Request,
    latitude: float = Query(..., ge=-90, le=90, description="Latitude of search center"),
    longitude: float = Query(..., ge=-180, le=180, description="Longitude of search center"),
    radius: int = Query(2000, ge=100, le=50000, description="Search radius in meters"),
    types: Optional[str] = Query(
        None,
        description="Material types separated by comma (plastic,glass,paper,metal,cardboard,electronics,batteries)"
    )
):
    """
    Search for nearby recycling points to a location.
    
    - **latitude**: Latitude of search center (-90 to 90)
    - **longitude**: Longitude of search center (-180 to 180)
    - **radius**: Search radius in meters (100 to 50000)
    - **types**: Filter by material types (optional)
    """
    try:
        validate_coordinates(latitude, longitude)
        validate_radius(radius)
        
        # Parse types if provided
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
        
        logger.info(f"Found {len(points)} recycling points for lat={latitude}, lon={longitude}")
        
        return RecyclingPointsResponse(
            success=True,
            count=len(points),
            radius=radius,
            center=Coordinates(latitude=latitude, longitude=longitude),
            points=points
        )
        
    except InvalidCoordinatesError as e:
        logger.warning(f"Invalid coordinates: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except InvalidRadiusError as e:
        logger.warning(f"Invalid radius: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except LocationError as e:
        logger.error(f"Location error: {e}")
        raise HTTPException(status_code=e.code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error searching recycling points: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal error searching recycling points"
        )


@router.post(
    "/recycling-points/search",
    response_model=RecyclingPointsResponse,
    summary="Search recycling points (POST)",
    description="POST alternative to search recycling points"
)
@limiter.limit("30/minute")
async def search_recycling_points(
    request: Request,
    search_request: RecyclingPointsRequest
):
    """
    Search recycling points using a JSON body.
    Useful for applications that prefer sending data via POST.
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
        logger.error(f"Error in POST search: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal error searching recycling points"
        )


@router.get(
    "/health",
    summary="Location service status",
    description="Verifies that the location service is working"
)
async def location_health():
    """Health endpoint for the location service."""
    return {
        "status": "healthy",
        "service": "location",
        "message": "Location service working correctly"
    }
