"""
Service to handle locations and recycling points.
Uses Overpass API to query OpenStreetMap.
"""

import logging
import math
import httpx
from typing import List, Optional, Dict, Any
from functools import lru_cache
import asyncio
from datetime import datetime, timedelta

from models.location_models import RecyclingPoint, Coordinates
import pybreaker

# Circuit Breaker configuration
# Opens after 5 failures, stays open for 60 seconds
overpass_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60)

logger = logging.getLogger(__name__)

_cache: Dict[str, tuple] = {}
CACHE_TTL = timedelta(minutes=30)

OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"


def _get_cache_key(lat: float, lon: float, radius: int) -> str:
    """Generates a cache key based on rounded coordinates."""
    return f"{round(lat, 2)}_{round(lon, 2)}_{radius}"


def _is_cache_valid(cache_entry: tuple) -> bool:
    """Checks if a cache entry is still valid."""
    if not cache_entry:
        return False
    _, timestamp = cache_entry
    return datetime.now() - timestamp < CACHE_TTL


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the distance between two points using the Haversine formula.
    
    Args:
        lat1, lon1: Coordinates of the first point
        lat2, lon2: Coordinates of the second point
        
    Returns:
        Distance in meters
    """
    R = 6371e3  # Earth radius in meters
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def _parse_recycling_types(tags: Dict[str, str]) -> List[str]:
    """
    Parses OSM tags to extract recyclable material types.
    """
    types = []
    
    type_mapping = {
        'recycling:plastic': 'plastic',
        'recycling:plastic_bottles': 'plastic',
        'recycling:plastic_packaging': 'plastic',
        'recycling:glass': 'glass',
        'recycling:glass_bottles': 'glass',
        'recycling:paper': 'paper',
        'recycling:newspaper': 'paper',
        'recycling:magazines': 'paper',
        'recycling:cardboard': 'cardboard',
        'recycling:metal': 'metal',
        'recycling:cans': 'metal',
        'recycling:aluminium': 'metal',
        'recycling:scrap_metal': 'metal',
        'recycling:batteries': 'batteries',
        'recycling:electrical_appliances': 'electronics',
        'recycling:small_electrical_appliances': 'electronics',
        'recycling:computers': 'electronics',
        'recycling:mobile_phones': 'electronics',
        'recycling:clothes': 'clothes',
        'recycling:shoes': 'clothes',
        'recycling:organic': 'organic',
        'recycling:green_waste': 'organic',
        'recycling:cooking_oil': 'oil',
        'recycling:engine_oil': 'oil',
    }
    
    for tag, material_type in type_mapping.items():
        if tags.get(tag) == 'yes' and material_type not in types:
            types.append(material_type)
    
    if not types:
        amenity = tags.get('amenity', '')
        recycling_type = tags.get('recycling_type', '')
        
        if 'container' in recycling_type:
            types = ['general']
        elif amenity == 'recycling':
            types = ['general']
    
    return types if types else ['general']


def _parse_osm_element(element: Dict[str, Any], user_lat: float, user_lon: float) -> Optional[RecyclingPoint]:
    """
    Converts an OSM element to RecyclingPoint.
    """
    try:
        tags = element.get('tags', {})
        
        if element['type'] == 'node':
            lat = element['lat']
            lon = element['lon']
        else:
            center = element.get('center', {})
            lat = center.get('lat')
            lon = center.get('lon')
            if not lat or not lon:
                return None
        
        name = tags.get('name', '')
        if not name:
            operator = tags.get('operator', '')
            recycling_type = tags.get('recycling_type', 'recycling')
            if operator:
                name = f"{operator} - {recycling_type.title()}"
            else:
                name = f"Recycling Point ({recycling_type.title()})"
        
        address_parts = []
        if tags.get('addr:street'):
            street = tags['addr:street']
            if tags.get('addr:housenumber'):
                street += f" {tags['addr:housenumber']}"
            address_parts.append(street)
        if tags.get('addr:city'):
            address_parts.append(tags['addr:city'])
        address = ', '.join(address_parts) if address_parts else None
        
        distance = calculate_distance(user_lat, user_lon, lat, lon)
        
        return RecyclingPoint(
            id=f"{element['type']}_{element['id']}",
            name=name,
            latitude=lat,
            longitude=lon,
            address=address,
            types=_parse_recycling_types(tags),
            opening_hours=tags.get('opening_hours'),
            phone=tags.get('phone') or tags.get('contact:phone'),
            website=tags.get('website') or tags.get('contact:website'),
            distance=round(distance, 2),
            operator=tags.get('operator')
        )
    except Exception as e:
        logger.warning(f"Error parsing OSM element {element.get('id')}: {e}")
        return None


async def fetch_recycling_points(
    latitude: float,
    longitude: float,
    radius: int = 2000,
    types_filter: Optional[List[str]] = None
) -> List[RecyclingPoint]:
    """
    Searches for nearby recycling points using Overpass API.
    
    Args:
        latitude: Latitude of search center
        longitude: Longitude of search center
        radius: Search radius in meters
        types_filter: Optional list of material types to filter
        
    Returns:
        List of RecyclingPoint sorted by distance
    """
    cache_key = _get_cache_key(latitude, longitude, radius)
    
    if cache_key in _cache and _is_cache_valid(_cache[cache_key]):
        logger.info(f"Cache hit for {cache_key}")
        cached_elements, _ = _cache[cache_key]
        elements = cached_elements
    else:
        query = f"""
        [out:json][timeout:25];
        (
          node["amenity"="recycling"](around:{radius},{latitude},{longitude});
          way["amenity"="recycling"](around:{radius},{latitude},{longitude});
          node["recycling_type"](around:{radius},{latitude},{longitude});
          way["recycling_type"](around:{radius},{latitude},{longitude});
        );
        out center tags;
        """
        
        # List of Overpass servers for retries
        overpass_servers = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        ]
        
        data = None
        last_error = None
        
        for server_url in overpass_servers:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    @overpass_breaker
                    async def make_request():
                        return await client.post(
                            server_url,
                            data={'data': query},
                            headers={'Content-Type': 'application/x-www-form-urlencoded'}
                        )
                    
                    response = await make_request()
                    response.raise_for_status()
                    data = response.json()
                    logger.info(f"Successful query to {server_url}")
                    break
            except httpx.TimeoutException as e:
                logger.warning(f"Timeout on {server_url}: {e}")
                last_error = e
                continue
            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error {e.response.status_code} on {server_url}: {e}")
                last_error = e
                continue
            except pybreaker.CircuitBreakerError as e:
                logger.warning(f"Circuit breaker open for {server_url}: {e}")
                last_error = e
                continue
            except Exception as e:
                logger.warning(f"Error on {server_url}: {e}")
                last_error = e
                continue
        
        if data is None:
            logger.error(f"All Overpass servers failed. Last error: {last_error}")
            raise Exception("Could not connect to map service. Please try again.")
        
        elements = data.get('elements', [])
        logger.info(f"Found {len(elements)} elements in OSM")
        
        _cache[cache_key] = (elements, datetime.now())
    
    points = []
    for element in elements:
        point = _parse_osm_element(element, latitude, longitude)
        if point:
            points.append(point)
    
    if types_filter:
        points = [
            p for p in points
            if any(t in p.types for t in types_filter) or 'general' in p.types
        ]
    
    points.sort(key=lambda p: p.distance or float('inf'))
    
    logger.info(f"Returning {len(points)} recycling points")
    return points


def clear_cache():
    """Clears the recycling points cache."""
    global _cache
    _cache = {}
    logger.info("Location cache cleared")


class LocationService:
    """Singleton service for location operations."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def get_recycling_points(
        self,
        latitude: float,
        longitude: float,
        radius: int = 2000,
        types_filter: Optional[List[str]] = None
    ) -> List[RecyclingPoint]:
        """Gets nearby recycling points."""
        return await fetch_recycling_points(latitude, longitude, radius, types_filter)
    
    def calculate_distance(
        self,
        from_coords: Coordinates,
        to_coords: Coordinates
    ) -> float:
        """Calculates the distance between two coordinates."""
        return calculate_distance(
            from_coords.latitude,
            from_coords.longitude,
            to_coords.latitude,
            to_coords.longitude
        )


def get_location_service() -> LocationService:
    """Gets the location service instance."""
    return LocationService()
