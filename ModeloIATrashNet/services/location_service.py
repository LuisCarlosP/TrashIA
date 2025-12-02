"""
Servicio para manejar ubicaciones y puntos de reciclaje.
Utiliza Overpass API para consultar OpenStreetMap.
"""

import logging
import math
import httpx
from typing import List, Optional, Dict, Any
from functools import lru_cache
import asyncio
from datetime import datetime, timedelta

from models.location_models import RecyclingPoint, Coordinates

logger = logging.getLogger(__name__)

# Cache simple en memoria
_cache: Dict[str, tuple] = {}
CACHE_TTL = timedelta(minutes=30)

OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"


def _get_cache_key(lat: float, lon: float, radius: int) -> str:
    """Genera una clave de cache basada en coordenadas redondeadas."""
    # Redondear a 2 decimales para agrupar búsquedas cercanas
    return f"{round(lat, 2)}_{round(lon, 2)}_{radius}"


def _is_cache_valid(cache_entry: tuple) -> bool:
    """Verifica si una entrada de cache sigue siendo válida."""
    if not cache_entry:
        return False
    _, timestamp = cache_entry
    return datetime.now() - timestamp < CACHE_TTL


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula la distancia entre dos puntos usando la fórmula de Haversine.
    
    Args:
        lat1, lon1: Coordenadas del primer punto
        lat2, lon2: Coordenadas del segundo punto
        
    Returns:
        Distancia en metros
    """
    R = 6371e3  # Radio de la Tierra en metros
    
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
    Parsea los tags de OSM para extraer los tipos de materiales reciclables.
    """
    types = []
    
    # Mapeo de tags OSM a tipos normalizados
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
    
    # Si no hay tipos específicos, inferir del tipo de amenidad
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
    Convierte un elemento de OSM a RecyclingPoint.
    """
    try:
        tags = element.get('tags', {})
        
        # Obtener coordenadas
        if element['type'] == 'node':
            lat = element['lat']
            lon = element['lon']
        else:
            # Para ways y relations, usar el centro
            center = element.get('center', {})
            lat = center.get('lat')
            lon = center.get('lon')
            if not lat or not lon:
                return None
        
        # Construir nombre
        name = tags.get('name', '')
        if not name:
            operator = tags.get('operator', '')
            recycling_type = tags.get('recycling_type', 'recycling')
            if operator:
                name = f"{operator} - {recycling_type.title()}"
            else:
                name = f"Punto de Reciclaje ({recycling_type.title()})"
        
        # Construir dirección
        address_parts = []
        if tags.get('addr:street'):
            street = tags['addr:street']
            if tags.get('addr:housenumber'):
                street += f" {tags['addr:housenumber']}"
            address_parts.append(street)
        if tags.get('addr:city'):
            address_parts.append(tags['addr:city'])
        address = ', '.join(address_parts) if address_parts else None
        
        # Calcular distancia
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
    radius: int = 5000,
    types_filter: Optional[List[str]] = None
) -> List[RecyclingPoint]:
    """
    Busca puntos de reciclaje cercanos usando Overpass API.
    
    Args:
        latitude: Latitud del centro de búsqueda
        longitude: Longitud del centro de búsqueda
        radius: Radio de búsqueda en metros
        types_filter: Lista opcional de tipos de materiales para filtrar
        
    Returns:
        Lista de RecyclingPoint ordenados por distancia
    """
    cache_key = _get_cache_key(latitude, longitude, radius)
    
    # Verificar cache
    if cache_key in _cache and _is_cache_valid(_cache[cache_key]):
        logger.info(f"Cache hit for {cache_key}")
        cached_points, _ = _cache[cache_key]
        points = cached_points
    else:
        # Construir query Overpass
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
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    OVERPASS_API_URL,
                    data={'data': query},
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException:
            logger.error("Timeout al consultar Overpass API")
            raise Exception("La búsqueda tardó demasiado. Intente con un radio menor.")
        except httpx.HTTPStatusError as e:
            logger.error(f"Error HTTP de Overpass API: {e}")
            raise Exception("Error al consultar el servicio de mapas.")
        except Exception as e:
            logger.error(f"Error al consultar Overpass API: {e}")
            raise Exception("No se pudo obtener información de puntos de reciclaje.")
        
        elements = data.get('elements', [])
        logger.info(f"Encontrados {len(elements)} elementos en OSM")
        
        # Parsear elementos
        points = []
        for element in elements:
            point = _parse_osm_element(element, latitude, longitude)
            if point:
                points.append(point)
        
        # Guardar en cache
        _cache[cache_key] = (points, datetime.now())
    
    # Aplicar filtro de tipos si se especifica
    if types_filter:
        points = [
            p for p in points
            if any(t in p.types for t in types_filter) or 'general' in p.types
        ]
    
    # Ordenar por distancia
    points.sort(key=lambda p: p.distance or float('inf'))
    
    logger.info(f"Retornando {len(points)} puntos de reciclaje")
    return points


def clear_cache():
    """Limpia la cache de puntos de reciclaje."""
    global _cache
    _cache = {}
    logger.info("Cache de ubicaciones limpiada")


class LocationService:
    """Servicio singleton para operaciones de ubicación."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def get_recycling_points(
        self,
        latitude: float,
        longitude: float,
        radius: int = 5000,
        types_filter: Optional[List[str]] = None
    ) -> List[RecyclingPoint]:
        """Obtiene puntos de reciclaje cercanos."""
        return await fetch_recycling_points(latitude, longitude, radius, types_filter)
    
    def calculate_distance(
        self,
        from_coords: Coordinates,
        to_coords: Coordinates
    ) -> float:
        """Calcula la distancia entre dos coordenadas."""
        return calculate_distance(
            from_coords.latitude,
            from_coords.longitude,
            to_coords.latitude,
            to_coords.longitude
        )


def get_location_service() -> LocationService:
    """Obtiene la instancia del servicio de ubicación."""
    return LocationService()
