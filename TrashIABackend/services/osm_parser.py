import math
from typing import Dict, Any, List, Optional
from models.location_models import RecyclingPoint


class OSMParser:
    RECYCLING_TAG_PREFIXES = [
        "recycling:",
        "waste:",
        "recycling_type:",
    ]

    def calculate_distance(
        self, 
        lat1: float, 
        lon1: float, 
        lat2: float, 
        lon2: float
    ) -> float:
        R = 6371000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = (
            math.sin(delta_phi / 2) ** 2 +
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

    def parse_recycling_types(self, tags: Dict[str, str]) -> List[str]:
        types = []
        
        for key, value in tags.items():
            for prefix in self.RECYCLING_TAG_PREFIXES:
                if key.startswith(prefix) and value == "yes":
                    material = key.replace(prefix, "")
                    types.append(material)
        
        if not types:
            amenity = tags.get("amenity", "")
            if "recycling" in amenity:
                types.append("general")
        
        return types

    def parse_osm_element(
        self,
        element: Dict[str, Any],
        user_lat: float,
        user_lon: float
    ) -> Optional[RecyclingPoint]:
        tags = element.get("tags", {})
        
        if element["type"] == "node":
            lat = element.get("lat")
            lon = element.get("lon")
        else:
            center = element.get("center", {})
            lat = center.get("lat")
            lon = center.get("lon")
        
        if lat is None or lon is None:
            return None
        
        distance = self.calculate_distance(user_lat, user_lon, lat, lon)
        recycling_types = self.parse_recycling_types(tags)
        
        return RecyclingPoint(
            id=str(element.get("id", "")),
            name=tags.get("name", tags.get("operator", "Recycling Point")),
            lat=lat,
            lon=lon,
            distance_meters=round(distance, 1),
            types=recycling_types,
            address=tags.get("addr:street", ""),
            opening_hours=tags.get("opening_hours"),
            operator=tags.get("operator"),
            website=tags.get("website"),
            phone=tags.get("phone"),
        )

    def parse_response(
        self,
        elements: List[Dict[str, Any]],
        user_lat: float,
        user_lon: float,
        types_filter: Optional[List[str]] = None
    ) -> List[RecyclingPoint]:
        points = []
        
        for element in elements:
            point = self.parse_osm_element(element, user_lat, user_lon)
            if point is None:
                continue
            
            if types_filter:
                if any(t in point.types for t in types_filter):
                    points.append(point)
            else:
                points.append(point)
        
        return sorted(points, key=lambda p: p.distance_meters)
