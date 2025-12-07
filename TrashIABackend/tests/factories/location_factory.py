import random
from typing import Dict, Any, List, Optional


class LocationDataFactory:
    
    LOCATIONS = {
        "madrid": {"lat": 40.4168, "lon": -3.7038},
        "barcelona": {"lat": 41.3851, "lon": 2.1734},
        "new_york": {"lat": 40.7128, "lon": -74.0060},
        "london": {"lat": 51.5074, "lon": -0.1278},
    }
    
    RECYCLING_TYPES = ['plastic', 'glass', 'paper', 'metal', 'cardboard', 'batteries', 'electronics', 'general']
    
    @staticmethod
    def create_coordinates(
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        location_name: Optional[str] = None
    ) -> Dict[str, float]:
        if location_name and location_name in LocationDataFactory.LOCATIONS:
            loc = LocationDataFactory.LOCATIONS[location_name]
            return {"latitude": loc["lat"], "longitude": loc["lon"]}
        
        return {
            "latitude": latitude if latitude is not None else round(random.uniform(-90, 90), 4),
            "longitude": longitude if longitude is not None else round(random.uniform(-180, 180), 4)
        }
    
    @staticmethod
    def create_recycling_point(
        id: Optional[str] = None,
        name: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        distance: Optional[float] = None,
        types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        return {
            "id": id or f"node_{random.randint(1000, 9999)}",
            "name": name or f"Recycling Point {random.randint(1, 100)}",
            "latitude": latitude if latitude is not None else round(random.uniform(40.0, 41.0), 4),
            "longitude": longitude if longitude is not None else round(random.uniform(-4.0, -3.0), 4),
            "address": f"Street {random.randint(1, 100)}, City",
            "types": types or random.sample(LocationDataFactory.RECYCLING_TYPES, k=random.randint(1, 3)),
            "opening_hours": "Mo-Fr 09:00-18:00",
            "phone": "+34 123 456 789",
            "website": "https://example.com/recycling",
            "distance": distance if distance is not None else round(random.uniform(100, 2000), 2),
            "operator": "City Recycling Services"
        }
    
    @staticmethod
    def create_recycling_points_batch(count: int = 5) -> List[Dict[str, Any]]:
        return [LocationDataFactory.create_recycling_point() for _ in range(count)]
