"""
Excepciones personalizadas para operaciones de ubicación.
"""


class LocationError(Exception):
    """Error base para operaciones de ubicación."""
    
    def __init__(self, message: str, code: int = 500):
        self.message = message
        self.code = code
        super().__init__(self.message)


class OverpassAPIError(LocationError):
    """Error al consultar Overpass API."""
    
    def __init__(self, message: str = "Error al consultar el servicio de mapas"):
        super().__init__(message, 503)


class OverpassTimeoutError(LocationError):
    """Timeout al consultar Overpass API."""
    
    def __init__(self, message: str = "La búsqueda tardó demasiado. Intente con un radio menor."):
        super().__init__(message, 504)


class InvalidCoordinatesError(LocationError):
    """Coordenadas inválidas proporcionadas."""
    
    def __init__(self, message: str = "Las coordenadas proporcionadas no son válidas"):
        super().__init__(message, 400)


class InvalidRadiusError(LocationError):
    """Radio de búsqueda inválido."""
    
    def __init__(self, message: str = "El radio debe estar entre 100 y 50000 metros"):
        super().__init__(message, 400)


class NoResultsError(LocationError):
    """No se encontraron resultados."""
    
    def __init__(self, message: str = "No se encontraron puntos de reciclaje en el área"):
        super().__init__(message, 404)
