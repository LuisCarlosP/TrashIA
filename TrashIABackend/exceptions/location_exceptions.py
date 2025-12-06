"""
Custom exceptions for location operations.
"""


class LocationError(Exception):
    """Base error for location operations."""
    
    def __init__(self, message: str, code: int = 500):
        self.message = message
        self.code = code
        super().__init__(self.message)


class OverpassAPIError(LocationError):
    """Error querying Overpass API."""
    
    def __init__(self, message: str = "Error querying map service"):
        super().__init__(message, 503)


class OverpassTimeoutError(LocationError):
    """Timeout querying Overpass API."""
    
    def __init__(self, message: str = "Search took too long. Try with a smaller radius."):
        super().__init__(message, 504)


class InvalidCoordinatesError(LocationError):
    """Invalid coordinates provided."""
    
    def __init__(self, message: str = "Provided coordinates are not valid"):
        super().__init__(message, 400)


class InvalidRadiusError(LocationError):
    """Invalid search radius."""
    
    def __init__(self, message: str = "Radius must be between 100 and 50000 meters"):
        super().__init__(message, 400)


class NoResultsError(LocationError):
    """No results found."""
    
    def __init__(self, message: str = "No recycling points found in the area"):
        super().__init__(message, 404)
