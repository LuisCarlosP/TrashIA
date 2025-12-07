class ValidationError(Exception):
    
    def __init__(self, field: str = None, message: str = "Validation error"):
        if field:
            message = f"Validation error in field '{field}': {message}"
        super().__init__(message)
