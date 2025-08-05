"""
Excepciones personalizadas para validación de datos.
"""


class ValidationError(Exception):
    """
    Excepción lanzada cuando hay problemas de validación de datos.
    """
    
    def __init__(self, field: str = None, message: str = "Error de validación"):
        if field:
            message = f"Error de validación en el campo '{field}': {message}"
        super().__init__(message)
