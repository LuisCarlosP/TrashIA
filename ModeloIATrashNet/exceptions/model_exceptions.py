"""
Excepciones personalizadas para operaciones del modelo de IA.
"""


class ModelLoadError(Exception):
    """
    Excepción lanzada cuando hay problemas al cargar el modelo.
    """
    
    def __init__(self, model_path: str, original_error: str = None):
        self.model_path = model_path
        self.original_error = original_error
        message = f"Error al cargar el modelo desde {model_path}"
        if original_error:
            message += f": {original_error}"
        super().__init__(message)


class PredictionError(Exception):
    """
    Excepción lanzada cuando hay problemas durante la predicción.
    """
    
    def __init__(self, message: str = "Error durante la predicción", original_error: str = None):
        if original_error:
            message += f": {original_error}"
        super().__init__(message)
