"""
Excepciones personalizadas para el procesamiento de imágenes.
"""


class ImageProcessingError(Exception):
    """
    Excepción lanzada cuando hay problemas al procesar una imagen.
    """
    
    def __init__(self, message: str = "Error al procesar la imagen", original_error: str = None):
        if original_error:
            message += f": {original_error}"
        super().__init__(message)
