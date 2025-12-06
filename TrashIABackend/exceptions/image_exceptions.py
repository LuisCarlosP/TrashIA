"""
Custom exceptions for image processing.
"""


class ImageProcessingError(Exception):
    """
    Exception raised when there are problems processing an image.
    """
    
    def __init__(self, message: str = "Error processing image", original_error: str = None):
        if original_error:
            message += f": {original_error}"
        super().__init__(message)
