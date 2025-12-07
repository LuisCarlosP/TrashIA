class ImageProcessingError(Exception):
    
    def __init__(self, message: str = "Error processing image", original_error: str = None):
        if original_error:
            message += f": {original_error}"
        super().__init__(message)
