"""
Custom exceptions for AI model operations.
"""


class ModelLoadError(Exception):
    """
    Exception raised when there are problems loading the model.
    """
    
    def __init__(self, model_path: str, original_error: str = None):
        self.model_path = model_path
        self.original_error = original_error
        message = f"Error loading model from {model_path}"
        if original_error:
            message += f": {original_error}"
        super().__init__(message)


class PredictionError(Exception):
    """
    Exception raised when there are problems during prediction.
    """
    
    def __init__(self, message: str = "Error during prediction", original_error: str = None):
        if original_error:
            message += f": {original_error}"
        super().__init__(message)
