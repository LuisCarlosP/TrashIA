"""
Custom exceptions module for TrashIA project.
"""

from .model_exceptions import ModelLoadError, PredictionError
from .image_exceptions import ImageProcessingError
from .validation_exceptions import ValidationError

__all__ = [
    'ModelLoadError',
    'PredictionError', 
    'ImageProcessingError',
    'ValidationError'
]
