import logging
from functools import lru_cache
from typing import Dict, Any, Optional
from services.trash_services import ModelService, ImageProcessor, ResponseFormatter

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.response_formatter = ResponseFormatter()
        self._model_available = False
        self._model_error: Optional[str] = None
        self.model_service: Optional[ModelService] = None
        
        try:
            self.model_service = ModelService()
            self._model_available = True
            logger.info("ModelService initialized successfully")
        except Exception as e:
            self._model_error = str(e)
            logger.error(f"Failed to initialize ModelService: {e}")
    
    @property
    def model_available(self) -> bool:
        """Check if the ML model is loaded and available."""
        return self._model_available and self.model_service is not None
    
    @property
    def model_error(self) -> Optional[str]:
        """Return the initialization error if model failed to load."""
        return self._model_error
    
    def check_model_health(self) -> Dict[str, Any]:
        """Return detailed health status of the ML model."""
        if self.model_available:
            return {
                "status": "healthy",
                "model_loaded": True,
                "error": None
            }
        else:
            return {
                "status": "unhealthy",
                "model_loaded": False,
                "error": self._model_error or "Model not initialized"
            }
    
    async def predict_image(self, file_bytes: bytes, filename: str = None, language: str = "en"):
        if not self.model_available:
            raise RuntimeError(f"ML model is not available: {self._model_error}")
        
        img_array = self.image_processor.process_image(file_bytes)
        class_name, confidence = self.model_service.predict(img_array)
        return self.response_formatter.format_prediction_response(class_name, confidence, language)
    
    def format_error(self, error_message: str, status_code: int = 500):
        return self.response_formatter.format_error_response(error_message, status_code)

@lru_cache()
def get_prediction_service() -> PredictionService:
    return PredictionService()
