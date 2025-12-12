import logging
from functools import lru_cache
from typing import Dict, Any, Optional
from services.trash_services import ModelService, ImageProcessor, ResponseFormatter

logger = logging.getLogger(__name__)

class PredictionService:
    
    def __init__(
        self, 
        model_service: ModelService = None,
        image_processor: ImageProcessor = None,
        response_formatter: ResponseFormatter = None
    ):
        self.image_processor = image_processor or ImageProcessor()
        self.response_formatter = response_formatter or ResponseFormatter()
        self._model_available = False
        self._model_error: Optional[str] = None
        
        # Only create ModelService if not injected
        if model_service is not None:
            self.model_service = model_service
            self._model_available = True
            logger.info("ModelService injected successfully")
        else:
            try:
                self.model_service = ModelService()
                self._model_available = True
                logger.info("ModelService initialized successfully")
            except Exception as e:
                self.model_service = None
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


@lru_cache()
def get_chat_service():
    """
    Cached ChatService instance for dependency injection.
    Uses lru_cache to ensure only one instance is created (thread-safe singleton).
    """
    from services.chat_service import ChatService
    from services.providers.gemini_provider import GeminiChatProvider
    from services.chat_session_repository import InMemoryChatSessionRepository
    
    try:
        provider = GeminiChatProvider()
        repository = InMemoryChatSessionRepository()
        return ChatService(provider, repository)
    except Exception as e:
        logger.error(f"Error initializing ChatService: {e}")
        raise RuntimeError(
            f"Chat service unavailable. Verify that GEMINI_API_KEY is configured. Error: {e}"
        )
