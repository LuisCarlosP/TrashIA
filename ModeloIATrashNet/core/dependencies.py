from functools import lru_cache
from services.trash_services import ModelService, ImageProcessor, ResponseFormatter

class PredictionService:
    def __init__(self):
        self.model_service = ModelService()
        self.image_processor = ImageProcessor()
        self.response_formatter = ResponseFormatter()
    
    async def predict_image(self, file_bytes: bytes, filename: str = None):
        img_array = self.image_processor.process_image(file_bytes)
        class_name, confidence = self.model_service.predict(img_array)
        return self.response_formatter.format_prediction_response(class_name, confidence)
    
    def format_error(self, error_message: str, status_code: int = 500):
        return self.response_formatter.format_error_response(error_message, status_code)

@lru_cache()
def get_prediction_service() -> PredictionService:
    return PredictionService()
