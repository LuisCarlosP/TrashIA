import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from typing import Tuple, Dict, Any
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import logging

from config.settings import IMAGE_WIDTH, IMAGE_HEIGHT, CLASS_NAMES, get_recyclable_info, MODEL_PATH
from exceptions import ModelLoadError, PredictionError, ImageProcessingError

logger = logging.getLogger(__name__)

class ModelService:
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ModelLoadError(MODEL_PATH, str(e))
    
    def predict(self, img_array: np.ndarray) -> Tuple[str, float]:
        try:
            predictions = self.model.predict(img_array)
            index = np.argmax(predictions)
            class_name = CLASS_NAMES[index]
            confidence = float(tf.nn.softmax(predictions[0])[index].numpy())
            
            logger.info(f"Prediction made: {class_name} with confidence {confidence:.2f}")
            return class_name, confidence
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise PredictionError("Error performing prediction", str(e))

class ImageProcessor:
    
    @staticmethod
    def process_image(file_bytes: bytes) -> np.ndarray:
        try:
            image = Image.open(BytesIO(file_bytes)).convert('RGB')
            image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            img_array = tf.keras.utils.img_to_array(image)
            img_array = tf.expand_dims(img_array, 0)
            img_array = preprocess_input(img_array)
            
            logger.info("Image processed successfully")
            return img_array
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise ImageProcessingError("Error processing image", str(e))

class ResponseFormatter:
    
    @staticmethod
    def format_prediction_response(class_name: str, confidence: float, language: str = "en") -> Dict[str, Any]:
        try:
            is_recyclable, message = get_recyclable_info(class_name, language)
            
            return {
                "class": class_name,
                "confidence": confidence,
                "is_recyclable": is_recyclable,
                "message": message
            }
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            raise RuntimeError(f"Error formatting response: {e}")
    
    @staticmethod
    def format_error_response(error_message: str, status_code: int = 500) -> Dict[str, Any]:
        return {
            "error": True,
            "message": error_message,
            "code": status_code
        }
