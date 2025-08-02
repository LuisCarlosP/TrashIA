import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from typing import Tuple, Dict, Any
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import logging

from config.settings import IMAGE_WIDTH, IMAGE_HEIGHT, CLASS_NAMES, RECYCLABLE_INFO, MODEL_PATH

logger = logging.getLogger(__name__)

class ModelService:
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            logger.info(f"Modelo cargado exitosamente desde {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise RuntimeError(f"No se pudo cargar el modelo desde {MODEL_PATH}: {e}")
    
    def predict(self, img_array: np.ndarray) -> Tuple[str, float]:
        try:
            predictions = self.model.predict(img_array)
            index = np.argmax(predictions)
            class_name = CLASS_NAMES[index]
            confidence = float(tf.nn.softmax(predictions[0])[index].numpy())
            
            logger.info(f"Predicción realizada: {class_name} con confianza {confidence:.2f}")
            return class_name, confidence
            
        except Exception as e:
            logger.error(f"Error en la predicción: {e}")
            raise RuntimeError(f"Error al realizar la predicción: {e}")

class ImageProcessor:
    
    @staticmethod
    def process_image(file_bytes: bytes) -> np.ndarray:
        try:
            image = Image.open(BytesIO(file_bytes)).convert('RGB')
            image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            img_array = tf.keras.utils.img_to_array(image)
            img_array = tf.expand_dims(img_array, 0)
            img_array = preprocess_input(img_array)
            
            logger.info("Imagen procesada exitosamente")
            return img_array
            
        except Exception as e:
            logger.error(f"Error al procesar la imagen: {e}")
            raise ValueError(f"Error al procesar la imagen: {e}")

class ResponseFormatter:
    
    @staticmethod
    def format_prediction_response(class_name: str, confidence: float) -> Dict[str, Any]:
        try:
            is_recyclable, message = RECYCLABLE_INFO.get(
                class_name, 
                (False, "No hay información sobre reciclabilidad.")
            )
            
            return {
                "clase": class_name,
                "confianza": confidence,
                "es_reciclable": is_recyclable,
                "mensaje": message
            }
            
        except Exception as e:
            logger.error(f"Error al formatear respuesta: {e}")
            raise RuntimeError(f"Error al formatear la respuesta: {e}")
    
    @staticmethod
    def format_error_response(error_message: str, status_code: int = 500) -> Dict[str, Any]:
        return {
            "error": True,
            "mensaje": error_message,
            "codigo": status_code
        }
