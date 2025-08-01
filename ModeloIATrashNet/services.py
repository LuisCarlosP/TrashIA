"""
Servicios para procesamiento de imágenes y predicción del modelo.
"""
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from typing import Tuple, Dict, Any
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import logging

from config import IMAGE_WIDTH, IMAGE_HEIGHT, CLASS_NAMES, RECYCLABLE_INFO, MODEL_PATH

logger = logging.getLogger(__name__)

class ModelService:
    """Servicio para manejo del modelo de IA."""
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Carga el modelo de TensorFlow."""
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            logger.info(f"Modelo cargado exitosamente desde {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise RuntimeError(f"No se pudo cargar el modelo desde {MODEL_PATH}: {e}")
    
    def predict(self, img_array: np.ndarray) -> Tuple[str, float]:
        """
        Realiza la predicción sobre el array de imagen.
        
        Args:
            img_array: Array de imagen preprocesado
            
        Returns:
            Tuple con la clase predicha y la confianza
        """
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
    """Servicio para procesamiento de imágenes."""
    
    @staticmethod
    def process_image(file_bytes: bytes) -> np.ndarray:
        """
        Procesa los bytes de imagen para el modelo.
        
        Args:
            file_bytes: Bytes del archivo de imagen
            
        Returns:
            Array de imagen preprocesado para el modelo
        """
        try:
            # Cargar y convertir imagen
            image = Image.open(BytesIO(file_bytes)).convert('RGB')
            
            # Redimensionar
            image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            
            # Convertir a array y preprocesar
            img_array = tf.keras.utils.img_to_array(image)
            img_array = tf.expand_dims(img_array, 0)
            img_array = preprocess_input(img_array)
            
            logger.info("Imagen procesada exitosamente")
            return img_array
            
        except Exception as e:
            logger.error(f"Error al procesar la imagen: {e}")
            raise ValueError(f"Error al procesar la imagen: {e}")

class ResponseFormatter:
    """Servicio para formatear respuestas."""
    
    @staticmethod
    def format_prediction_response(class_name: str, confidence: float) -> Dict[str, Any]:
        """
        Formatea la respuesta de predicción.
        
        Args:
            class_name: Nombre de la clase predicha
            confidence: Confianza de la predicción
            
        Returns:
            Diccionario con la respuesta formateada
        """
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
        """
        Formatea respuesta de error.
        
        Args:
            error_message: Mensaje de error
            status_code: Código de estado HTTP
            
        Returns:
            Diccionario con la respuesta de error
        """
        return {
            "error": True,
            "mensaje": error_message,
            "codigo": status_code
        }
