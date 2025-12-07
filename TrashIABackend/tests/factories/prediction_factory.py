import random
from typing import Dict, Any, Optional
from io import BytesIO
from PIL import Image

from config.settings import ALLOWED_MIME_TYPES, MAX_FILE_SIZE


class PredictionDataFactory:
    
    MATERIALS = ['plastic', 'glass', 'metal', 'paper', 'cardboard', 'trash']
    ALLOWED_MIME_TYPES = ALLOWED_MIME_TYPES
    MAX_FILE_SIZE = MAX_FILE_SIZE
    
    @staticmethod
    def create_image_bytes(
        width: int = 224,
        height: int = 224,
        format: str = 'JPEG',
        size_bytes: Optional[int] = None
    ) -> bytes:
        img = Image.new('RGB', (width, height), color=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ))
        
        buffer = BytesIO()
        img.save(buffer, format=format)
        image_bytes = buffer.getvalue()
        
        if size_bytes and size_bytes > len(image_bytes):
            scale = int((size_bytes / len(image_bytes)) ** 0.5) + 1
            img = Image.new('RGB', (width * scale, height * scale), color='red')
            buffer = BytesIO()
            img.save(buffer, format=format, quality=95)
            image_bytes = buffer.getvalue()
            
            if len(image_bytes) < size_bytes:
                image_bytes = image_bytes + b'\x00' * (size_bytes - len(image_bytes))
        
        return image_bytes
    
    @staticmethod
    def create_oversized_file(size_mb: float = 6) -> bytes:
        target_size = int(size_mb * 1024 * 1024)
        return b'\x00' * target_size
    
    @staticmethod
    def create_invalid_file_bytes() -> bytes:
        return b"This is not an image file content"
    
    @staticmethod
    def create_prediction_response(
        material: Optional[str] = None,
        confidence: Optional[float] = None,
        is_recyclable: Optional[bool] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        material = material or random.choice(PredictionDataFactory.MATERIALS)
        confidence = confidence if confidence is not None else round(random.uniform(0.7, 0.99), 2)
        
        recyclable_materials = ['plastic', 'glass', 'metal', 'paper', 'cardboard']
        is_recyclable = is_recyclable if is_recyclable is not None else (material in recyclable_materials)
        
        return {
            "class": material,
            "confidence": confidence,
            "is_recyclable": is_recyclable,
            "message": f"Detected {material} with {confidence:.0%} confidence",
            "language": language
        }
