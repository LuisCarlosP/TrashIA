import logging
import magic
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.dependencies import get_prediction_service, PredictionService
from exceptions import ModelLoadError, PredictionError, ImageProcessingError, ValidationError

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Tamaño máximo de archivo: 5MB
MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/jpg']

@router.post("/predict")
@limiter.limit("10/minute")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    try:
        # Leer contenido del archivo
        file_bytes = await file.read()
        
        # Validar tamaño del archivo
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"El archivo excede el tamaño máximo permitido de {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Validar tipo MIME real del contenido
        mime = magic.from_buffer(file_bytes, mime=True)
        if mime not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de archivo no permitido. Solo se aceptan: {', '.join(ALLOWED_MIME_TYPES)}"
            )
        
        # Validar content-type del header
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="El archivo debe ser una imagen válida"
            )
        
        response_data = await prediction_service.predict_image(file_bytes, file.filename)
        
        logger.info(f"Predicción exitosa para archivo: {file.filename}")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except (ImageProcessingError, ValidationError) as e:
        logger.error(f"Error de validación/procesamiento: {e}")
        error_response = prediction_service.format_error(str(e), 400)
        return JSONResponse(content=error_response, status_code=400)
    except (ModelLoadError, PredictionError) as e:
        logger.error(f"Error del modelo: {e}")
        error_response = prediction_service.format_error(str(e), 500)
        return JSONResponse(content=error_response, status_code=500)
    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        error_response = prediction_service.format_error(str(e), 400)
        return JSONResponse(content=error_response, status_code=400)
    except Exception as e:
        logger.error(f"Error interno del servidor: {e}")
        error_response = prediction_service.format_error("Error interno del servidor", 500)
        return JSONResponse(content=error_response, status_code=500)
