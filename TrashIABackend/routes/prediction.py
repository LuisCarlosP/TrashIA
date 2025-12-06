import logging
import magic
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Request, Query
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.dependencies import get_prediction_service, PredictionService
from exceptions import ModelLoadError, PredictionError, ImageProcessingError, ValidationError

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Maximum file size: 5MB
MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/jpg']

@router.options("/predict")
async def predict_options():
    """Handle CORS preflight requests"""
    return JSONResponse(content={}, status_code=200)

@router.post("/predict")
@limiter.limit("10/minute")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    language: str = Query("en", description="Response language (en/es)"),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    try:
        # Read file content
        file_bytes = await file.read()
        
        # Validate file size
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Validate actual MIME type of content
        mime = magic.from_buffer(file_bytes, mime=True)
        if mime not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Only accepted: {', '.join(ALLOWED_MIME_TYPES)}"
            )
        
        # Validate content-type header
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be a valid image"
            )
        
        response_data = await prediction_service.predict_image(file_bytes, file.filename, language)
        
        logger.info(f"Successful prediction for file: {file.filename}")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except (ImageProcessingError, ValidationError) as e:
        logger.error(f"Validation/processing error: {e}")
        error_response = prediction_service.format_error(str(e), 400)
        return JSONResponse(content=error_response, status_code=400)
    except (ModelLoadError, PredictionError) as e:
        logger.error(f"Model error: {e}")
        error_response = prediction_service.format_error(str(e), 500)
        return JSONResponse(content=error_response, status_code=500)
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        error_response = prediction_service.format_error(str(e), 400)
        return JSONResponse(content=error_response, status_code=400)
    except Exception as e:
        logger.error(f"Internal server error: {e}")
        error_response = prediction_service.format_error("Internal server error", 500)
        return JSONResponse(content=error_response, status_code=500)
