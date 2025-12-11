import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Request, Query
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from functools import lru_cache

from core.dependencies import get_prediction_service, PredictionService
from core.file_validator import FileValidator, FileSizeExceededError
from exceptions import ModelLoadError, PredictionError, ImageProcessingError, ValidationError
from config.settings import RATE_LIMIT_PREDICT

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@lru_cache()
def get_file_validator() -> FileValidator:
    return FileValidator()


@router.post("/predict")
@limiter.limit(RATE_LIMIT_PREDICT)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    language: str = Query("en", description="Response language (en/es)"),
    prediction_service: PredictionService = Depends(get_prediction_service),
    file_validator: FileValidator = Depends(get_file_validator)
):
    try:
        file_bytes = await file.read()
        
        file_validator.validate(file_bytes, file.content_type)
        
        response_data = await prediction_service.predict_image(file_bytes, file.filename, language)
        
        logger.info(f"Successful prediction for file: {file.filename}")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except FileSizeExceededError as e:
        logger.error(f"File size exceeded: {e}")
        raise HTTPException(status_code=413, detail=str(e))
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ImageProcessingError as e:
        logger.error(f"Processing error: {e}")
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
