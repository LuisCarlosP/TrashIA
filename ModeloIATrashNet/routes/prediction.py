import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse

from core.dependencies import get_prediction_service, PredictionService

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="El archivo debe ser una imagen válida"
            )
        
        file_bytes = await file.read()
        response_data = await prediction_service.predict_image(file_bytes, file.filename)
        
        logger.info(f"Predicción exitosa para archivo: {file.filename}")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        error_response = prediction_service.format_error(str(e), 400)
        return JSONResponse(content=error_response, status_code=400)
    except Exception as e:
        logger.error(f"Error interno del servidor: {e}")
        error_response = prediction_service.format_error("Error interno del servidor", 500)
        return JSONResponse(content=error_response, status_code=500)
