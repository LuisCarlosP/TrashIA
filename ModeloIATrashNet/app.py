"""
API FastAPI para clasificación de basura usando IA.
"""
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import ALLOWED_ORIGINS
from services import ModelService, ImageProcessor, ResponseFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Clasificador de Basura IA",
    description="API para clasificar tipos de basura y determinar reciclabilidad",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model_service = ModelService()
    image_processor = ImageProcessor()
    response_formatter = ResponseFormatter()
    logger.info("Servicios inicializados correctamente")
except Exception as e:
    logger.error(f"Error al inicializar servicios: {e}")
    raise

@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud."""
    return {"status": "healthy", "message": "API funcionando correctamente"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predice el tipo de basura y su reciclabilidad.
    
    Args:
        file: Archivo de imagen a clasificar
        
    Returns:
        JSON con la clasificación y información de reciclabilidad
    """
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="El archivo debe ser una imagen válida"
            )
        file_bytes = await file.read()
        img_array = image_processor.process_image(file_bytes)
        class_name, confidence = model_service.predict(img_array)
        response_data = response_formatter.format_prediction_response(
            class_name, confidence
        )
        
        logger.info(f"Predicción exitosa para archivo: {file.filename}")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        error_response = response_formatter.format_error_response(
            str(e), 400
        )
        return JSONResponse(content=error_response, status_code=400)
    except Exception as e:
        logger.error(f"Error interno del servidor: {e}")
        error_response = response_formatter.format_error_response(
            "Error interno del servidor", 500
        )
        return JSONResponse(content=error_response, status_code=500)

if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT
    
    uvicorn.run(app, host=HOST, port=PORT)