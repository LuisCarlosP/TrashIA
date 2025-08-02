import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import ALLOWED_ORIGINS
from routes.prediction import router as prediction_router
from core.dependencies import get_prediction_service

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

app.include_router(prediction_router)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API funcionando correctamente"}

try:
    get_prediction_service()
    logger.info("Servicios inicializados correctamente")
except Exception as e:
    logger.error(f"Error al inicializar servicios: {e}")
    raise

if __name__ == "__main__":
    import uvicorn
    from config.settings import HOST, PORT
    
    uvicorn.run(app, host=HOST, port=PORT)