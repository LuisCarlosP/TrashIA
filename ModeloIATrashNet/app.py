import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config.settings import ALLOWED_ORIGINS
from routes.prediction import router as prediction_router
from routes.chat import router as chat_router
from core.dependencies import get_prediction_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar rate limiting
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Clasificador de Basura IA",
    description="API para clasificar tipos de basura y determinar reciclabilidad",
    version="1.0.0"
)

# Agregar rate limiter al state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)
app.include_router(chat_router)

@app.get("/")
async def root():
    return {
        "message": "Bienvenido a TrashIA - Clasificador de Basura",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "chat": "/chat",
            "docs": "/docs"
        }
    }

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