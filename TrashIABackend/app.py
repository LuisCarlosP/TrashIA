import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config.settings import ALLOWED_ORIGINS, REDIS_URL, ENVIRONMENT
from routes.prediction import router as prediction_router
from routes.chat import router as chat_router
from routes.location import router as location_router
from routes.barcode import router as barcode_router
from routes.health import router as health_router
from core.dependencies import get_prediction_service
from core.security import get_api_key, validate_security_config
from core.error_handler import register_exception_handlers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler for proper startup/shutdown."""
    # Startup
    logger.info("Starting application...")
    validate_security_config()
    try:
        get_prediction_service()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise
    yield
    # Shutdown
    logger.info("Shutting down application...")

limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL)

tags_metadata = [
    {"name": "General", "description": "General API information"},
    {"name": "Health", "description": "Health check endpoints"},
    {"name": "Prediction", "description": "AI waste classification endpoints"},
    {"name": "Chat", "description": "Recycling assistant chat endpoints"},
    {"name": "Location", "description": "Recycling point location endpoints"},
    {"name": "Barcode", "description": "Product barcode scanning endpoints"},
]

app = FastAPI(
    title="TrashIA - AI Waste Classifier",
    description="API for classifying waste types and determining recyclability",
    version="1.0.0",
    openapi_tags=tags_metadata,
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

register_exception_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def skip_auth_for_options(request: Request, call_next):
    if request.method == "OPTIONS":
        response = await call_next(request)
        return response
    return await call_next(request)

if ENVIRONMENT == "production":
    app.add_middleware(HTTPSRedirectMiddleware)
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

app.include_router(health_router, tags=["Health"])
app.include_router(prediction_router, tags=["Prediction"], dependencies=[Depends(get_api_key)])
app.include_router(chat_router, tags=["Chat"], dependencies=[Depends(get_api_key)])
app.include_router(location_router, tags=["Location"], dependencies=[Depends(get_api_key)])
app.include_router(barcode_router, tags=["Barcode"], dependencies=[Depends(get_api_key)])

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Welcome to TrashIA - AI Waste Classifier",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "chat": "/chat",
            "location": "/location/recycling-points",
            "barcode": "/barcode/{code}",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    from config.settings import HOST, PORT
    
    uvicorn.run(app, host=HOST, port=PORT)