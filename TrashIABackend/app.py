import logging
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
from core.dependencies import get_prediction_service
from core.security import get_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure rate limiting with Redis
limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL)

app = FastAPI(
    title="TrashIA - AI Waste Classifier",
    description="API for classifying waste types and determining recyclability",
    version="1.0.0"
)

# Add rate limiter to state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware must be added before routers to handle OPTIONS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to skip API key validation for OPTIONS requests
@app.middleware("http")
async def skip_auth_for_options(request: Request, call_next):
    if request.method == "OPTIONS":
        # Skip API key validation for OPTIONS (handled by CORS middleware)
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

# Include routers with API key dependency
# Note: OPTIONS requests are handled by CORS middleware and don't require authentication
app.include_router(prediction_router, dependencies=[Depends(get_api_key)])
app.include_router(chat_router, dependencies=[Depends(get_api_key)])
app.include_router(location_router, dependencies=[Depends(get_api_key)])
app.include_router(barcode_router, dependencies=[Depends(get_api_key)])

@app.get("/")
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is working correctly"}

try:
    get_prediction_service()
    logger.info("Services initialized successfully")
except Exception as e:
    logger.error(f"Error initializing services: {e}")
    raise

if __name__ == "__main__":
    import uvicorn
    from config.settings import HOST, PORT
    
    uvicorn.run(app, host=HOST, port=PORT)