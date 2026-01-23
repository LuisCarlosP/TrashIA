import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any
import httpx
from fastapi import APIRouter, Request


from core.dependencies import get_prediction_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health")

_groq_health_cache: Dict[str, Any] | None = None
_groq_health_cache_time: float | None = None


async def check_service_health(
    name: str,
    url: str,
    timeout: float = 5.0,
    expected_status: int = 200
) -> Dict[str, Any]:
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            latency_ms = int((time.time() - start_time) * 1000)
            
            if response.status_code == expected_status:
                status = "healthy"
            else:
                status = "degraded"
            
            return {
                "service": name,
                "status": status,
                "latency_ms": latency_ms,
                "last_check": datetime.now(timezone.utc).isoformat()
            }
    except httpx.TimeoutException:
        return {
            "service": name,
            "status": "unhealthy",
            "error": "timeout",
            "last_check": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {
            "service": name,
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.now(timezone.utc).isoformat()
        }


@router.get("")
@router.head("")
async def health_check():
    return {
        "status": "healthy",
        "service": "trashia-api",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def check_model_health() -> Dict[str, Any]:
    """Check ML model health status."""
    try:
        prediction_service = get_prediction_service()
        health_info = prediction_service.check_model_health()
        health_info["service"] = "ml_model"
        health_info["last_check"] = datetime.now(timezone.utc).isoformat()
        return health_info
    except Exception as e:
        return {
            "service": "ml_model",
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e),
            "last_check": datetime.now(timezone.utc).isoformat()
        }


@router.get("/model")
async def check_model_health_endpoint():
    """
    Check ML model availability and health.
    
    Returns detailed status of the machine learning model including:
    - Whether the model is loaded
    - Any initialization errors
    - Model status (healthy/unhealthy)
    """
    return check_model_health()

@router.get("/dependencies")
async def check_all_dependencies():
    results = {}
    
    # Check ML model health
    model_result = check_model_health()
    results["ml_model"] = model_result
    
    groq_result = await check_groq_health()
    results["groq"] = groq_result
    
    osm_result = await check_osm_health()
    results["openstreetmap"] = osm_result
    
    off_result = await check_openfoodfacts_health()
    results["openfoodfacts"] = off_result
    
    all_healthy = all(r["status"] == "healthy" for r in results.values())
    any_unhealthy = any(r["status"] == "unhealthy" for r in results.values())
    
    if all_healthy:
        overall_status = "healthy"
    elif any_unhealthy:
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": results
    }


@router.get("/groq")
async def check_groq_health():
    """
    Results are cached for 5 minutes.
    """
    global _groq_health_cache, _groq_health_cache_time
    
    cache_ttl_seconds = 300  # 5 minutes
    current_time = time.time()
    
    # Reuse existing cache variables for simplicity (renaming variables would require broader refactor)
    # Ideally should be renamed to generic names or groq specific
    if (_groq_health_cache is not None and 
        _groq_health_cache_time is not None and
        current_time - _groq_health_cache_time < cache_ttl_seconds):
        cached = _groq_health_cache.copy()
        if cached.get("service") == "groq": # Ensure we don't return old gemini cache
            cached["cached"] = True
            cached["cache_expires_in_seconds"] = int(cache_ttl_seconds - (current_time - _groq_health_cache_time))
            return cached
    
    # Import here to avoid circular dependencies if any
    from config.settings import get_settings
    settings = get_settings()
    
    if not settings.GROQ_API_KEY:
        result = {
            "service": "groq",
            "status": "unhealthy",
            "error": "API key not configured",
            "last_check": datetime.now(timezone.utc).isoformat()
        }
        _groq_health_cache = result
        _groq_health_cache_time = current_time
        return result
    
    start_time = time.time()
    try:
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        
        # Simple generation to check connectivity
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        result = {
            "service": "groq",
            "status": "healthy",
            "model": settings.GROQ_MODEL,
            "latency_ms": latency_ms,
            "last_check": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        result = {
            "service": "groq",
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.now(timezone.utc).isoformat()
        }
    
    # Update cache
    _groq_health_cache = result
    _groq_health_cache_time = current_time
    
    return result


@router.get("/osm")
async def check_osm_health():
    return await check_service_health(
        name="openstreetmap",
        url="https://overpass-api.de/api/status",
        timeout=10.0
    )


@router.get("/openfoodfacts")
async def check_openfoodfacts_health():
    return await check_service_health(
        name="openfoodfacts",
        url="https://world.openfoodfacts.org/api/v2/product/737628064502.json",
        timeout=10.0
    )
