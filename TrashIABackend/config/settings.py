import json
from typing import Dict, Tuple, List
from pathlib import Path
from datetime import timedelta
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Type-safe environment configuration using Pydantic BaseSettings.
    All environment variables are automatically loaded and validated.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # =========================================================================
    # FILE UPLOAD LIMITS
    # =========================================================================
    MAX_FILE_SIZE_MB: int = Field(description="Maximum file upload size in MB")
    ALLOWED_MIME_TYPES: str = Field(description="Comma-separated list of allowed MIME types")
    
    # =========================================================================
    # MODEL CONFIGURATION
    # =========================================================================
    MODEL_PATH: str = Field(description="Path to the ML model file")
    
    # =========================================================================
    # CORS CONFIGURATION
    # =========================================================================
    ALLOWED_ORIGINS: str = Field(description="Comma-separated list of allowed CORS origins")
    
    # =========================================================================
    # EXTERNAL API KEYS
    # =========================================================================

    
    # Groq API (primary LLM chat provider)
    GROQ_API_KEY: str = Field(description="Groq API key for LLM chat")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile", description="Groq model to use")
    
    # =========================================================================
    # SERVER CONFIGURATION
    # =========================================================================
    HOST: str = Field(description="Server host address")
    PORT: int = Field(description="Server port number")
    ENVIRONMENT: str = Field(description="Environment name (development/production)")
    
    # =========================================================================
    # SECURITY
    # =========================================================================
    API_KEY: str = Field(description="API key for authentication")
    REDIS_URL: str = Field(description="Redis connection URL")
    
    # =========================================================================
    # SUPABASE CONFIGURATION
    # =========================================================================
    SUPABASE_URL: str = Field(description="Supabase project URL")
    SUPABASE_KEY: str = Field(description="Supabase anon/public key")
    SUPABASE_SERVICE_KEY: str = Field(description="Supabase service role key")
    
    # =========================================================================
    # FRONTEND CONFIGURATION
    # =========================================================================
    FRONTEND_URL: str = Field(
        default="http://localhost:4200", 
        description="Frontend URL for email redirects"
    )
    
    # =========================================================================
    # RATE LIMITING (requests per minute)
    # =========================================================================
    RATE_LIMIT_PREDICT: str = Field(description="Rate limit for prediction endpoint")
    RATE_LIMIT_CHAT_SESSION: str = Field(description="Rate limit for chat session endpoint")
    RATE_LIMIT_CHAT_MESSAGE: str = Field(description="Rate limit for chat message endpoint")
    RATE_LIMIT_CHAT_HISTORY: str = Field(description="Rate limit for chat history endpoint")
    RATE_LIMIT_CHAT_DELETE: str = Field(description="Rate limit for chat delete endpoint")
    RATE_LIMIT_CHAT_UPDATE: str = Field(description="Rate limit for chat update endpoint")
    RATE_LIMIT_LOCATION: str = Field(description="Rate limit for location endpoint")
    RATE_LIMIT_BARCODE: str = Field(description="Rate limit for barcode endpoint")
    
    # =========================================================================
    # CIRCUIT BREAKER CONFIGURATION
    # =========================================================================
    CIRCUIT_BREAKER_FAIL_MAX: int = Field(description="Max failures before circuit opens")
    CIRCUIT_BREAKER_RESET_TIMEOUT: int = Field(description="Seconds before circuit resets")
    
    # =========================================================================
    # HTTP TIMEOUTS (seconds)
    # =========================================================================
    HTTP_TIMEOUT_LOCATION: float = Field(description="HTTP timeout for location service")
    HTTP_TIMEOUT_BARCODE: float = Field(description="HTTP timeout for barcode service")
    HTTP_TIMEOUT_HEALTH_CHECK: float = Field(description="HTTP timeout for health checks")
    OVERPASS_QUERY_TIMEOUT: int = Field(description="Overpass API query timeout")
    
    # =========================================================================
    # LOCATION SERVICE CONFIGURATION
    # =========================================================================
    LOCATION_CACHE_TTL_MINUTES: int = Field(description="Location cache TTL in minutes")
    LOCATION_DEFAULT_RADIUS: int = Field(description="Default search radius in meters")
    LOCATION_MIN_RADIUS: int = Field(description="Minimum search radius in meters")
    LOCATION_MAX_RADIUS: int = Field(description="Maximum search radius in meters")
    
    # =========================================================================
    # BARCODE SERVICE CONFIGURATION
    # =========================================================================
    BARCODE_MIN_LENGTH: int = Field(description="Minimum barcode length")
    
    # =========================================================================
    # EXTERNAL API URLS
    # =========================================================================
    OPEN_FOOD_FACTS_URL: str = Field(description="Open Food Facts API URL")
    UPCITEMDB_URL: str = Field(description="UPCitemdb API URL")
    OVERPASS_SERVERS: str = Field(description="Comma-separated list of Overpass servers")
    
    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================
    @property
    def max_file_size_bytes(self) -> int:
        """Convert MAX_FILE_SIZE_MB to bytes"""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024
    
    @property
    def allowed_mime_types_list(self) -> List[str]:
        """Parse comma-separated ALLOWED_MIME_TYPES to list"""
        return [mt.strip() for mt in self.ALLOWED_MIME_TYPES.split(',')]
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse comma-separated ALLOWED_ORIGINS to list"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(',')]
    
    @property
    def overpass_servers_list(self) -> List[str]:
        """Parse comma-separated OVERPASS_SERVERS to list"""
        return [server.strip() for server in self.OVERPASS_SERVERS.split(',')]
    
    @property
    def location_cache_ttl(self) -> timedelta:
        """Convert LOCATION_CACHE_TTL_MINUTES to timedelta"""
        return timedelta(minutes=self.LOCATION_CACHE_TTL_MINUTES)


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance for dependency injection.
    Uses lru_cache to ensure only one instance is created.
    """
    return Settings()


# =============================================================================
# IMAGE PROCESSING (Constants - not environment-dependent)
# =============================================================================
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

MATERIAL_TRANSLATIONS = {
    'en': {
        'cardboard': 'cardboard',
        'glass': 'glass',
        'metal': 'metal',
        'paper': 'paper',
        'plastic': 'plastic',
        'trash': 'trash'
    },
    'es': {
        'cardboard': 'cartón',
        'glass': 'vidrio',
        'metal': 'metal',
        'paper': 'papel',
        'plastic': 'plástico',
        'trash': 'basura'
    }
}


def load_recyclable_info() -> Dict[str, Dict]:
    """Load recyclability information from recyclable_info.json"""
    json_path = Path(__file__).parent / 'recyclable_info.json'
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def get_recyclable_info(material: str, language: str = "en") -> Tuple[bool, str]:
    """Get recyclability info for a material in the specified language"""
    info = RECYCLABLE_INFO.get(material, {})
    if not info:
        return False, "No recyclability information available."
    
    recyclable = info.get('recyclable', False)
    info_text = info.get('info', {})
    
    if isinstance(info_text, dict):
        text = info_text.get(language, info_text.get('en', ''))
    else:
        # Backward compatibility
        text = info_text
    
    return recyclable, text


def load_chat_prompts() -> Dict:
    """Load chat prompts from chat_prompts.json"""
    json_path = Path(__file__).parent / 'chat_prompts.json'
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


RECYCLABLE_INFO: Dict[str, Dict] = load_recyclable_info()

CHAT_PROMPTS: Dict = load_chat_prompts()


# =============================================================================
# BACKWARD COMPATIBILITY - Module-level exports
# These provide backward compatibility with existing code that imports
# directly from settings. New code should use get_settings() instead.
# =============================================================================
_settings = get_settings()

# File upload
MAX_FILE_SIZE_MB = _settings.MAX_FILE_SIZE_MB
MAX_FILE_SIZE = _settings.max_file_size_bytes
ALLOWED_MIME_TYPES = _settings.allowed_mime_types_list

# Model
MODEL_PATH = _settings.MODEL_PATH

# CORS
ALLOWED_ORIGINS = _settings.allowed_origins_list

# API Keys


# Server
HOST = _settings.HOST
PORT = _settings.PORT
ENVIRONMENT = _settings.ENVIRONMENT

# Security
API_KEY = _settings.API_KEY
REDIS_URL = _settings.REDIS_URL

# Rate limiting
RATE_LIMIT_PREDICT = _settings.RATE_LIMIT_PREDICT
RATE_LIMIT_CHAT_SESSION = _settings.RATE_LIMIT_CHAT_SESSION
RATE_LIMIT_CHAT_MESSAGE = _settings.RATE_LIMIT_CHAT_MESSAGE
RATE_LIMIT_CHAT_HISTORY = _settings.RATE_LIMIT_CHAT_HISTORY
RATE_LIMIT_CHAT_DELETE = _settings.RATE_LIMIT_CHAT_DELETE
RATE_LIMIT_CHAT_UPDATE = _settings.RATE_LIMIT_CHAT_UPDATE
RATE_LIMIT_LOCATION = _settings.RATE_LIMIT_LOCATION
RATE_LIMIT_BARCODE = _settings.RATE_LIMIT_BARCODE

# Circuit breaker
CIRCUIT_BREAKER_FAIL_MAX = _settings.CIRCUIT_BREAKER_FAIL_MAX
CIRCUIT_BREAKER_RESET_TIMEOUT = _settings.CIRCUIT_BREAKER_RESET_TIMEOUT

# HTTP timeouts
HTTP_TIMEOUT_LOCATION = _settings.HTTP_TIMEOUT_LOCATION
HTTP_TIMEOUT_BARCODE = _settings.HTTP_TIMEOUT_BARCODE
HTTP_TIMEOUT_HEALTH_CHECK = _settings.HTTP_TIMEOUT_HEALTH_CHECK
OVERPASS_QUERY_TIMEOUT = _settings.OVERPASS_QUERY_TIMEOUT

# Location service
LOCATION_CACHE_TTL_MINUTES = _settings.LOCATION_CACHE_TTL_MINUTES
LOCATION_CACHE_TTL = _settings.location_cache_ttl
LOCATION_DEFAULT_RADIUS = _settings.LOCATION_DEFAULT_RADIUS
LOCATION_MIN_RADIUS = _settings.LOCATION_MIN_RADIUS
LOCATION_MAX_RADIUS = _settings.LOCATION_MAX_RADIUS

# Barcode service
BARCODE_MIN_LENGTH = _settings.BARCODE_MIN_LENGTH

# External API URLs
OPEN_FOOD_FACTS_URL = _settings.OPEN_FOOD_FACTS_URL
UPCITEMDB_URL = _settings.UPCITEMDB_URL
OVERPASS_SERVERS = _settings.overpass_servers_list

# Supabase
SUPABASE_URL = _settings.SUPABASE_URL
SUPABASE_KEY = _settings.SUPABASE_KEY
SUPABASE_SERVICE_KEY = _settings.SUPABASE_SERVICE_KEY

# Frontend
FRONTEND_URL = _settings.FRONTEND_URL
