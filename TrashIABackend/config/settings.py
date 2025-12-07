import os
import json
from typing import Dict, Tuple, List
from pathlib import Path
from datetime import timedelta

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# IMAGE PROCESSING
# =============================================================================
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# =============================================================================
# FILE UPLOAD LIMITS
# =============================================================================
MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB'))
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
ALLOWED_MIME_TYPES: List[str] = os.getenv('ALLOWED_MIME_TYPES').split(',')

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_PATH = os.getenv('MODEL_PATH')

# =============================================================================
# CORS CONFIGURATION
# =============================================================================
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS').split(',')

# =============================================================================
# EXTERNAL API KEYS
# =============================================================================
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# =============================================================================
# SERVER CONFIGURATION
# =============================================================================
HOST = os.getenv('HOST')
PORT = int(os.getenv('PORT'))
ENVIRONMENT = os.getenv('ENVIRONMENT')

# =============================================================================
# SECURITY
# =============================================================================
API_KEY = os.getenv('API_KEY')
REDIS_URL = os.getenv('REDIS_URL')

# =============================================================================
# RATE LIMITING (requests per minute)
# =============================================================================
RATE_LIMIT_PREDICT = os.getenv('RATE_LIMIT_PREDICT')
RATE_LIMIT_CHAT_SESSION = os.getenv('RATE_LIMIT_CHAT_SESSION')
RATE_LIMIT_CHAT_MESSAGE = os.getenv('RATE_LIMIT_CHAT_MESSAGE')
RATE_LIMIT_CHAT_HISTORY = os.getenv('RATE_LIMIT_CHAT_HISTORY')
RATE_LIMIT_CHAT_DELETE = os.getenv('RATE_LIMIT_CHAT_DELETE')
RATE_LIMIT_CHAT_UPDATE = os.getenv('RATE_LIMIT_CHAT_UPDATE')
RATE_LIMIT_LOCATION = os.getenv('RATE_LIMIT_LOCATION')
RATE_LIMIT_BARCODE = os.getenv('RATE_LIMIT_BARCODE')

# =============================================================================
# CIRCUIT BREAKER CONFIGURATION
# =============================================================================
CIRCUIT_BREAKER_FAIL_MAX = int(os.getenv('CIRCUIT_BREAKER_FAIL_MAX'))
CIRCUIT_BREAKER_RESET_TIMEOUT = int(os.getenv('CIRCUIT_BREAKER_RESET_TIMEOUT'))

# =============================================================================
# HTTP TIMEOUTS (seconds)
# =============================================================================
HTTP_TIMEOUT_LOCATION = float(os.getenv('HTTP_TIMEOUT_LOCATION'))
HTTP_TIMEOUT_BARCODE = float(os.getenv('HTTP_TIMEOUT_BARCODE'))
HTTP_TIMEOUT_HEALTH_CHECK = float(os.getenv('HTTP_TIMEOUT_HEALTH_CHECK'))
OVERPASS_QUERY_TIMEOUT = int(os.getenv('OVERPASS_QUERY_TIMEOUT'))

# =============================================================================
# LOCATION SERVICE CONFIGURATION
# =============================================================================
LOCATION_CACHE_TTL_MINUTES = int(os.getenv('LOCATION_CACHE_TTL_MINUTES'))
LOCATION_CACHE_TTL = timedelta(minutes=LOCATION_CACHE_TTL_MINUTES)
LOCATION_DEFAULT_RADIUS = int(os.getenv('LOCATION_DEFAULT_RADIUS'))
LOCATION_MIN_RADIUS = int(os.getenv('LOCATION_MIN_RADIUS'))
LOCATION_MAX_RADIUS = int(os.getenv('LOCATION_MAX_RADIUS'))

# =============================================================================
# BARCODE SERVICE CONFIGURATION
# =============================================================================
BARCODE_MIN_LENGTH = int(os.getenv('BARCODE_MIN_LENGTH'))

# =============================================================================
# EXTERNAL API URLS
# =============================================================================
OPEN_FOOD_FACTS_URL = os.getenv('OPEN_FOOD_FACTS_URL')
UPCITEMDB_URL = os.getenv('UPCITEMDB_URL')
OVERPASS_SERVERS = os.getenv('OVERPASS_SERVERS').split(',')

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
