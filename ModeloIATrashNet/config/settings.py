import os
import json
from typing import Dict, Tuple
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    def load_env_file():
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    load_env_file()

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

MODEL_PATH = os.getenv('MODEL_PATH', 'models/modelo_basura.h5')

ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:8080,https://luiscarlosp.github.io').split(',')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 8000))
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

API_KEY = os.getenv('API_KEY')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

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

def load_recyclable_info() -> Dict[str, Tuple[bool, str]]:
    """Carga información de reciclabilidad desde recyclable_info.json"""
    json_path = Path(__file__).parent / 'recyclable_info.json'
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return {key: (value['recyclable'], value['info']) for key, value in data.items()}

def load_chat_prompts() -> Dict:
    """Carga los prompts del chat desde chat_prompts.json"""
    json_path = Path(__file__).parent / 'chat_prompts.json'
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

RECYCLABLE_INFO: Dict[str, Tuple[bool, str]] = load_recyclable_info()

CHAT_PROMPTS: Dict = load_chat_prompts()
