import os
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

ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:8080').split(',')

HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 8000))

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

RECYCLABLE_INFO: Dict[str, Tuple[bool, str]] = {
    'cardboard': (True, "El cartón es reciclable y debe colocarse en el contenedor azul."),
    'glass': (True, "El vidrio es reciclable, pero debe estar limpio y sin tapas."),
    'metal': (True, "Los metales son reciclables y se pueden depositar en puntos específicos."),
    'paper': (True, "El papel es reciclable siempre que no esté muy sucio."),
    'plastic': (True, "El plástico es reciclable, pero algunos tipos requieren separación."),
    'trash': (False, "Este material no es reciclable y debe ir a la basura común.")
}
