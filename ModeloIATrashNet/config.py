"""
Configuración centralizada para la aplicación de clasificación de basura.
"""
import os
from typing import Dict, Tuple
from pathlib import Path

# Cargar variables de entorno desde archivo .env si existe
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback manual si python-dotenv no está instalado
    def load_env_file():
        """Carga variables de entorno desde archivo .env"""
        env_path = Path(__file__).parent / '.env'
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    load_env_file()

# Configuración de imagen
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# Configuración del modelo
MODEL_PATH = os.getenv('MODEL_PATH', 'modelo_basura.h5')

# Configuración CORS
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:8080').split(',')

# Configuración del servidor
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 8000))

# Clases del modelo
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Información de reciclaje
RECYCLABLE_INFO: Dict[str, Tuple[bool, str]] = {
    'cardboard': (True, "El cartón es reciclable y debe colocarse en el contenedor azul."),
    'glass': (True, "El vidrio es reciclable, pero debe estar limpio y sin tapas."),
    'metal': (True, "Los metales son reciclables y se pueden depositar en puntos específicos."),
    'paper': (True, "El papel es reciclable siempre que no esté muy sucio."),
    'plastic': (True, "El plástico es reciclable, pero algunos tipos requieren separación."),
    'trash': (False, "Este material no es reciclable y debe ir a la basura común.")
}
