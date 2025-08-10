"""
Script de inicio para el servidor TrashIA.
Utiliza la configuración definida en config/settings.py y el archivo .env
"""
import uvicorn
import logging
from pathlib import Path
from config.settings import HOST, PORT, MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_environment():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        logger.warning(f"Archivo del modelo no encontrado en: {MODEL_PATH}")
        logger.warning("Asegúrate de que el archivo del modelo exista antes de ejecutar predicciones")
        return False
    else:
        logger.info(f"Archivo del modelo encontrado en: {MODEL_PATH}")
        return True

def main():
    """
    Inicia el servidor uvicorn con la configuración del proyecto.
    """
    logger.info("=" * 60)
    logger.info("Iniciando servidor TrashIA...")
    logger.info("=" * 60)
    
    model_exists = validate_environment()
    
    # Mostrar configuración
    logger.info(f"Host: {HOST}")
    logger.info(f"Puerto: {PORT}")
    logger.info(f"Modelo: {MODEL_PATH}")
    logger.info(f"Recarga automática: Habilitada")
    
    if not model_exists:
        logger.warning("⚠️  El servidor se iniciará pero las predicciones fallarán hasta que el modelo esté disponible")
    
    logger.info("=" * 60)
    
    try:
        uvicorn.run(
            "app:app", 
            host=HOST, 
            port=PORT, 
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Servidor detenido por el usuario")
    except Exception as e:
        logger.error(f"Error al iniciar el servidor: {e}")
        raise

if __name__ == "__main__":
    main()
