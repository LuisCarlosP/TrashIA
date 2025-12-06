"""
Startup script for TrashIA server.
Uses configuration defined in config/settings.py and .env file
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
        logger.warning(f"Model file not found at: {MODEL_PATH}")
        logger.warning("Make sure the model file exists before running predictions")
        return False
    else:
        logger.info(f"Model file found at: {MODEL_PATH}")
        return True

def main():
    """
    Starts the uvicorn server with project configuration.
    """
    logger.info("=" * 60)
    logger.info("Starting TrashIA server...")
    logger.info("=" * 60)
    
    model_exists = validate_environment()
    
    # Show configuration
    logger.info(f"Host: {HOST}")
    logger.info(f"Port: {PORT}")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Auto-reload: Enabled")
    
    if not model_exists:
        logger.warning("Server will start but predictions will fail")
    
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
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        raise

if __name__ == "__main__":
    main()
