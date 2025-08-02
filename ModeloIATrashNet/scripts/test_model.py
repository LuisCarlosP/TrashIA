import sys
import logging
from pathlib import Path
from tkinter import Tk, filedialog

from services.trash_services import ModelService, ImageProcessor, ResponseFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def select_image_file() -> str:
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
    )
    return file_path

def test_model_with_image(image_path: str) -> None:
    try:
        model_service = ModelService()
        image_processor = ImageProcessor()
        response_formatter = ResponseFormatter()
        path = Path(image_path)
        if not path.exists():
            logger.error(f"El archivo {image_path} no existe")
            return
        
        with open(image_path, 'rb') as f:
            file_bytes = f.read()
        
        logger.info(f"Procesando imagen: {image_path}")
        img_array = image_processor.process_image(file_bytes)
        
        class_name, confidence = model_service.predict(img_array)
        
        response = response_formatter.format_prediction_response(class_name, confidence)
        
        print("\n" + "="*50)
        print(f"RESULTADOS PARA: {path.name}")
        print("="*50)
        print(f"Clase predicha: {response['clase'].upper()}")
        print(f"Confianza: {response['confianza']:.2%}")
        print(f"Es reciclable: {'Sí' if response['es_reciclable'] else 'No'}")
        print(f"Mensaje: {response['mensaje']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error al probar el modelo: {e}")
        print(f"Error: {e}")

def main():
    if len(sys.argv) == 2:
        image_path = sys.argv[1]
    else:
        image_path = select_image_file()
        
        if not image_path:
            print("No se seleccionó ninguna imagen.")
            sys.exit(1)
    
    test_model_with_image(image_path)

if __name__ == "__main__":
    main()
