import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import tensorflow as tf
from services.trash_services import ImageProcessor, ResponseFormatter, ModelService
from exceptions import ImageProcessingError, ModelLoadError

def test_process_image_success():
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    mock_img.resize.return_value = mock_img
    
    with patch("PIL.Image.open", return_value=mock_img):
        with patch("tensorflow.keras.utils.img_to_array", return_value=np.zeros((224, 224, 3))):
            with patch("tensorflow.expand_dims", return_value=np.zeros((1, 224, 224, 3))):
                with patch("services.trash_services.preprocess_input", return_value=np.zeros((1, 224, 224, 3))):
                    processor = ImageProcessor()
                    result = processor.process_image(b"fake_bytes")
                    assert isinstance(result, np.ndarray)

def test_process_image_error():
    with patch("PIL.Image.open", side_effect=Exception("Invalid image")):
        processor = ImageProcessor()
        with pytest.raises(ImageProcessingError):
            processor.process_image(b"fake_bytes")

def test_format_prediction_response():
    formatter = ResponseFormatter()
    response = formatter.format_prediction_response("plastic", 0.95)
    
    assert response["clase"] == "plastic"
    assert response["confianza"] == 0.95
    assert response["es_reciclable"] is True

def test_format_prediction_response_unknown():
    formatter = ResponseFormatter()
    response = formatter.format_prediction_response("unknown_class", 0.5)
    
    assert response["clase"] == "unknown_class"
    assert response["es_reciclable"] is False

def test_model_service_predict():
    with patch("tensorflow.keras.models.load_model") as mock_load:
        mock_model = MagicMock()
        predictions_array = np.array([[0.1, 0.9]])
        mock_model.predict.return_value = predictions_array
        mock_load.return_value = mock_model
        
        with patch("services.trash_services.CLASS_NAMES", ["paper", "plastic"]):
            mock_tensor_item = MagicMock()
            mock_tensor_item.numpy.return_value = 0.9
            
            class MockSoftmaxResult:
                def __getitem__(self, idx):
                    if idx == 1:
                        return mock_tensor_item
                    return MagicMock()
            
            with patch("tensorflow.nn.softmax", return_value=MockSoftmaxResult()):
                service = ModelService()
                class_name, confidence = service.predict(np.zeros((1, 224, 224, 3)))
                
                assert class_name == "plastic"
                assert confidence == 0.9 
