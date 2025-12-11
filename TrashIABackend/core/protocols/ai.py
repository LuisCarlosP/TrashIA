from typing import Protocol, Tuple, runtime_checkable
from abc import abstractmethod
import numpy as np


@runtime_checkable
class AIModelProtocol(Protocol):
    @abstractmethod
    def predict(self, input_data: np.ndarray) -> Tuple[str, float]:
        ...


@runtime_checkable
class ImageProcessorProtocol(Protocol):
    @abstractmethod
    def process_image(self, file_bytes: bytes) -> np.ndarray:
        ...
