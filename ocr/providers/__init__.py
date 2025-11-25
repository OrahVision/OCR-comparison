"""
OCR Providers package.

Each provider implements a common interface for OCR processing.
All providers inherit from OCRProvider base class and register
with the ProviderRegistry.
"""

# Cloud providers (already implemented)
from .google_vision import GoogleVisionOCR, check_google_cloud_auth

# Local providers
from .easyocr_provider import EasyOCRProvider, EASYOCR_AVAILABLE
from .paddleocr_provider import PaddleOCRProvider, PADDLEOCR_AVAILABLE
from .doctr_provider import DocTRProvider, DOCTR_AVAILABLE
from .windows_ocr import WindowsOCRProvider, IS_WINDOWS, WINOCR_AVAILABLE
from .tesseract_provider import TesseractProvider, TESSERACT_AVAILABLE

# keras-ocr has compatibility issues with NumPy 2.0 (imgaug dependency)
try:
    from .keras_ocr_provider import KerasOCRProvider, KERAS_OCR_AVAILABLE
except (ImportError, AttributeError) as e:
    KERAS_OCR_AVAILABLE = False
    KerasOCRProvider = None

# VLM-based local providers (require GPU)
from .deepseek_ocr import DeepSeekOCRProvider, DEEPSEEK_AVAILABLE
from .florence2_provider import Florence2Provider, FLORENCE_AVAILABLE
from .qwen2vl_provider import Qwen2VLProvider, QWEN2VL_AVAILABLE

__all__ = [
    # Cloud providers
    'GoogleVisionOCR',
    'check_google_cloud_auth',
    # Local providers
    'EasyOCRProvider',
    'PaddleOCRProvider',
    'DocTRProvider',
    'WindowsOCRProvider',
    'KerasOCRProvider',
    'TesseractProvider',
    # VLM providers
    'DeepSeekOCRProvider',
    'Florence2Provider',
    'Qwen2VLProvider',
    # Availability flags
    'EASYOCR_AVAILABLE',
    'PADDLEOCR_AVAILABLE',
    'DOCTR_AVAILABLE',
    'KERAS_OCR_AVAILABLE',
    'TESSERACT_AVAILABLE',
    'DEEPSEEK_AVAILABLE',
    'FLORENCE_AVAILABLE',
    'QWEN2VL_AVAILABLE',
    'IS_WINDOWS',
    'WINOCR_AVAILABLE',
]
