"""
OCR Providers package.

Each provider implements a common interface for OCR processing.
All providers inherit from OCRProvider base class and register
with the ProviderRegistry.
"""

# Cloud providers
from .google_vision import GoogleVisionOCR, check_google_cloud_auth

# LLM Vision providers (cloud)
try:
    from .gemini_vision import GeminiVisionOCR, get_gemini_api_key, GEMINI_AVAILABLE
except ImportError:
    GEMINI_AVAILABLE = False
    GeminiVisionOCR = None
    get_gemini_api_key = lambda: None

try:
    from .claude_vision import ClaudeVisionOCR, get_anthropic_api_key, CLAUDE_AVAILABLE
except ImportError:
    CLAUDE_AVAILABLE = False
    ClaudeVisionOCR = None
    get_anthropic_api_key = lambda: None

try:
    from .openai_vision import OpenAIVisionOCR, get_openai_api_key, OPENAI_VISION_AVAILABLE
except ImportError:
    OPENAI_VISION_AVAILABLE = False
    OpenAIVisionOCR = None
    get_openai_api_key = lambda: None

try:
    from .azure_doc_intel import AzureDocIntelOCR, get_azure_credentials, AZURE_AVAILABLE
except ImportError:
    AZURE_AVAILABLE = False
    AzureDocIntelOCR = None
    get_azure_credentials = lambda: (None, None)

try:
    from .aws_textract import AWSTextractOCR, check_aws_credentials, TEXTRACT_AVAILABLE
except ImportError:
    TEXTRACT_AVAILABLE = False
    AWSTextractOCR = None
    check_aws_credentials = lambda: False

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
    # LLM Vision providers
    'GeminiVisionOCR',
    'get_gemini_api_key',
    'ClaudeVisionOCR',
    'get_anthropic_api_key',
    'OpenAIVisionOCR',
    'get_openai_api_key',
    'AzureDocIntelOCR',
    'get_azure_credentials',
    'AWSTextractOCR',
    'check_aws_credentials',
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
    'GEMINI_AVAILABLE',
    'CLAUDE_AVAILABLE',
    'OPENAI_VISION_AVAILABLE',
    'AZURE_AVAILABLE',
    'TEXTRACT_AVAILABLE',
    'IS_WINDOWS',
    'WINOCR_AVAILABLE',
]
