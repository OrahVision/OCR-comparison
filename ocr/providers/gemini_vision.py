"""
Gemini Vision OCR Provider

Uses Google's Gemini 2.0 Flash for OCR via vision capabilities.
https://ai.google.dev/gemini-api

License: Commercial API
Pricing: ~$0.10/1M tokens (~$0.001/image) - Best price/performance for LLM OCR
"""

import base64
import logging
import os
from typing import List, Optional

from ..base import OCRProvider, OCRResult, ProviderType, Timer
from ..registry import register_provider

# Check for keyring (shared with TTS)
try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

# Check for google-generativeai
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logging.info("google-generativeai not installed. Install with: pip install google-generativeai")


def get_gemini_api_key() -> Optional[str]:
    """Get Gemini API key from keyring or environment."""
    # Try keyring first (shared with TTS service)
    if KEYRING_AVAILABLE:
        try:
            key = keyring.get_password("gemini", "api_key")
            if key:
                return key
        except Exception:
            pass

    # Try environment variable
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key

    return None


GEMINI_AVAILABLE = GENAI_AVAILABLE and get_gemini_api_key() is not None


@register_provider
class GeminiVisionOCR(OCRProvider):
    """
    Gemini Vision OCR Provider

    Features:
    - Uses Gemini 2.0 Flash for fast, cheap OCR
    - Excellent multilingual support
    - Can handle complex layouts
    - Shares API key with TTS service (keyring: gemini/api_key)
    """

    PROVIDER_NAME = "gemini_vision"
    PROVIDER_TYPE = ProviderType.LLM
    COMMERCIAL_USE = True
    LICENSE = "Commercial API"

    # Pricing per 1M tokens (approximate)
    INPUT_COST_PER_1M = 0.10  # $0.10/1M input tokens
    OUTPUT_COST_PER_1M = 0.40  # $0.40/1M output tokens

    DEFAULT_MODEL = "gemini-2.0-flash"

    OCR_PROMPT = """Extract ALL text from this image exactly as it appears.
Preserve the original formatting, line breaks, and structure.
Do not add any commentary, descriptions, or explanations.
Output ONLY the extracted text, nothing else.
If there are multiple columns, transcribe left to right, top to bottom.
Include all visible text including headers, footers, page numbers, and captions."""

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        **kwargs
    ):
        """
        Initialize Gemini Vision OCR provider.

        Args:
            model: Gemini model to use (default: gemini-2.0-flash)
            api_key: API key (defaults to keyring/env)
        """
        self.model_name = model or self.DEFAULT_MODEL
        self._api_key = api_key
        self._initialized = False
        self._init_error = None
        self._model = None

        if GENAI_AVAILABLE:
            self._initialize()

    def _initialize(self):
        """Initialize the Gemini client."""
        if self._initialized:
            return

        try:
            api_key = self._api_key or get_gemini_api_key()
            if not api_key:
                self._init_error = "No Gemini API key found. Set via keyring (gemini/api_key) or GEMINI_API_KEY env var."
                return

            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(self.model_name)
            self._initialized = True
            logging.info(f"Gemini Vision initialized with model: {self.model_name}")

        except Exception as e:
            self._init_error = str(e)
            logging.error(f"Failed to initialize Gemini Vision: {e}")

    def is_available(self) -> bool:
        """Check if Gemini Vision is available."""
        if not GENAI_AVAILABLE:
            return False
        if not self._initialized:
            self._initialize()
        return self._initialized and self._model is not None

    async def process_image(self, image_path: str, **options) -> OCRResult:
        """
        Process an image with Gemini Vision.

        Args:
            image_path: Path to the image file
            **options:
                prompt: Custom OCR prompt
                model: Override model

        Returns:
            OCRResult with extracted text
        """
        if not os.path.exists(image_path):
            return self._create_error_result(f"Image not found: {image_path}")

        if not self.is_available():
            error = self._init_error or "Gemini Vision not available"
            return self._create_error_result(error)

        with Timer() as timer:
            try:
                # Read and encode image
                with open(image_path, "rb") as f:
                    image_data = f.read()

                # Determine MIME type
                ext = os.path.splitext(image_path)[1].lower()
                mime_types = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }
                mime_type = mime_types.get(ext, "image/jpeg")

                # Create image part
                image_part = {
                    "mime_type": mime_type,
                    "data": image_data
                }

                # Get prompt
                prompt = options.get("prompt", self.OCR_PROMPT)

                # Generate content
                response = self._model.generate_content([prompt, image_part])

                # Extract text
                text = response.text.strip()

                # Estimate tokens and cost
                # Rough estimate: images ~250-500 tokens, output varies
                input_tokens = 500  # image + prompt estimate
                output_tokens = len(text.split()) * 1.3  # rough token estimate
                estimated_cost = (
                    (input_tokens / 1_000_000) * self.INPUT_COST_PER_1M +
                    (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M
                )

                return self._create_success_result(
                    text=text,
                    processing_time_ms=timer.elapsed_ms,
                    confidence=0.95,  # LLMs don't provide confidence scores
                    estimated_cost_usd=estimated_cost,
                    tokens_used=int(input_tokens + output_tokens),
                    raw_response={"model": self.model_name}
                )

            except Exception as e:
                logging.error(f"Gemini Vision processing error: {e}")
                return self._create_error_result(str(e), timer.elapsed_ms)

    def get_supported_languages(self) -> List[str]:
        """Return list of supported languages (Gemini supports many)."""
        return ["en", "he", "ar", "zh", "ja", "ko", "fr", "de", "es", "it", "pt", "ru"]

    def estimate_cost(self, num_pages: int = 1) -> float:
        """Estimate cost for processing pages."""
        # ~$0.001 per image
        return num_pages * 0.001


# CLI interface for testing
if __name__ == "__main__":
    import sys

    def print_help():
        print("Gemini Vision OCR Provider")
        print("")
        print("Usage:")
        print("  python -m ocr.providers.gemini_vision test <image>  - Test OCR")
        print("  python -m ocr.providers.gemini_vision check         - Check availability")
        print("")
        print("Environment:")
        print("  GEMINI_API_KEY or GOOGLE_API_KEY - API key")
        print("  Or use keyring: keyring set gemini api_key")

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "check":
        print(f"google-generativeai installed: {GENAI_AVAILABLE}")
        print(f"API key found: {get_gemini_api_key() is not None}")
        if GENAI_AVAILABLE:
            provider = GeminiVisionOCR()
            print(f"Provider available: {provider.is_available()}")
            if provider._init_error:
                print(f"Init error: {provider._init_error}")
        sys.exit(0)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Error: Please provide an image path")
            sys.exit(1)

        image_path = sys.argv[2]

        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        print(f"Testing Gemini Vision on: {image_path}")
        print("")

        provider = GeminiVisionOCR()

        if not provider.is_available():
            print(f"Error: Gemini Vision not available")
            if provider._init_error:
                print(f"Reason: {provider._init_error}")
            sys.exit(1)

        result = provider.process_image_sync(image_path)

        if result.success:
            print("SUCCESS!")
            print(f"Processing time: {result.processing_time_ms:.0f}ms")
            print(f"Words detected: {result.word_count}")
            print(f"Est. cost: ${result.estimated_cost_usd:.6f}")
            print("")
            print("Extracted text:")
            print("-" * 50)
            print(result.text[:2000] + ("..." if len(result.text) > 2000 else ""))
            sys.exit(0)
        else:
            print(f"FAILED: {result.error}")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        print_help()
        sys.exit(1)
