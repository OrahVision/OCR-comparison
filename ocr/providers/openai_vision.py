"""
OpenAI Vision OCR Provider

Uses OpenAI's GPT-4o models for OCR via vision capabilities.
https://platform.openai.com/docs/guides/vision

License: Commercial API
Pricing:
  - GPT-4o-mini: $0.15/1M input, $0.60/1M output (~$0.001/image)
  - GPT-4o: $2.50/1M input, $10/1M output (~$0.008/image)
"""

import base64
import logging
import os
from typing import List, Optional

from ..base import OCRProvider, OCRResult, ProviderType, Timer
from ..registry import register_provider

# Check for keyring
try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

# Check for openai SDK
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.info("openai not installed. Install with: pip install openai")


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from keyring or environment."""
    # Try keyring first
    if KEYRING_AVAILABLE:
        try:
            key = keyring.get_password("openai", "api_key")
            if key:
                return key
        except Exception:
            pass

    # Try environment variable
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    return None


OPENAI_VISION_AVAILABLE = OPENAI_AVAILABLE and get_openai_api_key() is not None


@register_provider
class OpenAIVisionOCR(OCRProvider):
    """
    OpenAI Vision OCR Provider

    Features:
    - Multiple model options (GPT-4o-mini for speed/cost, GPT-4o for accuracy)
    - Excellent multilingual support
    - Strong at complex layouts
    - Good reasoning about document structure
    """

    PROVIDER_NAME = "openai_vision"
    PROVIDER_TYPE = ProviderType.LLM
    COMMERCIAL_USE = True
    LICENSE = "Commercial API"

    # Available models and their pricing (per 1M tokens)
    MODELS = {
        "gpt-4o-mini": {
            "name": "GPT-4o Mini",
            "input_cost": 0.15,
            "output_cost": 0.60,
        },
        "gpt-4o": {
            "name": "GPT-4o",
            "input_cost": 2.50,
            "output_cost": 10.00,
        },
        "gpt-4-turbo": {
            "name": "GPT-4 Turbo",
            "input_cost": 10.00,
            "output_cost": 30.00,
        },
    }

    DEFAULT_MODEL = "gpt-4o-mini"  # Best price/performance

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
        Initialize OpenAI Vision OCR provider.

        Args:
            model: OpenAI model to use (default: gpt-4o-mini)
            api_key: API key (defaults to keyring/env)
        """
        self.model_name = model or self.DEFAULT_MODEL
        self._api_key = api_key
        self._initialized = False
        self._init_error = None
        self._client = None

        if OPENAI_AVAILABLE:
            self._initialize()

    def _initialize(self):
        """Initialize the OpenAI client."""
        if self._initialized:
            return

        try:
            api_key = self._api_key or get_openai_api_key()
            if not api_key:
                self._init_error = "No OpenAI API key found. Set via keyring (openai/api_key) or OPENAI_API_KEY env var."
                return

            self._client = openai.OpenAI(api_key=api_key)
            self._initialized = True
            logging.info(f"OpenAI Vision initialized with model: {self.model_name}")

        except Exception as e:
            self._init_error = str(e)
            logging.error(f"Failed to initialize OpenAI Vision: {e}")

    def is_available(self) -> bool:
        """Check if OpenAI Vision is available."""
        if not OPENAI_AVAILABLE:
            return False
        if not self._initialized:
            self._initialize()
        return self._initialized and self._client is not None

    async def process_image(self, image_path: str, **options) -> OCRResult:
        """
        Process an image with OpenAI Vision.

        Args:
            image_path: Path to the image file
            **options:
                prompt: Custom OCR prompt
                model: Override model
                detail: Image detail level ('low', 'high', 'auto')

        Returns:
            OCRResult with extracted text
        """
        if not os.path.exists(image_path):
            return self._create_error_result(f"Image not found: {image_path}")

        if not self.is_available():
            error = self._init_error or "OpenAI Vision not available"
            return self._create_error_result(error)

        with Timer() as timer:
            try:
                # Read and encode image
                with open(image_path, "rb") as f:
                    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

                # Determine media type
                ext = os.path.splitext(image_path)[1].lower()
                media_types = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }
                media_type = media_types.get(ext, "image/jpeg")

                # Get options
                prompt = options.get("prompt", self.OCR_PROMPT)
                model = options.get("model", self.model_name)
                detail = options.get("detail", "auto")

                # Create message
                response = self._client.chat.completions.create(
                    model=model,
                    max_tokens=4096,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{media_type};base64,{image_data}",
                                        "detail": detail,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": prompt,
                                }
                            ],
                        }
                    ],
                )

                # Extract text
                text = response.choices[0].message.content.strip()

                # Get token usage and calculate cost
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

                model_info = self.MODELS.get(model, self.MODELS[self.DEFAULT_MODEL])
                estimated_cost = (
                    (input_tokens / 1_000_000) * model_info["input_cost"] +
                    (output_tokens / 1_000_000) * model_info["output_cost"]
                )

                return self._create_success_result(
                    text=text,
                    processing_time_ms=timer.elapsed_ms,
                    confidence=0.95,  # LLMs don't provide confidence scores
                    estimated_cost_usd=estimated_cost,
                    tokens_used=input_tokens + output_tokens,
                    raw_response={
                        "model": model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                )

            except Exception as e:
                logging.error(f"OpenAI Vision processing error: {e}")
                return self._create_error_result(str(e), timer.elapsed_ms)

    def get_supported_languages(self) -> List[str]:
        """Return list of supported languages (GPT-4 supports many)."""
        return ["en", "he", "ar", "zh", "ja", "ko", "fr", "de", "es", "it", "pt", "ru"]

    def estimate_cost(self, num_pages: int = 1) -> float:
        """Estimate cost for processing pages."""
        model_info = self.MODELS.get(self.model_name, self.MODELS[self.DEFAULT_MODEL])
        # Estimate ~1000 input tokens per image, ~500 output
        per_image = (1000 / 1_000_000) * model_info["input_cost"] + (500 / 1_000_000) * model_info["output_cost"]
        return num_pages * per_image


# CLI interface for testing
if __name__ == "__main__":
    import sys

    def print_help():
        print("OpenAI Vision OCR Provider")
        print("")
        print("Usage:")
        print("  python -m ocr.providers.openai_vision test <image>  - Test OCR")
        print("  python -m ocr.providers.openai_vision check         - Check availability")
        print("  python -m ocr.providers.openai_vision models        - List models")
        print("")
        print("Options:")
        print("  --model <name>   Use specific model (default: gpt-4o-mini)")
        print("")
        print("Environment:")
        print("  OPENAI_API_KEY - API key")
        print("  Or use keyring: keyring set openai api_key")

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "check":
        print(f"openai SDK installed: {OPENAI_AVAILABLE}")
        print(f"API key found: {get_openai_api_key() is not None}")
        if OPENAI_AVAILABLE:
            provider = OpenAIVisionOCR()
            print(f"Provider available: {provider.is_available()}")
            if provider._init_error:
                print(f"Init error: {provider._init_error}")
        sys.exit(0)

    elif command == "models":
        print("Available OpenAI models:")
        for model_id, info in OpenAIVisionOCR.MODELS.items():
            print(f"  {model_id}")
            print(f"    Name: {info['name']}")
            print(f"    Input: ${info['input_cost']}/1M tokens")
            print(f"    Output: ${info['output_cost']}/1M tokens")
            print()
        sys.exit(0)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Error: Please provide an image path")
            sys.exit(1)

        image_path = sys.argv[2]

        # Parse optional arguments
        model = None
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--model" and i + 1 < len(sys.argv):
                model = sys.argv[i + 1]
                i += 2
            else:
                i += 1

        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        print(f"Testing OpenAI Vision on: {image_path}")
        if model:
            print(f"Model: {model}")
        print("")

        provider = OpenAIVisionOCR(model=model) if model else OpenAIVisionOCR()

        if not provider.is_available():
            print(f"Error: OpenAI Vision not available")
            if provider._init_error:
                print(f"Reason: {provider._init_error}")
            sys.exit(1)

        result = provider.process_image_sync(image_path)

        if result.success:
            print("SUCCESS!")
            print(f"Processing time: {result.processing_time_ms:.0f}ms")
            print(f"Words detected: {result.word_count}")
            print(f"Tokens used: {result.tokens_used}")
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
