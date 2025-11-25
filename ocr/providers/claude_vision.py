"""
Claude Vision OCR Provider

Uses Anthropic's Claude models for OCR via vision capabilities.
https://docs.anthropic.com/claude/docs/vision

License: Commercial API
Pricing:
  - Claude 3 Haiku: $0.25/1M input, $1.25/1M output (~$0.0004/image)
  - Claude 3.5 Sonnet: $3/1M input, $15/1M output (~$0.005/image)
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

# Check for anthropic SDK
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.info("anthropic not installed. Install with: pip install anthropic")


def get_anthropic_api_key() -> Optional[str]:
    """Get Anthropic API key from keyring or environment."""
    # Try keyring first
    if KEYRING_AVAILABLE:
        try:
            key = keyring.get_password("anthropic", "api_key")
            if key:
                return key
        except Exception:
            pass

    # Try environment variable
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key

    return None


CLAUDE_AVAILABLE = ANTHROPIC_AVAILABLE and get_anthropic_api_key() is not None


@register_provider
class ClaudeVisionOCR(OCRProvider):
    """
    Claude Vision OCR Provider

    Features:
    - Multiple model options (Haiku for speed/cost, Sonnet for accuracy)
    - Excellent multilingual support
    - Strong at complex layouts and handwriting
    - Good at understanding context
    """

    PROVIDER_NAME = "claude_vision"
    PROVIDER_TYPE = ProviderType.LLM
    COMMERCIAL_USE = True
    LICENSE = "Commercial API"

    # Available models and their pricing (per 1M tokens)
    MODELS = {
        "claude-3-haiku-20240307": {
            "name": "Claude 3 Haiku",
            "input_cost": 0.25,
            "output_cost": 1.25,
        },
        "claude-3-5-haiku-20241022": {
            "name": "Claude 3.5 Haiku",
            "input_cost": 1.00,
            "output_cost": 5.00,
        },
        "claude-3-5-sonnet-20241022": {
            "name": "Claude 3.5 Sonnet",
            "input_cost": 3.00,
            "output_cost": 15.00,
        },
        "claude-sonnet-4-20250514": {
            "name": "Claude Sonnet 4",
            "input_cost": 3.00,
            "output_cost": 15.00,
        },
    }

    DEFAULT_MODEL = "claude-3-haiku-20240307"  # Cheapest option

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
        Initialize Claude Vision OCR provider.

        Args:
            model: Claude model to use (default: claude-3-haiku)
            api_key: API key (defaults to keyring/env)
        """
        self.model_name = model or self.DEFAULT_MODEL
        self._api_key = api_key
        self._initialized = False
        self._init_error = None
        self._client = None

        if ANTHROPIC_AVAILABLE:
            self._initialize()

    def _initialize(self):
        """Initialize the Anthropic client."""
        if self._initialized:
            return

        try:
            api_key = self._api_key or get_anthropic_api_key()
            if not api_key:
                self._init_error = "No Anthropic API key found. Set via keyring (anthropic/api_key) or ANTHROPIC_API_KEY env var."
                return

            self._client = anthropic.Anthropic(api_key=api_key)
            self._initialized = True
            logging.info(f"Claude Vision initialized with model: {self.model_name}")

        except Exception as e:
            self._init_error = str(e)
            logging.error(f"Failed to initialize Claude Vision: {e}")

    def is_available(self) -> bool:
        """Check if Claude Vision is available."""
        if not ANTHROPIC_AVAILABLE:
            return False
        if not self._initialized:
            self._initialize()
        return self._initialized and self._client is not None

    async def process_image(self, image_path: str, **options) -> OCRResult:
        """
        Process an image with Claude Vision.

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
            error = self._init_error or "Claude Vision not available"
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

                # Get prompt and model
                prompt = options.get("prompt", self.OCR_PROMPT)
                model = options.get("model", self.model_name)

                # Create message
                message = self._client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_data,
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
                text = message.content[0].text.strip()

                # Get token usage and calculate cost
                input_tokens = message.usage.input_tokens
                output_tokens = message.usage.output_tokens

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
                logging.error(f"Claude Vision processing error: {e}")
                return self._create_error_result(str(e), timer.elapsed_ms)

    def get_supported_languages(self) -> List[str]:
        """Return list of supported languages (Claude supports many)."""
        return ["en", "he", "ar", "zh", "ja", "ko", "fr", "de", "es", "it", "pt", "ru"]

    def estimate_cost(self, num_pages: int = 1) -> float:
        """Estimate cost for processing pages."""
        model_info = self.MODELS.get(self.model_name, self.MODELS[self.DEFAULT_MODEL])
        # Estimate ~1500 input tokens per image, ~500 output
        per_image = (1500 / 1_000_000) * model_info["input_cost"] + (500 / 1_000_000) * model_info["output_cost"]
        return num_pages * per_image


# CLI interface for testing
if __name__ == "__main__":
    import sys

    def print_help():
        print("Claude Vision OCR Provider")
        print("")
        print("Usage:")
        print("  python -m ocr.providers.claude_vision test <image>  - Test OCR")
        print("  python -m ocr.providers.claude_vision check         - Check availability")
        print("  python -m ocr.providers.claude_vision models        - List models")
        print("")
        print("Options:")
        print("  --model <name>   Use specific model (default: claude-3-haiku)")
        print("")
        print("Environment:")
        print("  ANTHROPIC_API_KEY - API key")
        print("  Or use keyring: keyring set anthropic api_key")

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "check":
        print(f"anthropic SDK installed: {ANTHROPIC_AVAILABLE}")
        print(f"API key found: {get_anthropic_api_key() is not None}")
        if ANTHROPIC_AVAILABLE:
            provider = ClaudeVisionOCR()
            print(f"Provider available: {provider.is_available()}")
            if provider._init_error:
                print(f"Init error: {provider._init_error}")
        sys.exit(0)

    elif command == "models":
        print("Available Claude models:")
        for model_id, info in ClaudeVisionOCR.MODELS.items():
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

        print(f"Testing Claude Vision on: {image_path}")
        if model:
            print(f"Model: {model}")
        print("")

        provider = ClaudeVisionOCR(model=model) if model else ClaudeVisionOCR()

        if not provider.is_available():
            print(f"Error: Claude Vision not available")
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
