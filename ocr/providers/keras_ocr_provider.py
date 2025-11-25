"""
keras-ocr Provider

Lightweight OCR using CRAFT text detector and Keras CRNN recognition.
https://github.com/faustomorales/keras-ocr

License: MIT - Commercial use allowed
"""

import logging
import os
from typing import List, Optional

from ..base import OCRProvider, OCRResult, BoundingBox, ProviderType, Timer
from ..registry import register_provider

# Check if keras-ocr is available
try:
    import keras_ocr
    KERAS_OCR_AVAILABLE = True
except ImportError:
    KERAS_OCR_AVAILABLE = False
    logging.info("keras-ocr not installed. Install with: pip install keras-ocr")


@register_provider
class KerasOCRProvider(OCRProvider):
    """
    keras-ocr Provider

    Features:
    - CRAFT text detector (character-level)
    - CRNN text recognizer
    - Lightweight and fast on CPU
    - Good for scene text (signs, labels, etc.)
    """

    PROVIDER_NAME = "keras_ocr"
    PROVIDER_TYPE = ProviderType.LOCAL
    COMMERCIAL_USE = True
    LICENSE = "MIT"

    # Primarily supports Latin scripts
    SUPPORTED_LANGUAGES = [
        "en",    # English (primary)
        # Latin-script languages
    ]

    def __init__(self, **config):
        """
        Initialize keras-ocr provider.

        Args:
            **config: Additional configuration options
        """
        self.pipeline = None
        self._initialized = False
        self._init_error = None

        if KERAS_OCR_AVAILABLE:
            self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize keras-ocr pipeline (lazy loading)."""
        if self._initialized:
            return

        try:
            # keras-ocr downloads models automatically on first use
            self.pipeline = keras_ocr.pipeline.Pipeline()
            self._initialized = True
            logging.info("keras-ocr pipeline initialized")
        except Exception as e:
            self._init_error = str(e)
            logging.error(f"Failed to initialize keras-ocr: {e}")

    def is_available(self) -> bool:
        """Check if keras-ocr is available and initialized."""
        if not KERAS_OCR_AVAILABLE:
            return False
        if not self._initialized:
            self._initialize_pipeline()
        return self._initialized and self.pipeline is not None

    async def process_image(self, image_path: str, **options) -> OCRResult:
        """
        Process an image with keras-ocr.

        Args:
            image_path: Path to the image file
            **options: Additional options (currently unused)

        Returns:
            OCRResult with extracted text
        """
        if not os.path.exists(image_path):
            return self._create_error_result(f"Image not found: {image_path}")

        if not self.is_available():
            error = self._init_error or "keras-ocr not available"
            return self._create_error_result(error)

        with Timer() as timer:
            try:
                # Read image using keras-ocr's utility
                image = keras_ocr.tools.read(image_path)

                # Run pipeline (returns list of (word, box) tuples for each image)
                prediction_groups = self.pipeline.recognize([image])

                # Process results (we only passed one image)
                predictions = prediction_groups[0]

                text_parts = []
                bounding_boxes = []

                for word, box in predictions:
                    text_parts.append(word)

                    # box is array of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    if len(box) >= 4:
                        xs = [p[0] for p in box]
                        ys = [p[1] for p in box]
                        x = int(min(xs))
                        y = int(min(ys))
                        width = int(max(xs) - x)
                        height = int(max(ys) - y)

                        bounding_boxes.append(BoundingBox(
                            x=x,
                            y=y,
                            width=width,
                            height=height,
                            text=word,
                            confidence=1.0  # keras-ocr doesn't provide confidence
                        ))

                # Join words with spaces (keras-ocr detects words, not lines)
                text = " ".join(text_parts)

                return self._create_success_result(
                    text=text,
                    processing_time_ms=timer.elapsed_ms,
                    confidence=1.0,
                    bounding_boxes=bounding_boxes if bounding_boxes else None,
                    raw_response=predictions
                )

            except Exception as e:
                logging.error(f"keras-ocr processing error: {e}")
                return self._create_error_result(str(e), timer.elapsed_ms)

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return self.SUPPORTED_LANGUAGES


# CLI interface for testing
if __name__ == "__main__":
    import sys

    def print_help():
        print("keras-ocr Provider")
        print("")
        print("Usage:")
        print("  python -m ocr.providers.keras_ocr_provider test <image>  - Test OCR")
        print("  python -m ocr.providers.keras_ocr_provider check         - Check availability")
        print("")
        print("Examples:")
        print("  python -m ocr.providers.keras_ocr_provider test document.jpg")

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "check":
        print(f"keras-ocr installed: {KERAS_OCR_AVAILABLE}")
        if KERAS_OCR_AVAILABLE:
            print("Initializing pipeline (will download models on first run)...")
            provider = KerasOCRProvider()
            print(f"Provider available: {provider.is_available()}")
        sys.exit(0)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Error: Please provide an image path")
            sys.exit(1)

        image_path = sys.argv[2]

        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        print(f"Testing keras-ocr on: {image_path}")
        print("Initializing (downloading models on first run)...")
        print("")

        provider = KerasOCRProvider()

        if not provider.is_available():
            print("Error: keras-ocr not available")
            sys.exit(1)

        result = provider.process_image_sync(image_path)

        if result.success:
            print("SUCCESS!")
            print(f"Processing time: {result.processing_time_ms:.0f}ms")
            print(f"Words detected: {result.word_count}")
            if result.bounding_boxes:
                print(f"Regions detected: {len(result.bounding_boxes)}")
            print("")
            print("Extracted text:")
            print("-" * 50)
            print(result.text[:1500] + ("..." if len(result.text) > 1500 else ""))
            sys.exit(0)
        else:
            print(f"FAILED: {result.error}")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        print_help()
        sys.exit(1)
