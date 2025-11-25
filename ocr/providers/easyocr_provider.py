"""
EasyOCR Provider

PyTorch-based OCR with excellent multilingual support.
https://github.com/JaidedAI/EasyOCR

License: Apache 2.0 - Commercial use allowed
"""

import logging
import os
from typing import List, Optional, Dict, Any

from ..base import OCRProvider, OCRResult, BoundingBox, ProviderType, Timer
from ..registry import register_provider

# Check if EasyOCR is available
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.info("EasyOCR not installed. Install with: pip install easyocr")


@register_provider
class EasyOCRProvider(OCRProvider):
    """
    EasyOCR Provider

    Features:
    - 80+ language support
    - GPU acceleration (optional)
    - Good accuracy for printed and handwritten text
    - Bounding box detection
    """

    PROVIDER_NAME = "easyocr"
    PROVIDER_TYPE = ProviderType.LOCAL
    COMMERCIAL_USE = True
    LICENSE = "Apache-2.0"

    # Common language codes (EasyOCR uses different codes than ISO)
    SUPPORTED_LANGUAGES = [
        "en",    # English
        "he",    # Hebrew
        "ar",    # Arabic
        "fr",    # French
        "de",    # German
        "es",    # Spanish
        "it",    # Italian
        "pt",    # Portuguese
        "ru",    # Russian
        "zh_sim",  # Chinese Simplified
        "zh_tra",  # Chinese Traditional
        "ja",    # Japanese
        "ko",    # Korean
        # Many more available - see EasyOCR docs
    ]

    def __init__(
        self,
        languages: List[str] = None,
        gpu: bool = True,
        model_storage_directory: str = None,
        download_enabled: bool = True,
        **config
    ):
        """
        Initialize EasyOCR provider.

        Args:
            languages: List of language codes (default: ['en'])
            gpu: Whether to use GPU acceleration
            model_storage_directory: Custom directory for model storage
            download_enabled: Allow automatic model download
        """
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.model_storage_directory = model_storage_directory
        self.download_enabled = download_enabled
        self.reader = None
        self._initialized = False
        self._init_error = None

        if EASYOCR_AVAILABLE:
            self._initialize_reader()

    def _initialize_reader(self):
        """Initialize the EasyOCR reader (lazy loading)."""
        if self._initialized:
            return

        try:
            kwargs = {
                "lang_list": self.languages,
                "gpu": self.gpu,
                "download_enabled": self.download_enabled,
            }

            if self.model_storage_directory:
                kwargs["model_storage_directory"] = self.model_storage_directory

            self.reader = easyocr.Reader(**kwargs)
            self._initialized = True
            logging.info(
                f"EasyOCR initialized with languages: {self.languages}, GPU: {self.gpu}"
            )
        except Exception as e:
            self._init_error = str(e)
            logging.error(f"Failed to initialize EasyOCR: {e}")

    def is_available(self) -> bool:
        """Check if EasyOCR is available and initialized."""
        if not EASYOCR_AVAILABLE:
            return False
        if not self._initialized:
            self._initialize_reader()
        return self._initialized and self.reader is not None

    async def process_image(self, image_path: str, **options) -> OCRResult:
        """
        Process an image with EasyOCR.

        Args:
            image_path: Path to the image file
            **options:
                detail: Level of detail (0=text only, 1=with boxes)
                paragraph: Combine text into paragraphs
                min_size: Minimum text size to detect
                contrast_ths: Contrast threshold
                adjust_contrast: Auto-adjust contrast
                text_threshold: Text confidence threshold
                low_text: Low text confidence threshold

        Returns:
            OCRResult with extracted text
        """
        if not os.path.exists(image_path):
            return self._create_error_result(f"Image not found: {image_path}")

        if not self.is_available():
            error = self._init_error or "EasyOCR not available"
            return self._create_error_result(error)

        with Timer() as timer:
            try:
                # Extract options with defaults
                detail = options.get("detail", 1)
                paragraph = options.get("paragraph", False)
                min_size = options.get("min_size", 10)
                text_threshold = options.get("text_threshold", 0.7)
                low_text = options.get("low_text", 0.4)

                # Run OCR
                results = self.reader.readtext(
                    image_path,
                    detail=detail,
                    paragraph=paragraph,
                    min_size=min_size,
                    text_threshold=text_threshold,
                    low_text=low_text,
                )

                # Process results
                if detail == 0:
                    # Simple text list
                    text = "\n".join(results)
                    bounding_boxes = None
                    avg_confidence = 0.0
                else:
                    # Results with bounding boxes: [(bbox, text, confidence), ...]
                    text_parts = []
                    bounding_boxes = []
                    confidences = []

                    for item in results:
                        bbox_points, detected_text, confidence = item

                        text_parts.append(detected_text)
                        confidences.append(confidence)

                        # Convert bbox points to x, y, width, height
                        # EasyOCR returns 4 corner points
                        if len(bbox_points) >= 4:
                            xs = [p[0] for p in bbox_points]
                            ys = [p[1] for p in bbox_points]
                            x = int(min(xs))
                            y = int(min(ys))
                            width = int(max(xs) - x)
                            height = int(max(ys) - y)

                            bounding_boxes.append(BoundingBox(
                                x=x,
                                y=y,
                                width=width,
                                height=height,
                                text=detected_text,
                                confidence=confidence
                            ))

                    text = "\n".join(text_parts)
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                return self._create_success_result(
                    text=text,
                    processing_time_ms=timer.elapsed_ms,
                    confidence=avg_confidence,
                    bounding_boxes=bounding_boxes if bounding_boxes else None,
                    raw_response=results
                )

            except Exception as e:
                logging.error(f"EasyOCR processing error: {e}")
                return self._create_error_result(str(e), timer.elapsed_ms)

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return self.SUPPORTED_LANGUAGES

    def set_languages(self, languages: List[str]):
        """
        Change the language configuration.

        Note: This requires reinitializing the reader.

        Args:
            languages: List of language codes
        """
        self.languages = languages
        self._initialized = False
        self.reader = None
        self._initialize_reader()


# CLI interface for testing
if __name__ == "__main__":
    import sys

    def print_help():
        print("EasyOCR Provider")
        print("")
        print("Usage:")
        print("  python -m ocr.providers.easyocr_provider test <image>  - Test OCR on image")
        print("  python -m ocr.providers.easyocr_provider check         - Check availability")
        print("  python -m ocr.providers.easyocr_provider langs         - List languages")
        print("")
        print("Options:")
        print("  --lang <code>    Add language (can repeat, default: en)")
        print("  --no-gpu         Disable GPU acceleration")
        print("")
        print("Examples:")
        print("  python -m ocr.providers.easyocr_provider test doc.jpg")
        print("  python -m ocr.providers.easyocr_provider test doc.jpg --lang en --lang he")

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "check":
        print(f"EasyOCR installed: {EASYOCR_AVAILABLE}")
        if EASYOCR_AVAILABLE:
            provider = EasyOCRProvider(gpu=False)
            print(f"Provider available: {provider.is_available()}")
        sys.exit(0)

    elif command == "langs":
        print("Commonly supported languages:")
        for lang in EasyOCRProvider.SUPPORTED_LANGUAGES:
            print(f"  {lang}")
        print("\nSee EasyOCR docs for full list: https://www.jaided.ai/easyocr/")
        sys.exit(0)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Error: Please provide an image path")
            sys.exit(1)

        image_path = sys.argv[2]

        # Parse optional arguments
        languages = []
        use_gpu = True
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--lang" and i + 1 < len(sys.argv):
                languages.append(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--no-gpu":
                use_gpu = False
                i += 1
            else:
                i += 1

        if not languages:
            languages = ["en"]

        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        print(f"Testing EasyOCR on: {image_path}")
        print(f"Languages: {languages}")
        print(f"GPU: {use_gpu}")
        print("")

        provider = EasyOCRProvider(languages=languages, gpu=use_gpu)

        if not provider.is_available():
            print("Error: EasyOCR not available")
            sys.exit(1)

        result = provider.process_image_sync(image_path)

        if result.success:
            print("SUCCESS!")
            print(f"Processing time: {result.processing_time_ms:.0f}ms")
            print(f"Words detected: {result.word_count}")
            print(f"Confidence: {result.confidence:.1%}")
            print("")
            print("Extracted text:")
            print("-" * 50)
            print(result.text[:1000] + ("..." if len(result.text) > 1000 else ""))
            sys.exit(0)
        else:
            print(f"FAILED: {result.error}")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        print_help()
        sys.exit(1)
