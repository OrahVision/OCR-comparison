"""
Tesseract OCR Provider

The classic open-source OCR engine.
https://github.com/tesseract-ocr/tesseract

License: Apache 2.0 - Commercial use allowed
Requires: tesseract-ocr installed on system + pytesseract
"""

import logging
import os
from typing import List, Optional

from ..base import OCRProvider, OCRResult, BoundingBox, ProviderType, Timer
from ..registry import register_provider

# Check if pytesseract is available
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True

    # Set Tesseract path for Windows if not in PATH
    if os.name == 'nt':
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break

    # Try to verify tesseract is installed
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        TESSERACT_AVAILABLE = False
        logging.info("Tesseract not found. Install from: https://github.com/tesseract-ocr/tesseract")

except ImportError:
    TESSERACT_AVAILABLE = False
    logging.info("pytesseract not installed. Install with: pip install pytesseract pillow")


@register_provider
class TesseractProvider(OCRProvider):
    """
    Tesseract OCR Provider

    Features:
    - 100+ language support
    - Mature, well-tested engine
    - Good for clean document scans
    - Bounding box support
    """

    PROVIDER_NAME = "tesseract"
    PROVIDER_TYPE = ProviderType.LOCAL
    COMMERCIAL_USE = True
    LICENSE = "Apache-2.0"

    SUPPORTED_LANGUAGES = [
        "eng",    # English
        "heb",    # Hebrew
        "ara",    # Arabic
        "fra",    # French
        "deu",    # German
        "spa",    # Spanish
        "ita",    # Italian
        "por",    # Portuguese
        "rus",    # Russian
        "chi_sim",  # Chinese Simplified
        "chi_tra",  # Chinese Traditional
        "jpn",    # Japanese
        "kor",    # Korean
        # Many more - run 'tesseract --list-langs'
    ]

    def __init__(
        self,
        lang: str = "eng",
        config: str = "",
        tesseract_cmd: str = None,
        **kwargs
    ):
        """
        Initialize Tesseract provider.

        Args:
            lang: Language code (default: 'eng')
            config: Additional tesseract config options
            tesseract_cmd: Path to tesseract executable (if not in PATH)
        """
        self.lang = lang
        self.config = config
        self._initialized = False
        self._init_error = None

        if tesseract_cmd and TESSERACT_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        if TESSERACT_AVAILABLE:
            self._initialize()

    def _initialize(self):
        """Verify tesseract is working."""
        if self._initialized:
            return

        try:
            version = pytesseract.get_tesseract_version()
            self._initialized = True
            logging.info(f"Tesseract initialized, version: {version}")
        except Exception as e:
            self._init_error = str(e)
            logging.error(f"Failed to initialize Tesseract: {e}")

    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        if not TESSERACT_AVAILABLE:
            return False
        if not self._initialized:
            self._initialize()
        return self._initialized

    async def process_image(self, image_path: str, **options) -> OCRResult:
        """
        Process an image with Tesseract.

        Args:
            image_path: Path to the image file
            **options:
                lang: Override language
                config: Override config
                psm: Page segmentation mode (0-13)
                oem: OCR engine mode (0-3)

        Returns:
            OCRResult with extracted text
        """
        if not os.path.exists(image_path):
            return self._create_error_result(f"Image not found: {image_path}")

        if not self.is_available():
            error = self._init_error or "Tesseract not available"
            return self._create_error_result(error)

        with Timer() as timer:
            try:
                # Load image
                image = Image.open(image_path)

                # Build config string
                lang = options.get('lang', self.lang)
                config = options.get('config', self.config)

                psm = options.get('psm')
                if psm is not None:
                    config += f" --psm {psm}"

                oem = options.get('oem')
                if oem is not None:
                    config += f" --oem {oem}"

                # Run OCR
                text = pytesseract.image_to_string(image, lang=lang, config=config)

                # Get detailed data with confidence
                data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)

                # Process bounding boxes and confidence
                bounding_boxes = []
                confidences = []

                n_boxes = len(data['text'])
                for i in range(n_boxes):
                    word = data['text'][i].strip()
                    conf = int(data['conf'][i])

                    if word and conf > 0:  # Skip empty and low-confidence
                        confidences.append(conf / 100.0)

                        bounding_boxes.append(BoundingBox(
                            x=data['left'][i],
                            y=data['top'][i],
                            width=data['width'][i],
                            height=data['height'][i],
                            text=word,
                            confidence=conf / 100.0
                        ))

                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                image.close()

                return self._create_success_result(
                    text=text.strip(),
                    processing_time_ms=timer.elapsed_ms,
                    confidence=avg_confidence,
                    bounding_boxes=bounding_boxes if bounding_boxes else None,
                    raw_response=data
                )

            except Exception as e:
                logging.error(f"Tesseract processing error: {e}")
                return self._create_error_result(str(e), timer.elapsed_ms)

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return self.SUPPORTED_LANGUAGES

    def get_installed_languages(self) -> List[str]:
        """Get list of actually installed language packs."""
        if not self.is_available():
            return []

        try:
            langs = pytesseract.get_languages()
            return langs
        except Exception:
            return self.SUPPORTED_LANGUAGES


# CLI interface for testing
if __name__ == "__main__":
    import sys

    def print_help():
        print("Tesseract OCR Provider")
        print("")
        print("Usage:")
        print("  python -m ocr.providers.tesseract_provider test <image>  - Test OCR")
        print("  python -m ocr.providers.tesseract_provider check         - Check availability")
        print("  python -m ocr.providers.tesseract_provider langs         - List languages")
        print("")
        print("Options:")
        print("  --lang <code>    Set language (default: eng)")
        print("")
        print("Examples:")
        print("  python -m ocr.providers.tesseract_provider test document.jpg")
        print("  python -m ocr.providers.tesseract_provider test hebrew.jpg --lang heb")

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "check":
        print(f"pytesseract installed: {TESSERACT_AVAILABLE}")
        if TESSERACT_AVAILABLE:
            try:
                version = pytesseract.get_tesseract_version()
                print(f"Tesseract version: {version}")
                provider = TesseractProvider()
                print(f"Provider available: {provider.is_available()}")
            except Exception as e:
                print(f"Error: {e}")
        sys.exit(0)

    elif command == "langs":
        if TESSERACT_AVAILABLE:
            provider = TesseractProvider()
            installed = provider.get_installed_languages()
            print("Installed languages:")
            for lang in installed:
                print(f"  {lang}")
        else:
            print("Tesseract not available")
        sys.exit(0)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Error: Please provide an image path")
            sys.exit(1)

        image_path = sys.argv[2]

        # Parse optional arguments
        lang = "eng"
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--lang" and i + 1 < len(sys.argv):
                lang = sys.argv[i + 1]
                i += 2
            else:
                i += 1

        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        print(f"Testing Tesseract on: {image_path}")
        print(f"Language: {lang}")
        print("")

        provider = TesseractProvider(lang=lang)

        if not provider.is_available():
            print("Error: Tesseract not available")
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
