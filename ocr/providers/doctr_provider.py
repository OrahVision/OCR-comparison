"""
docTR Provider

Document Text Recognition by Mindee.
https://github.com/mindee/doctr

License: Apache 2.0 - Commercial use allowed
"""

import logging
import os
from typing import List, Optional, Dict, Any

from ..base import OCRProvider, OCRResult, BoundingBox, ProviderType, Timer
from ..registry import register_provider

# Check if docTR is available
DOCTR_AVAILABLE = False
DOCTR_BACKEND = None

try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True

    # Try to detect backend
    try:
        import torch
        DOCTR_BACKEND = "pytorch"
    except ImportError:
        try:
            import tensorflow
            DOCTR_BACKEND = "tensorflow"
        except ImportError:
            pass

except ImportError:
    logging.info(
        "docTR not installed. Install with: "
        "pip install python-doctr[torch] or pip install python-doctr[tf]"
    )


@register_provider
class DocTRProvider(OCRProvider):
    """
    docTR Provider

    Features:
    - Document-focused OCR
    - Supports PyTorch or TensorFlow backend
    - Good for structured documents (invoices, forms, etc.)
    - Word and line-level detection
    """

    PROVIDER_NAME = "doctr"
    PROVIDER_TYPE = ProviderType.LOCAL
    COMMERCIAL_USE = True
    LICENSE = "Apache-2.0"

    # docTR primarily supports Latin-based languages
    SUPPORTED_LANGUAGES = [
        "en",    # English
        "fr",    # French
        "de",    # German
        "es",    # Spanish
        "pt",    # Portuguese
        "it",    # Italian
        # Latin script languages
    ]

    def __init__(
        self,
        det_arch: str = "db_resnet50",
        reco_arch: str = "crnn_vgg16_bn",
        pretrained: bool = True,
        assume_straight_pages: bool = True,
        straighten_pages: bool = False,
        export_as_straight_boxes: bool = False,
        **config
    ):
        """
        Initialize docTR provider.

        Args:
            det_arch: Detection model architecture
            reco_arch: Recognition model architecture
            pretrained: Use pretrained models
            assume_straight_pages: Assume pages are not rotated
            straighten_pages: Apply page straightening
            export_as_straight_boxes: Export boxes as axis-aligned rectangles
        """
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self.pretrained = pretrained
        self.assume_straight_pages = assume_straight_pages
        self.straighten_pages = straighten_pages
        self.export_as_straight_boxes = export_as_straight_boxes

        self.predictor = None
        self._initialized = False
        self._init_error = None

        if DOCTR_AVAILABLE:
            self._initialize_predictor()

    def _initialize_predictor(self):
        """Initialize docTR predictor (lazy loading)."""
        if self._initialized:
            return

        try:
            self.predictor = ocr_predictor(
                det_arch=self.det_arch,
                reco_arch=self.reco_arch,
                pretrained=self.pretrained,
                assume_straight_pages=self.assume_straight_pages,
                straighten_pages=self.straighten_pages,
                export_as_straight_boxes=self.export_as_straight_boxes,
            )
            self._initialized = True
            logging.info(
                f"docTR initialized with det={self.det_arch}, reco={self.reco_arch}, "
                f"backend={DOCTR_BACKEND}"
            )
        except Exception as e:
            self._init_error = str(e)
            logging.error(f"Failed to initialize docTR: {e}")

    def is_available(self) -> bool:
        """Check if docTR is available and initialized."""
        if not DOCTR_AVAILABLE:
            return False
        if not self._initialized:
            self._initialize_predictor()
        return self._initialized and self.predictor is not None

    async def process_image(self, image_path: str, **options) -> OCRResult:
        """
        Process an image with docTR.

        Args:
            image_path: Path to the image file
            **options: Additional processing options

        Returns:
            OCRResult with extracted text
        """
        if not os.path.exists(image_path):
            return self._create_error_result(f"Image not found: {image_path}")

        if not self.is_available():
            error = self._init_error or "docTR not available"
            return self._create_error_result(error)

        with Timer() as timer:
            try:
                # Load document
                doc = DocumentFile.from_images(image_path)

                # Run OCR
                result = self.predictor(doc)

                # Process results
                text_parts = []
                bounding_boxes = []
                confidences = []

                # docTR returns structured output with pages > blocks > lines > words
                for page in result.pages:
                    page_height = page.dimensions[0]
                    page_width = page.dimensions[1]

                    for block in page.blocks:
                        block_lines = []

                        for line in block.lines:
                            line_words = []

                            for word in line.words:
                                line_words.append(word.value)
                                confidences.append(word.confidence)

                                # Convert relative coordinates to absolute
                                # docTR uses relative coords (0-1)
                                geo = word.geometry
                                if len(geo) >= 2:
                                    x = int(geo[0][0] * page_width)
                                    y = int(geo[0][1] * page_height)
                                    x2 = int(geo[1][0] * page_width)
                                    y2 = int(geo[1][1] * page_height)

                                    bounding_boxes.append(BoundingBox(
                                        x=x,
                                        y=y,
                                        width=x2 - x,
                                        height=y2 - y,
                                        text=word.value,
                                        confidence=word.confidence
                                    ))

                            if line_words:
                                block_lines.append(" ".join(line_words))

                        if block_lines:
                            text_parts.append("\n".join(block_lines))

                text = "\n\n".join(text_parts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                # Build page data for structured output
                pages = []
                for page in result.pages:
                    page_data = {
                        "dimensions": page.dimensions,
                        "blocks": []
                    }
                    for block in page.blocks:
                        block_data = {
                            "lines": []
                        }
                        for line in block.lines:
                            line_data = {
                                "words": [
                                    {"text": w.value, "confidence": w.confidence}
                                    for w in line.words
                                ]
                            }
                            block_data["lines"].append(line_data)
                        page_data["blocks"].append(block_data)
                    pages.append(page_data)

                return self._create_success_result(
                    text=text,
                    processing_time_ms=timer.elapsed_ms,
                    confidence=avg_confidence,
                    bounding_boxes=bounding_boxes if bounding_boxes else None,
                    pages=pages,
                    raw_response=result
                )

            except Exception as e:
                logging.error(f"docTR processing error: {e}")
                return self._create_error_result(str(e), timer.elapsed_ms)

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return self.SUPPORTED_LANGUAGES


# CLI interface for testing
if __name__ == "__main__":
    import sys

    def print_help():
        print("docTR Provider")
        print("")
        print("Usage:")
        print("  python -m ocr.providers.doctr_provider test <image>  - Test OCR")
        print("  python -m ocr.providers.doctr_provider check         - Check availability")
        print("")
        print("Examples:")
        print("  python -m ocr.providers.doctr_provider test document.jpg")

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "check":
        print(f"docTR installed: {DOCTR_AVAILABLE}")
        print(f"Backend: {DOCTR_BACKEND or 'none'}")
        if DOCTR_AVAILABLE:
            provider = DocTRProvider()
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

        print(f"Testing docTR on: {image_path}")
        print(f"Backend: {DOCTR_BACKEND or 'unknown'}")
        print("")

        provider = DocTRProvider()

        if not provider.is_available():
            print("Error: docTR not available")
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
