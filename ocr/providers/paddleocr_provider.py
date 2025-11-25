"""
PaddleOCR Provider

High-performance OCR from Baidu, excellent for CJK languages.
https://github.com/PaddlePaddle/PaddleOCR

License: Apache 2.0 - Commercial use allowed
Note: Avoid PyMuPDF dependency (AGPL) for commercial use - use pdf2image instead.
"""

import logging
import os
import tempfile
from typing import List, Optional, Dict, Any

from ..base import OCRProvider, OCRResult, BoundingBox, ProviderType, Timer
from ..registry import register_provider

# For image resizing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Check if PaddleOCR is available
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.info("PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle")


@register_provider
class PaddleOCRProvider(OCRProvider):
    """
    PaddleOCR Provider

    Features:
    - 80+ language support
    - Excellent CJK (Chinese, Japanese, Korean) recognition
    - Very fast inference
    - Angle detection and correction
    - PP-OCRv4 model (latest)
    """

    PROVIDER_NAME = "paddleocr"
    PROVIDER_TYPE = ProviderType.LOCAL
    COMMERCIAL_USE = True
    LICENSE = "Apache-2.0"

    # Language codes used by PaddleOCR
    SUPPORTED_LANGUAGES = [
        "en",      # English
        "ch",      # Chinese (simplified)
        "chinese_cht",  # Chinese (traditional)
        "japan",   # Japanese
        "korean",  # Korean
        "french",  # French
        "german",  # German
        "arabic",  # Arabic
        "cyrillic", # Russian and Cyrillic languages
        "devanagari", # Hindi
        "latin",   # Latin-based languages
        # Many more - see PaddleOCR docs
    ]

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        use_gpu: bool = True,
        show_log: bool = False,
        det_model_dir: str = None,
        rec_model_dir: str = None,
        cls_model_dir: str = None,
        **config
    ):
        """
        Initialize PaddleOCR provider.

        Args:
            lang: Language code (default: 'en')
            use_angle_cls: Enable angle classification for rotated text
            use_gpu: Use GPU acceleration
            show_log: Show PaddleOCR logs
            det_model_dir: Custom detection model directory
            rec_model_dir: Custom recognition model directory
            cls_model_dir: Custom classification model directory
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.show_log = show_log
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.cls_model_dir = cls_model_dir

        self.ocr = None
        self._initialized = False
        self._init_error = None

        if PADDLEOCR_AVAILABLE:
            self._initialize_ocr()

    def _initialize_ocr(self):
        """Initialize PaddleOCR engine (lazy loading)."""
        if self._initialized:
            return

        try:
            kwargs = {
                "lang": self.lang,
            }

            # New API uses use_textline_orientation instead of use_angle_cls
            if self.use_angle_cls:
                kwargs["use_textline_orientation"] = True

            # Model directories use new naming convention
            if self.det_model_dir:
                kwargs["text_detection_model_dir"] = self.det_model_dir
            if self.rec_model_dir:
                kwargs["text_recognition_model_dir"] = self.rec_model_dir

            self.ocr = PaddleOCR(**kwargs)
            self._initialized = True
            logging.info(
                f"PaddleOCR initialized with lang: {self.lang}, GPU: {self.use_gpu}"
            )
        except Exception as e:
            self._init_error = str(e)
            logging.error(f"Failed to initialize PaddleOCR: {e}")

    def is_available(self) -> bool:
        """Check if PaddleOCR is available and initialized."""
        if not PADDLEOCR_AVAILABLE:
            return False
        if not self._initialized:
            self._initialize_ocr()
        return self._initialized and self.ocr is not None

    def _resize_image_if_needed(self, image_path: str, max_size: int = 3000) -> str:
        """
        Resize image if it exceeds max_size on any dimension.
        Returns path to resized image (temp file) or original path.
        """
        if not PIL_AVAILABLE:
            return image_path

        try:
            img = Image.open(image_path)
            width, height = img.size

            # Check if resize needed
            if width <= max_size and height <= max_size:
                img.close()
                return image_path

            # Calculate new size maintaining aspect ratio
            scale = min(max_size / width, max_size / height)
            new_width = int(width * scale)
            new_height = int(height * scale)

            logging.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")

            # Resize
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save to temp file
            suffix = os.path.splitext(image_path)[1] or '.jpg'
            temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
            os.close(temp_fd)

            img.save(temp_path, quality=95)
            img.close()

            return temp_path

        except Exception as e:
            logging.warning(f"Could not resize image: {e}")
            return image_path

    async def process_image(self, image_path: str, **options) -> OCRResult:
        """
        Process an image with PaddleOCR.

        Args:
            image_path: Path to the image file
            **options:
                det: Enable text detection (default: True)
                rec: Enable text recognition (default: True)
                cls: Enable angle classification (default: use_angle_cls)

        Returns:
            OCRResult with extracted text
        """
        if not os.path.exists(image_path):
            return self._create_error_result(f"Image not found: {image_path}")

        if not self.is_available():
            error = self._init_error or "PaddleOCR not available"
            return self._create_error_result(error)

        # Resize large images to prevent crashes
        processing_path = self._resize_image_if_needed(image_path, max_size=3000)
        temp_file_created = processing_path != image_path

        with Timer() as timer:
            try:
                # Run OCR using new predict() API
                results = self.ocr.predict(processing_path)

                # Process results from PaddleOCR 3.x
                text_parts = []
                bounding_boxes = []
                confidences = []

                if results:
                    # PaddleOCR 3.x returns a list of OCRResult objects
                    # Each OCRResult is dict-like with keys:
                    # rec_texts, rec_scores, rec_polys/dt_polys, rec_boxes
                    if isinstance(results, list) and len(results) > 0:
                        result = results[0]  # First image result

                        # Check if it's the new dict-like OCRResult
                        if hasattr(result, 'keys') and hasattr(result, '__getitem__'):
                            rec_texts = result.get('rec_texts', [])
                            rec_scores = result.get('rec_scores', [])
                            rec_polys = result.get('rec_polys', result.get('dt_polys', []))

                            for i, text in enumerate(rec_texts):
                                text_parts.append(text)
                                conf = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                                confidences.append(conf)

                                if i < len(rec_polys):
                                    polygon = rec_polys[i]
                                    if hasattr(polygon, '__len__') and len(polygon) >= 4:
                                        xs = [p[0] for p in polygon]
                                        ys = [p[1] for p in polygon]
                                        x = int(min(xs))
                                        y = int(min(ys))
                                        width = int(max(xs) - x)
                                        height = int(max(ys) - y)

                                        bounding_boxes.append(BoundingBox(
                                            x=x,
                                            y=y,
                                            width=width,
                                            height=height,
                                            text=text,
                                            confidence=conf
                                        ))

                        # Handle old API response format (list of lists)
                        elif isinstance(result, list):
                            for item in result:
                                if item is None:
                                    continue

                                bbox_points = item[0]
                                text_conf = item[1]

                                if isinstance(text_conf, tuple):
                                    detected_text, confidence = text_conf
                                else:
                                    detected_text = str(text_conf)
                                    confidence = 1.0

                                text_parts.append(detected_text)
                                confidences.append(float(confidence))

                                if bbox_points and len(bbox_points) >= 4:
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
                                        confidence=float(confidence)
                                    ))

                text = "\n".join(text_parts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                # Clean up temp file
                if temp_file_created and os.path.exists(processing_path):
                    try:
                        os.remove(processing_path)
                    except Exception:
                        pass

                return self._create_success_result(
                    text=text,
                    processing_time_ms=timer.elapsed_ms,
                    confidence=avg_confidence,
                    bounding_boxes=bounding_boxes if bounding_boxes else None,
                    raw_response=results
                )

            except Exception as e:
                logging.error(f"PaddleOCR processing error: {e}")
                # Clean up temp file on error too
                if temp_file_created and os.path.exists(processing_path):
                    try:
                        os.remove(processing_path)
                    except Exception:
                        pass
                return self._create_error_result(str(e), timer.elapsed_ms)

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return self.SUPPORTED_LANGUAGES

    def set_language(self, lang: str):
        """
        Change the language configuration.

        Note: This requires reinitializing the OCR engine.

        Args:
            lang: Language code
        """
        self.lang = lang
        self._initialized = False
        self.ocr = None
        self._initialize_ocr()


# CLI interface for testing
if __name__ == "__main__":
    import sys

    def print_help():
        print("PaddleOCR Provider")
        print("")
        print("Usage:")
        print("  python -m ocr.providers.paddleocr_provider test <image>  - Test OCR")
        print("  python -m ocr.providers.paddleocr_provider check         - Check availability")
        print("  python -m ocr.providers.paddleocr_provider langs         - List languages")
        print("")
        print("Options:")
        print("  --lang <code>    Set language (default: en)")
        print("  --no-gpu         Disable GPU acceleration")
        print("  --no-angle       Disable angle classification")
        print("")
        print("Examples:")
        print("  python -m ocr.providers.paddleocr_provider test doc.jpg")
        print("  python -m ocr.providers.paddleocr_provider test chinese.jpg --lang ch")

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "check":
        print(f"PaddleOCR installed: {PADDLEOCR_AVAILABLE}")
        if PADDLEOCR_AVAILABLE:
            provider = PaddleOCRProvider(use_gpu=False, show_log=False)
            print(f"Provider available: {provider.is_available()}")
        sys.exit(0)

    elif command == "langs":
        print("Commonly supported languages:")
        for lang in PaddleOCRProvider.SUPPORTED_LANGUAGES:
            print(f"  {lang}")
        print("\nSee PaddleOCR docs for full list:")
        print("https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/multi_languages_en.md")
        sys.exit(0)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Error: Please provide an image path")
            sys.exit(1)

        image_path = sys.argv[2]

        # Parse optional arguments
        lang = "en"
        use_gpu = True
        use_angle = True
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--lang" and i + 1 < len(sys.argv):
                lang = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--no-gpu":
                use_gpu = False
                i += 1
            elif sys.argv[i] == "--no-angle":
                use_angle = False
                i += 1
            else:
                i += 1

        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        print(f"Testing PaddleOCR on: {image_path}")
        print(f"Language: {lang}")
        print(f"GPU: {use_gpu}")
        print(f"Angle classification: {use_angle}")
        print("")

        provider = PaddleOCRProvider(
            lang=lang,
            use_gpu=use_gpu,
            use_angle_cls=use_angle,
            show_log=False
        )

        if not provider.is_available():
            print("Error: PaddleOCR not available")
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
