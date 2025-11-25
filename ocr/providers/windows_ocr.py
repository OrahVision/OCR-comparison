"""
Windows OCR Provider

Built-in OCR in Windows 10/11 using Windows Runtime APIs.
Zero external dependencies - works on any Windows 10/11 system.

License: Proprietary (Microsoft) - Free to use on Windows
"""

import logging
import os
import sys
import asyncio
from typing import List, Optional

from ..base import OCRProvider, OCRResult, BoundingBox, ProviderType, Timer
from ..registry import register_provider

# Check platform and availability
IS_WINDOWS = sys.platform == "win32"
WINOCR_AVAILABLE = False
WINRT_AVAILABLE = False

if IS_WINDOWS:
    # Try winocr package first (cleaner API)
    try:
        import winocr
        WINOCR_AVAILABLE = True
    except ImportError:
        pass

    # Try winrt as fallback
    if not WINOCR_AVAILABLE:
        try:
            # For Python 3.9+
            import winrt.windows.media.ocr as win_ocr
            import winrt.windows.graphics.imaging as imaging
            import winrt.windows.storage as storage
            WINRT_AVAILABLE = True
        except ImportError:
            try:
                # Older package name
                from winsdk.windows.media.ocr import OcrEngine
                from winsdk.windows.graphics.imaging import BitmapDecoder
                from winsdk.windows.storage import StorageFile
                WINRT_AVAILABLE = True
            except ImportError:
                pass

if not IS_WINDOWS:
    logging.info("Windows OCR only available on Windows")
elif not WINOCR_AVAILABLE and not WINRT_AVAILABLE:
    logging.info(
        "Windows OCR bindings not found. Install with: pip install winocr"
    )


@register_provider
class WindowsOCRProvider(OCRProvider):
    """
    Windows OCR Provider

    Features:
    - Built into Windows 10/11
    - Very fast (<1s per page)
    - 25+ languages supported
    - No additional installation required
    - Bounding box support
    """

    PROVIDER_NAME = "windows_ocr"
    PROVIDER_TYPE = ProviderType.LOCAL
    COMMERCIAL_USE = True
    LICENSE = "Proprietary (included with Windows)"

    # Windows OCR language codes (BCP-47)
    SUPPORTED_LANGUAGES = [
        "en-US",   # English (US)
        "en-GB",   # English (UK)
        "he-IL",   # Hebrew
        "ar-SA",   # Arabic
        "fr-FR",   # French
        "de-DE",   # German
        "es-ES",   # Spanish
        "it-IT",   # Italian
        "pt-BR",   # Portuguese (Brazil)
        "ru-RU",   # Russian
        "zh-CN",   # Chinese Simplified
        "zh-TW",   # Chinese Traditional
        "ja-JP",   # Japanese
        "ko-KR",   # Korean
        # Many more - depends on Windows language packs installed
    ]

    def __init__(self, language: str = None, **config):
        """
        Initialize Windows OCR provider.

        Args:
            language: BCP-47 language tag (e.g., 'en-US', 'he-IL')
                     If None, uses system default or auto-detection
        """
        self.language = language
        self._initialized = False
        self._init_error = None
        self._engine = None

        if IS_WINDOWS and (WINOCR_AVAILABLE or WINRT_AVAILABLE):
            self._initialize_engine()

    def _initialize_engine(self):
        """Initialize Windows OCR engine."""
        if self._initialized:
            return

        try:
            if WINOCR_AVAILABLE:
                # winocr handles initialization internally
                self._initialized = True
                logging.info(
                    f"Windows OCR initialized via winocr, language: {self.language or 'auto'}"
                )
            elif WINRT_AVAILABLE:
                # WinRT initialization
                self._initialized = True
                logging.info(
                    f"Windows OCR initialized via WinRT, language: {self.language or 'auto'}"
                )
        except Exception as e:
            self._init_error = str(e)
            logging.error(f"Failed to initialize Windows OCR: {e}")

    def is_available(self) -> bool:
        """Check if Windows OCR is available."""
        if not IS_WINDOWS:
            return False
        if not WINOCR_AVAILABLE and not WINRT_AVAILABLE:
            return False
        if not self._initialized:
            self._initialize_engine()
        return self._initialized

    async def process_image(self, image_path: str, **options) -> OCRResult:
        """
        Process an image with Windows OCR.

        Args:
            image_path: Path to the image file
            **options: Additional options (currently unused)

        Returns:
            OCRResult with extracted text
        """
        if not os.path.exists(image_path):
            return self._create_error_result(f"Image not found: {image_path}")

        if not self.is_available():
            if not IS_WINDOWS:
                return self._create_error_result("Windows OCR only available on Windows")
            error = self._init_error or "Windows OCR not available"
            return self._create_error_result(error)

        with Timer() as timer:
            try:
                if WINOCR_AVAILABLE:
                    return await self._process_with_winocr(image_path, timer)
                elif WINRT_AVAILABLE:
                    return await self._process_with_winrt(image_path, timer)
                else:
                    return self._create_error_result("No Windows OCR backend available")

            except Exception as e:
                logging.error(f"Windows OCR processing error: {e}")
                return self._create_error_result(str(e), timer.elapsed_ms)

    async def _process_with_winocr(self, image_path: str, timer: Timer) -> OCRResult:
        """Process image using winocr package."""
        import winocr
        from PIL import Image

        # Load image as PIL Image
        image = Image.open(image_path)

        # Perform OCR
        if self.language:
            result = await winocr.recognize_pil(image, lang=self.language)
        else:
            result = await winocr.recognize_pil(image)

        # Extract text and bounding boxes
        text_parts = []
        bounding_boxes = []

        if hasattr(result, 'lines'):
            for line in result.lines:
                text_parts.append(line.text)

                if hasattr(line, 'words'):
                    for word in line.words:
                        if hasattr(word, 'bounding_rect'):
                            rect = word.bounding_rect
                            bounding_boxes.append(BoundingBox(
                                x=int(rect.x),
                                y=int(rect.y),
                                width=int(rect.width),
                                height=int(rect.height),
                                text=word.text,
                                confidence=1.0  # Windows OCR doesn't provide confidence
                            ))

        text = "\n".join(text_parts)

        return self._create_success_result(
            text=text,
            processing_time_ms=timer.elapsed_ms,
            confidence=1.0,  # Windows OCR doesn't provide confidence scores
            bounding_boxes=bounding_boxes if bounding_boxes else None,
            raw_response=result
        )

    async def _process_with_winrt(self, image_path: str, timer: Timer) -> OCRResult:
        """Process image using WinRT APIs directly."""
        # This is a fallback if winocr isn't available
        # Implementation depends on which winrt package is installed

        try:
            import winrt.windows.media.ocr as win_ocr
            import winrt.windows.graphics.imaging as imaging
            import winrt.windows.storage as storage

            # Load image file
            file = await storage.StorageFile.get_file_from_path_async(
                os.path.abspath(image_path)
            )
            stream = await file.open_read_async()

            # Decode image
            decoder = await imaging.BitmapDecoder.create_async(stream)
            bitmap = await decoder.get_software_bitmap_async()

            # Get OCR engine
            if self.language:
                from winrt.windows.globalization import Language
                lang = Language(self.language)
                engine = win_ocr.OcrEngine.try_create_from_language(lang)
            else:
                engine = win_ocr.OcrEngine.try_create_from_user_profile_languages()

            if not engine:
                return self._create_error_result(
                    f"Could not create OCR engine for language: {self.language or 'default'}"
                )

            # Perform OCR
            result = await engine.recognize_async(bitmap)

            # Extract text
            text_parts = []
            bounding_boxes = []

            for line in result.lines:
                text_parts.append(line.text)

                for word in line.words:
                    rect = word.bounding_rect
                    bounding_boxes.append(BoundingBox(
                        x=int(rect.x),
                        y=int(rect.y),
                        width=int(rect.width),
                        height=int(rect.height),
                        text=word.text,
                        confidence=1.0
                    ))

            text = "\n".join(text_parts)

            return self._create_success_result(
                text=text,
                processing_time_ms=timer.elapsed_ms,
                confidence=1.0,
                bounding_boxes=bounding_boxes if bounding_boxes else None,
                raw_response=result
            )

        except Exception as e:
            return self._create_error_result(f"WinRT processing failed: {e}", timer.elapsed_ms)

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return self.SUPPORTED_LANGUAGES

    def get_installed_languages(self) -> List[str]:
        """
        Get list of actually installed OCR languages on this system.

        Returns:
            List of installed language codes
        """
        if not IS_WINDOWS:
            return []

        installed = []

        try:
            if WINOCR_AVAILABLE:
                import winocr
                # winocr may provide a way to list languages
                # For now, return common ones
                return self.SUPPORTED_LANGUAGES

            elif WINRT_AVAILABLE:
                import winrt.windows.media.ocr as win_ocr
                available = win_ocr.OcrEngine.available_recognizer_languages
                installed = [lang.language_tag for lang in available]

        except Exception as e:
            logging.debug(f"Could not get installed languages: {e}")

        return installed or self.SUPPORTED_LANGUAGES


# CLI interface for testing
if __name__ == "__main__":

    def print_help():
        print("Windows OCR Provider")
        print("")
        print("Usage:")
        print("  python -m ocr.providers.windows_ocr test <image>   - Test OCR")
        print("  python -m ocr.providers.windows_ocr check          - Check availability")
        print("  python -m ocr.providers.windows_ocr langs          - List languages")
        print("")
        print("Options:")
        print("  --lang <code>    Set language (e.g., en-US, he-IL)")
        print("")
        print("Examples:")
        print("  python -m ocr.providers.windows_ocr test document.jpg")
        print("  python -m ocr.providers.windows_ocr test hebrew.jpg --lang he-IL")

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "check":
        print(f"Platform: {sys.platform}")
        print(f"Is Windows: {IS_WINDOWS}")
        print(f"winocr available: {WINOCR_AVAILABLE}")
        print(f"WinRT available: {WINRT_AVAILABLE}")

        if IS_WINDOWS and (WINOCR_AVAILABLE or WINRT_AVAILABLE):
            provider = WindowsOCRProvider()
            print(f"Provider available: {provider.is_available()}")
        sys.exit(0)

    elif command == "langs":
        print("Supported languages (requires Windows language pack):")
        for lang in WindowsOCRProvider.SUPPORTED_LANGUAGES:
            print(f"  {lang}")

        if IS_WINDOWS and (WINOCR_AVAILABLE or WINRT_AVAILABLE):
            provider = WindowsOCRProvider()
            installed = provider.get_installed_languages()
            if installed:
                print("\nInstalled on this system:")
                for lang in installed:
                    print(f"  {lang}")
        sys.exit(0)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Error: Please provide an image path")
            sys.exit(1)

        image_path = sys.argv[2]

        # Parse optional arguments
        language = None
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--lang" and i + 1 < len(sys.argv):
                language = sys.argv[i + 1]
                i += 2
            else:
                i += 1

        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        print(f"Testing Windows OCR on: {image_path}")
        print(f"Language: {language or 'auto'}")
        print("")

        provider = WindowsOCRProvider(language=language)

        if not provider.is_available():
            print("Error: Windows OCR not available")
            if not IS_WINDOWS:
                print("  This provider only works on Windows")
            else:
                print("  Install with: pip install winocr")
            sys.exit(1)

        result = provider.process_image_sync(image_path)

        if result.success:
            print("SUCCESS!")
            print(f"Processing time: {result.processing_time_ms:.0f}ms")
            print(f"Words detected: {result.word_count}")
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
