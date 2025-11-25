"""
Base classes for OCR providers.

This module defines the common interface that all OCR providers must implement,
along with data structures for OCR results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import asyncio
import time


class ProviderType(Enum):
    """Classification of OCR provider types."""
    LOCAL = "local"      # Runs locally (Tesseract, EasyOCR, etc.)
    CLOUD = "cloud"      # Cloud API (Google Vision, Azure, AWS)
    LLM = "llm"          # LLM-based vision (GPT-4o, Gemini, Claude)


@dataclass
class BoundingBox:
    """Represents a detected text region with its location and content."""
    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float = 1.0


@dataclass
class OCRResult:
    """
    Standardized result from any OCR provider.

    All providers return this structure for consistent comparison.
    """
    success: bool
    provider: str
    provider_type: ProviderType
    text: str
    processing_time_ms: float

    # Text statistics
    word_count: int = 0
    char_count: int = 0
    line_count: int = 0

    # Confidence (0.0 to 1.0, where available)
    confidence: float = 0.0

    # Optional detailed data
    bounding_boxes: Optional[List[BoundingBox]] = None
    language_detected: Optional[str] = None
    pages: Optional[List[Dict]] = None
    raw_response: Optional[Any] = None

    # Error info (when success=False)
    error: Optional[str] = None

    # Cost tracking (for cloud/LLM providers)
    estimated_cost_usd: Optional[float] = None
    tokens_used: Optional[int] = None

    def __post_init__(self):
        """Calculate derived fields if not provided."""
        if self.text and self.word_count == 0:
            self.word_count = len(self.text.split())
        if self.text and self.char_count == 0:
            self.char_count = len(self.text)
        if self.text and self.line_count == 0:
            self.line_count = len(self.text.splitlines())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "provider": self.provider,
            "provider_type": self.provider_type.value,
            "text": self.text,
            "processing_time_ms": self.processing_time_ms,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "line_count": self.line_count,
            "confidence": self.confidence,
            "language_detected": self.language_detected,
            "error": self.error,
            "estimated_cost_usd": self.estimated_cost_usd,
            "tokens_used": self.tokens_used,
        }


class OCRProvider(ABC):
    """
    Base interface for all OCR providers.

    All OCR providers must inherit from this class and implement
    the required abstract methods.
    """

    # Class attributes - must be defined by subclasses
    PROVIDER_NAME: str = "base"
    PROVIDER_TYPE: ProviderType = ProviderType.LOCAL
    COMMERCIAL_USE: bool = True
    LICENSE: str = "Unknown"
    SUPPORTED_LANGUAGES: List[str] = ["eng"]

    @abstractmethod
    def __init__(self, **config):
        """
        Initialize with provider-specific configuration.

        Args:
            **config: Provider-specific configuration options
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if provider is configured and ready to use.

        Returns:
            True if the provider can process images, False otherwise
        """
        pass

    @abstractmethod
    async def process_image(self, image_path: str, **options) -> OCRResult:
        """
        Process an image and extract text.

        Args:
            image_path: Path to the image file
            **options: Provider-specific processing options

        Returns:
            OCRResult containing extracted text and metadata
        """
        pass

    def process_image_sync(self, image_path: str, **options) -> OCRResult:
        """
        Synchronous wrapper for process_image.

        Args:
            image_path: Path to the image file
            **options: Provider-specific processing options

        Returns:
            OCRResult containing extracted text and metadata
        """
        return asyncio.run(self.process_image(image_path, **options))

    def get_supported_languages(self) -> List[str]:
        """
        Return list of supported language codes.

        Returns:
            List of ISO language codes (e.g., ['eng', 'heb', 'fra'])
        """
        return self.SUPPORTED_LANGUAGES

    def estimate_cost(self, num_pages: int = 1) -> Optional[float]:
        """
        Estimate cost for processing pages (USD).

        Args:
            num_pages: Number of pages to process

        Returns:
            Estimated cost in USD, or None for local/free providers
        """
        return None

    def _create_error_result(self, error: str, processing_time_ms: float = 0) -> OCRResult:
        """
        Helper to create an error result.

        Args:
            error: Error message
            processing_time_ms: Time spent before error

        Returns:
            OCRResult with success=False
        """
        return OCRResult(
            success=False,
            provider=self.PROVIDER_NAME,
            provider_type=self.PROVIDER_TYPE,
            text="",
            processing_time_ms=processing_time_ms,
            error=error
        )

    def _create_success_result(
        self,
        text: str,
        processing_time_ms: float,
        confidence: float = 0.0,
        bounding_boxes: Optional[List[BoundingBox]] = None,
        language_detected: Optional[str] = None,
        pages: Optional[List[Dict]] = None,
        raw_response: Optional[Any] = None,
        estimated_cost_usd: Optional[float] = None,
        tokens_used: Optional[int] = None
    ) -> OCRResult:
        """
        Helper to create a success result.

        Args:
            text: Extracted text
            processing_time_ms: Processing time
            confidence: Overall confidence score
            bounding_boxes: List of detected text regions
            language_detected: Detected language code
            pages: Structured page data
            raw_response: Raw provider response
            estimated_cost_usd: Cost estimate
            tokens_used: Token count (for LLM providers)

        Returns:
            OCRResult with success=True
        """
        return OCRResult(
            success=True,
            provider=self.PROVIDER_NAME,
            provider_type=self.PROVIDER_TYPE,
            text=text,
            processing_time_ms=processing_time_ms,
            confidence=confidence,
            bounding_boxes=bounding_boxes,
            language_detected=language_detected,
            pages=pages,
            raw_response=raw_response,
            estimated_cost_usd=estimated_cost_usd,
            tokens_used=tokens_used
        )


class Timer:
    """Simple context manager for timing operations."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000


# Exports
__all__ = [
    'ProviderType',
    'BoundingBox',
    'OCRResult',
    'OCRProvider',
    'Timer'
]
