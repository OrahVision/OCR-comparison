"""
OCR module for comparing different OCR providers.

This module provides a unified interface for multiple OCR providers,
enabling easy comparison of accuracy, speed, and cost across solutions.
"""

from .base import (
    ProviderType,
    BoundingBox,
    OCRResult,
    OCRProvider,
    Timer
)

from .registry import (
    ProviderRegistry,
    register_provider
)

__all__ = [
    # Base classes
    'ProviderType',
    'BoundingBox',
    'OCRResult',
    'OCRProvider',
    'Timer',
    # Registry
    'ProviderRegistry',
    'register_provider',
]
