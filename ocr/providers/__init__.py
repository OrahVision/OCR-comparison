"""
OCR Providers package.

Each provider implements a common interface for OCR processing.
"""

from .google_vision import GoogleVisionOCR, check_google_cloud_auth

__all__ = ['GoogleVisionOCR', 'check_google_cloud_auth']
