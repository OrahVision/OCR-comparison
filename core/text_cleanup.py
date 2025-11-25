"""
Text cleanup utilities for post-OCR processing.

Provides configurable cleanup of OCR text for storage and TTS use.
The TextProcessor class allows fine-grained control over which cleanup
stages are applied - useful for comparing raw vs cleaned OCR output.

Usage:
    # Full cleanup (default, TTS-optimized)
    cleaned = clean_ocr_text(raw_text)

    # No cleanup (raw)
    raw = clean_ocr_text(raw_text, CleanupConfig.none())

    # Custom cleanup
    config = CleanupConfig(normalize_linebreaks=True, normalize_whitespace=True)
    processed = clean_ocr_text(raw_text, config)

    # Or use the TextProcessor class directly
    processor = TextProcessor(CleanupConfig.minimal())
    result = processor.process(raw_text)
"""

import re
import logging
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class CleanupConfig:
    """
    Configuration for text cleanup stages.

    Each flag controls a specific cleanup operation. All default to True
    for full TTS-optimized cleanup.
    """

    # Individual cleanup stages
    rejoin_hyphenated: bool = True       # "bro-\nken" -> "broken"
    normalize_linebreaks: bool = True    # Single \n -> space, preserve \n\n
    remove_brackets: bool = True         # [text] -> text
    remove_ocr_artifacts: bool = True    # =e, trademark symbols, etc.
    normalize_quotes: bool = True        # Smart quotes -> ASCII quotes
    remove_pipes: bool = True            # Standalone | characters
    remove_isolated_chars: bool = True   # Single noise characters
    remove_decorative: bool = True       # Asterisks, bullets, dashes
    normalize_whitespace: bool = True    # Multiple spaces -> single
    normalize_punctuation: bool = True   # !!! -> !, .... -> .
    strip_text: bool = True              # Remove leading/trailing whitespace

    @classmethod
    def none(cls) -> 'CleanupConfig':
        """No cleanup - return raw text as-is."""
        return cls(
            rejoin_hyphenated=False,
            normalize_linebreaks=False,
            remove_brackets=False,
            remove_ocr_artifacts=False,
            normalize_quotes=False,
            remove_pipes=False,
            remove_isolated_chars=False,
            remove_decorative=False,
            normalize_whitespace=False,
            normalize_punctuation=False,
            strip_text=False,
        )

    @classmethod
    def minimal(cls) -> 'CleanupConfig':
        """Minimal cleanup - just whitespace normalization and strip."""
        return cls(
            rejoin_hyphenated=False,
            normalize_linebreaks=False,
            remove_brackets=False,
            remove_ocr_artifacts=False,
            normalize_quotes=False,
            remove_pipes=False,
            remove_isolated_chars=False,
            remove_decorative=False,
            normalize_whitespace=True,
            normalize_punctuation=False,
            strip_text=True,
        )

    @classmethod
    def tts_optimized(cls) -> 'CleanupConfig':
        """Full cleanup optimized for TTS (default)."""
        return cls()  # All defaults are True

    @classmethod
    def comparison(cls) -> 'CleanupConfig':
        """
        Cleanup suitable for OCR comparison.

        Normalizes whitespace and quotes for fair comparison,
        but preserves most OCR artifacts to see provider differences.
        """
        return cls(
            rejoin_hyphenated=True,
            normalize_linebreaks=False,  # Keep line structure for comparison
            remove_brackets=False,
            remove_ocr_artifacts=False,  # Keep artifacts to compare providers
            normalize_quotes=True,
            remove_pipes=False,
            remove_isolated_chars=False,
            remove_decorative=False,
            normalize_whitespace=True,
            normalize_punctuation=False,
            strip_text=True,
        )


class TextProcessor:
    """
    Configurable text processor for OCR cleanup.

    Allows fine-grained control over which cleanup stages are applied,
    useful for comparing raw vs cleaned OCR output.
    """

    def __init__(self, config: Optional[CleanupConfig] = None):
        """
        Initialize text processor.

        Args:
            config: Cleanup configuration. If None, uses full TTS-optimized cleanup.
        """
        self.config = config or CleanupConfig.tts_optimized()

    def process(self, text: str) -> str:
        """
        Process text according to configuration.

        Args:
            text: Raw OCR text

        Returns:
            Processed text
        """
        if not text:
            return text

        # Check if config is "none" - skip all processing
        if isinstance(self.config, CleanupConfig) and self.config == CleanupConfig.none():
            return text

        if not text.strip():
            return text

        original_length = len(text)

        if self.config.rejoin_hyphenated:
            # Rejoin hyphenated words at line breaks: "bro- \nken" -> "broken"
            text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

        if self.config.normalize_linebreaks:
            # Replace single line breaks with space, preserve paragraph breaks
            text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

        if self.config.remove_brackets:
            # Remove brackets but keep content: [text] -> text
            text = re.sub(r'\[([^\]]+)\]', r'\1', text)

        if self.config.remove_ocr_artifacts:
            text = re.sub(r'=e', '', text)  # Common OCR artifact
            text = re.sub(r'[\u00ae\u2122\u00a9]', '', text)  # Trademark symbols
            text = re.sub(r'\\', '', text)  # Backslashes
            text = re.sub(r'\ufffd', '', text)  # Unicode replacement character
            text = re.sub(r'_+', ' ', text)  # Underscores (often OCR noise)

        if self.config.normalize_quotes:
            # Smart quotes -> ASCII quotes
            text = re.sub(r'[\u2018\u2019\u201a\u201b]', "'", text)
            text = re.sub(r'[\u201c\u201d\u201e\u201f]', '"', text)

        if self.config.remove_pipes:
            text = re.sub(r'\s*\|\s*\|+\s*', ' ', text)  # Multiple pipes
            text = re.sub(r'^\s*\|\s*', '', text, flags=re.MULTILINE)
            text = re.sub(r'\s*\|\s*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'\s+\|\s+', ' ', text)

        if self.config.remove_isolated_chars:
            # Remove single noise characters (but not "I" or "a")
            text = re.sub(r'(?<!\w)\s+[^IAia\w\s]\s+(?!\w)', ' ', text)
            text = re.sub(r'(?<![A-Za-z])\s+[b-hj-zB-HJ-Z]\s+(?![A-Za-z])', ' ', text)

        if self.config.remove_decorative:
            text = re.sub(r'\*+', ' ', text)  # Asterisks
            text = re.sub(r'[\u2022\u25e6\u25aa\u25ab\u25a0\u25a1\u25cf\u25cb]', ' ', text)  # Bullets
            text = re.sub(r'[\u2014\u2013]', ' ', text)  # Em/en dashes

        if self.config.normalize_punctuation:
            text = re.sub(r'([!?])\1{2,}', r'\1', text)  # !!! -> !
            text = re.sub(r'\.{4,}', '.', text)  # .... -> .
            text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines

        if self.config.normalize_whitespace:
            text = re.sub(r' {2,}', ' ', text)  # Multiple spaces -> single
            text = re.sub(r' \n', '\n', text)  # Space before newline
            text = re.sub(r'\n ', '\n', text)  # Space after newline

        if self.config.strip_text:
            text = text.strip()

        cleaned_length = len(text)
        if original_length != cleaned_length:
            logging.debug(f"Text processing: {original_length} -> {cleaned_length} chars")

        return text


# Default processor instances for convenience
_tts_processor = TextProcessor(CleanupConfig.tts_optimized())
_raw_processor = TextProcessor(CleanupConfig.none())
_minimal_processor = TextProcessor(CleanupConfig.minimal())
_comparison_processor = TextProcessor(CleanupConfig.comparison())


def clean_ocr_text(text: str, config: Optional[CleanupConfig] = None) -> str:
    """
    Clean OCR text with configurable cleanup.

    Args:
        text: Raw OCR text
        config: Cleanup configuration. Options:
            - None: Full TTS-optimized cleanup (default)
            - CleanupConfig.none(): No cleanup, return raw
            - CleanupConfig.minimal(): Just whitespace normalization
            - CleanupConfig.comparison(): Suitable for OCR comparison
            - CleanupConfig(...): Custom configuration

    Returns:
        Cleaned text
    """
    if config is None:
        return _tts_processor.process(text)
    return TextProcessor(config).process(text)


def get_raw_text(text: str) -> str:
    """Return text without any cleanup."""
    return text


def get_minimal_cleanup(text: str) -> str:
    """Return text with minimal cleanup (whitespace only)."""
    return _minimal_processor.process(text)


def get_comparison_cleanup(text: str) -> str:
    """Return text with comparison-suitable cleanup."""
    return _comparison_processor.process(text)


def check_text_quality(text: str) -> List[str]:
    """
    Check text quality and return user-facing warnings.

    Used by TTS to warn users about potential OCR quality issues.

    Args:
        text: Cleaned text to check

    Returns:
        List of warning messages (empty if no issues)
    """
    warnings = []

    if not text or not text.strip():
        warnings.append("Warning: No text found on this page.")
        return warnings

    word_count = len(text.split())
    if word_count < 10:
        warnings.append(
            f"Warning: Very short text ({word_count} words). Page may be unreadable or contain OCR errors."
        )

    return warnings


def detect_text_quality(text: str) -> dict:
    """
    Analyze text quality and provide metrics.

    Useful for debugging OCR issues.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with quality metrics
    """
    if not text:
        return {
            'word_count': 0,
            'char_count': 0,
            'line_count': 0,
            'avg_word_length': 0,
            'has_hebrew': False,
            'quality': 'empty'
        }

    word_count = len(text.split())
    char_count = len(text)
    line_count = len(text.splitlines())

    words = text.split()
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0

    # Check for Hebrew characters (Unicode range U+0590 to U+05FF)
    has_hebrew = any('\u0590' <= c <= '\u05FF' for c in text)

    # Simple quality assessment
    if word_count < 5:
        quality = 'very_poor'
    elif word_count < 20:
        quality = 'poor'
    elif word_count < 100:
        quality = 'fair'
    else:
        quality = 'good'

    return {
        'word_count': word_count,
        'char_count': char_count,
        'line_count': line_count,
        'avg_word_length': round(avg_word_length, 2),
        'has_hebrew': has_hebrew,
        'quality': quality
    }


# Exports
__all__ = [
    'CleanupConfig',
    'TextProcessor',
    'clean_ocr_text',
    'get_raw_text',
    'get_minimal_cleanup',
    'get_comparison_cleanup',
    'check_text_quality',
    'detect_text_quality',
]
