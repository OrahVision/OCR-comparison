#!/usr/bin/env python3
"""
Test script to verify TTS streaming works correctly.

This script tests the core TTS functionality without any OCR.
Run this to verify your setup is working.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'core'))
sys.path.insert(0, str(Path(__file__).parent / 'api_integration'))

from api_utils import UserCredentials
from tts_service import TTSService, CleanupConfig
from text_cleanup import TextProcessor, clean_ocr_text
from audio_player import StreamingAudioPlayer, is_audio_available


def test_credentials():
    """Test that credentials can be loaded."""
    print("\n=== Testing Credentials ===")

    creds = UserCredentials()

    if creds.is_valid():
        print(f"Registration key: {creds.registration_key[:20]}...")
        print(f"Device ID: {creds.device_id[:20]}...")
        print("Credentials loaded successfully!")
        return creds
    else:
        print("ERROR: Failed to load credentials")
        return None


def test_audio():
    """Test that audio playback is available."""
    print("\n=== Testing Audio ===")

    if is_audio_available():
        print("PyAudio is available")
        try:
            player = StreamingAudioPlayer()
            print("Audio player created successfully")
            player.close()
            return True
        except Exception as e:
            print(f"ERROR creating audio player: {e}")
            return False
    else:
        print("ERROR: PyAudio not available")
        return False


def test_text_cleanup():
    """Test the configurable text cleanup."""
    print("\n=== Testing Text Cleanup ===")

    # Sample OCR text with various artifacts
    raw_text = """This is a test   with   multiple    spaces.
And line
breaks that
should be
handled.

This is a new paragraph with [bracketed text] and some
hyph-
enated words."""

    print("Raw text:")
    print(repr(raw_text[:80]) + "...")

    # Test different cleanup configs
    configs = [
        ("none", CleanupConfig.none()),
        ("minimal", CleanupConfig.minimal()),
        ("comparison", CleanupConfig.comparison()),
        ("tts_optimized", CleanupConfig.tts_optimized()),
    ]

    for name, config in configs:
        result = clean_ocr_text(raw_text, config)
        print(f"\n{name}: {len(result)} chars")
        print(f"  Preview: {repr(result[:60])}...")

    print("\nText cleanup test passed!")
    return True


def test_tts_streaming(credentials):
    """Test TTS streaming with a sample text."""
    print("\n=== Testing TTS Streaming ===")

    test_text = "Hello! This is a test of the text to speech streaming system. If you can hear this, the system is working correctly."

    print(f"Test text: {test_text}")
    print("Starting TTS stream...")

    service = TTSService(credentials)

    with StreamingAudioPlayer() as player:
        success = service.synthesize_streaming_progressive(test_text, player)

    if success:
        print("TTS streaming completed successfully!")
    else:
        print("ERROR: TTS streaming failed")

    return success


def main():
    """Run all tests."""
    print("=" * 50)
    print("OCR Comparison Tool - TTS Test")
    print("=" * 50)

    # Test text cleanup (no credentials needed)
    if not test_text_cleanup():
        print("\nText cleanup test failed.")
        return 1

    # Test credentials
    credentials = test_credentials()
    if not credentials:
        print("\nFailed to load credentials. Cannot continue.")
        return 1

    # Test audio
    if not test_audio():
        print("\nAudio not available. Cannot continue.")
        return 1

    # Test TTS streaming
    if not test_tts_streaming(credentials):
        print("\nTTS streaming failed.")
        return 1

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
    return 0


if __name__ == '__main__':
    sys.exit(main())
