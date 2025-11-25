"""
TTS Service using AWS Streaming API Gateway endpoint.

This service provides text-to-speech functionality with:
- AWS Lambda streaming endpoint
- TRUE streaming: Send full text, receive progressive audio stream
- Progressive playback starts after ~2 second prefetch
- Synchronous streaming with requests library for smooth playback
"""

import logging
import requests
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path to access api_integration
sys.path.insert(0, str(Path(__file__).parent.parent))

from api_integration.api_settings import API_URLS
from text_cleanup import clean_ocr_text, CleanupConfig


class TTSService:
    """TTS service using AWS streaming endpoint"""

    def __init__(self, credentials, apply_cleanup: bool = True, cleanup_config: CleanupConfig = None):
        """
        Initialize TTS service.

        Args:
            credentials: UserCredentials object with registration_key and device_id
            apply_cleanup: Whether to apply text cleanup before TTS (default True)
            cleanup_config: Optional CleanupConfig for custom cleanup settings.
                           If None and apply_cleanup=True, uses TTS-optimized cleanup.
        """
        self.streaming_url = API_URLS.TTS_STREAMING
        self.credentials = credentials
        self.apply_cleanup = apply_cleanup
        self.cleanup_config = cleanup_config

        logging.info(f"TTS Service initialized (cleanup={apply_cleanup})")

    def synthesize_streaming_progressive(self, text: str, audio_player, stop_callback=None) -> bool:
        """
        Synthesize text using AWS streaming endpoint with PROGRESSIVE PLAYBACK.

        Starts playing after ~2 second prefetch for smooth startup.

        Args:
            text: Full text to synthesize
            audio_player: StreamingAudioPlayer instance for playback
            stop_callback: Optional callable that returns True if streaming should stop

        Returns:
            True if successful, False if failed
        """
        if not text or not text.strip():
            logging.error("Cannot synthesize empty text")
            return False

        # Optionally clean text for TTS
        if self.apply_cleanup:
            original_len = len(text)
            text = clean_ocr_text(text, self.cleanup_config)
            logging.info(f"Text cleanup applied: {original_len} -> {len(text)} chars")

        logging.info(f"Starting streaming synthesis: {len(text)} chars ({len(text.split())} words)")

        payload = {
            "registration_key": self.credentials.registration_key,
            "device_id": self.credentials.device_id,
            "text": text
        }

        # Streaming settings
        prefetch_size = 192000  # ~2 seconds at 24kHz 16-bit (before starting)

        try:
            # Synchronous streaming request
            response = requests.post(
                self.streaming_url,
                json=payload,
                stream=True,
                timeout=120,
                verify=False
            )
            response.raise_for_status()

            logging.info("Receiving audio stream...")
            audio_buffer = b''
            stream_started = False
            total_bytes = 0
            first_chunk = True

            # Stream audio chunks as they arrive
            for chunk in response.iter_content(chunk_size=4096):
                # Check if we should stop
                if stop_callback and stop_callback():
                    logging.info("Streaming stopped by user callback")
                    response.close()
                    audio_player.close()
                    return False

                if chunk:
                    if first_chunk:
                        logging.info(f"First audio chunk received ({len(chunk)} bytes)")
                        first_chunk = False

                    total_bytes += len(chunk)

                    # Prefetch: wait for 2 seconds of audio before starting stream
                    if not stream_started:
                        audio_buffer += chunk
                        if len(audio_buffer) >= prefetch_size:
                            logging.info(f"Prefetch complete ({len(audio_buffer):,} bytes), starting stream")
                            stream_started = True
                            audio_player.start_stream()
                            # Write prefetched audio
                            audio_player.write_chunk(audio_buffer)
                            audio_buffer = b''
                    else:
                        # After stream started, write chunks IMMEDIATELY (no buffering!)
                        audio_player.write_chunk(chunk)

            # Write any remaining buffered audio (from prefetch phase)
            if len(audio_buffer) > 0:
                if not stream_started:
                    audio_player.start_stream()
                audio_player.write_chunk(audio_buffer)

            logging.info(f"Stream complete: {total_bytes:,} total bytes received")
            logging.info("Streaming playback completed")
            return True

        except requests.Timeout:
            logging.error("Streaming timeout")
            return False
        except requests.RequestException as e:
            logging.error(f"Streaming request error: {e}")
            return False
        except Exception as e:
            logging.error(f"Error in streaming: {e}")
            return False


def speak_text(text: str, credentials, apply_cleanup: bool = True,
               cleanup_config: CleanupConfig = None) -> bool:
    """
    Convenience function to speak text using TTS.

    Args:
        text: Text to speak
        credentials: UserCredentials object
        apply_cleanup: Whether to apply text cleanup before TTS (default True)
        cleanup_config: Optional CleanupConfig for custom cleanup settings

    Returns:
        True if successful, False if failed
    """
    from audio_player import StreamingAudioPlayer, is_audio_available

    if not is_audio_available():
        logging.error("Audio playback not available")
        return False

    service = TTSService(credentials, apply_cleanup=apply_cleanup,
                         cleanup_config=cleanup_config)

    with StreamingAudioPlayer() as player:
        return service.synthesize_streaming_progressive(text, player)


# Export main classes and configs
__all__ = ['TTSService', 'speak_text', 'CleanupConfig']
