"""
Streaming audio player for TTS playback.

Uses PyAudio for true progressive streaming - writes chunks directly to
audio output as they arrive from the TTS service.
"""

import logging

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logging.warning("pyaudio not installed - install with: pip install pyaudio")


class StreamingAudioPlayer:
    """
    Streaming audio player using PyAudio for true progressive playback.

    Writes audio chunks directly to output stream as they arrive.
    PyAudio is industry standard for streaming audio in Python.
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the streaming audio player.

        Args:
            sample_rate: Audio sample rate in Hz (default 24000 for TTS)
        """
        if not PYAUDIO_AVAILABLE:
            raise ImportError("pyaudio not available - install with: pip install pyaudio")

        self.sample_rate = sample_rate
        self.p = None
        self.stream = None
        self.playing = False

    def start_stream(self):
        """Start the audio output stream."""
        if self.stream is None:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=1024
            )
            self.playing = True
            logging.info("PyAudio stream started")

    def write_chunk(self, audio_bytes: bytes):
        """
        Write audio chunk to the stream.

        Args:
            audio_bytes: Raw PCM16 audio data
        """
        if len(audio_bytes) == 0:
            return

        try:
            if self.stream is None:
                self.start_stream()
            self.stream.write(audio_bytes)
        except Exception as e:
            logging.error(f"Error writing audio chunk: {e}")

    def stop(self):
        """Stop the audio stream."""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                self.playing = False
            except Exception as e:
                logging.error(f"Error stopping stream: {e}")

        if self.p:
            try:
                self.p.terminate()
                self.p = None
            except Exception as e:
                logging.error(f"Error terminating PyAudio: {e}")

    def close(self):
        """Close the audio player (alias for stop)."""
        self.stop()

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self.playing

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        self.close()
        return False


def is_audio_available() -> bool:
    """Check if audio playback is available."""
    return PYAUDIO_AVAILABLE
