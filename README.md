# OCR Comparison Tool

Compare OCR results from multiple providers and evaluate TTS quality.

## Project Structure

```
OCR-comparison/
|-- api_integration/       # AWS API connection
|   |-- api_settings.py    # Endpoint URLs
|   |-- api_utils.py       # Credential management
|
|-- core/                  # Core functionality
|   |-- device_info.py     # Device fingerprinting
|   |-- version.py         # Version constants
|   |-- version_utils.py   # Version utilities
|   |-- text_cleanup.py    # OCR text post-processing
|   |-- audio_player.py    # Streaming audio playback
|   |-- tts_service.py     # AWS TTS streaming
|
|-- samples/               # Test images
|-- results/               # OCR output comparisons
|-- test_tts.py            # TTS verification script
|-- requirements.txt       # Python dependencies
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Verify TTS streaming works:
   ```bash
   python test_tts.py
   ```

## Usage

### Test TTS Only

```python
from core.api_utils import UserCredentials
from core.tts_service import speak_text

credentials = UserCredentials()
speak_text("Hello, this is a test.", credentials)
```

### Stream TTS with Control

```python
from core.api_utils import UserCredentials
from core.tts_service import TTSService
from core.audio_player import StreamingAudioPlayer

credentials = UserCredentials()
service = TTSService(credentials)

with StreamingAudioPlayer() as player:
    service.synthesize_streaming_progressive("Your text here", player)
```

## Adding OCR Providers

OCR providers will be added to compare results. Each provider should:
1. Accept an image path
2. Return extracted text
3. Optionally return confidence scores

## Audio Format

TTS output is:
- Format: PCM16 (16-bit signed integers)
- Sample rate: 24kHz
- Channels: Mono

## License

Proprietary - OrahVision
