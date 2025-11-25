# OCR Providers Research and Implementation Plan

## Overview

This document outlines the comprehensive research, licensing analysis, and implementation plan for OCR providers. The goal is to evaluate multiple OCR solutions across accuracy, speed, cost, and language support while ensuring all providers allow commercial use.

---

## Phase 1: Provider Research and Analysis

### 1.1 Local/Offline OCR Providers

| Provider | License | Commercial Use | Languages | Est. Latency | Notes |
|----------|---------|----------------|-----------|--------------|-------|
| **Tesseract OCR** | Apache 2.0 | YES | 100+ | 1-5s/page | Industry standard, maintained by Google. Must include Apache license with product. [Source](https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE) |
| **EasyOCR** | Apache 2.0 | YES | 80+ | 2-8s/page | PyTorch-based. Good accuracy. [Source](https://github.com/JaidedAI/EasyOCR/blob/master/LICENSE) |
| **PaddleOCR** | Apache 2.0 | YES* | 80+ | 1-3s/page | Excellent for CJK. *Watch PyMuPDF dependency (AGPL). [Source](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE) |
| **docTR** | Apache 2.0 | YES | Latin | 1-4s/page | By Mindee, PyTorch ecosystem. [Source](https://github.com/mindee/doctr) |
| **TrOCR** | MIT | YES | Multi | 2-5s/page | Microsoft transformer-based. [Source](https://github.com/microsoft/unilm/blob/master/LICENSE) |
| **Florence-2** | MIT | YES | Multi | ~1s/page | Microsoft VLM, lightweight (0.23B-0.77B params). [Source](https://huggingface.co/microsoft/Florence-2-large) |
| **Qwen2-VL** | Apache 2.0 | YES (2B/7B) | Multi | 2-6s/page | Alibaba. 72B uses Qwen license. [Source](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) |
| **Windows OCR** | Proprietary | YES | 25+ | <1s/page | Built into Windows 10/11, free to use |
| **Surya OCR** | GPL-3.0 + Custom | LIMITED | 90+ | 2-5s/page | Free for <$2M revenue. Commercial license required above. [Source](https://github.com/VikParuchuri/surya/blob/master/LICENSE) |
| **GOT-OCR 2.0** | Research Only | NO | Multi | 1-3s/page | NOT for commercial use. [Source](https://github.com/Ucas-HaoranWei/GOT-OCR2.0) |
| **DeepSeek-OCR** | TBD | TBD | Multi | TBD | Released Oct 2025, 97% accuracy at 10x compression. [Source](https://github.com/deepseek-ai/DeepSeek-OCR) |

### 1.2 Cloud/API OCR Providers

| Provider | License/Terms | Commercial Use | Pricing | Est. Latency | Languages |
|----------|---------------|----------------|---------|--------------|-----------|
| **Google Cloud Vision** | Commercial API | YES | $1.50/1K images | 1-3s | 100+ |
| **Azure AI Document Intelligence** | Commercial API | YES | ~$1.50/1K pages (Read), $30/1K pages (Custom) | 2-5s | 100+ |
| **AWS Textract** | Commercial API | YES | $1.50/1K pages (Detect), $15/1K (Forms) | 2-4s | 6 |
| **OpenAI GPT-4o Vision** | Commercial API | YES | $2.50/1M input + ~$0.008/image | 2-5s | Multi |
| **OpenAI GPT-4o-mini** | Commercial API | YES | $0.15/1M input + ~$0.008/image | 1-3s | Multi |
| **Google Gemini 2.0 Flash** | Commercial API | YES | $0.10/1M tokens (~$0.001/image) | 1-3s | Multi |
| **Anthropic Claude 3.5 Sonnet** | Commercial API | YES | $3/1M input (~$0.005/image) | 2-4s | Multi |
| **Anthropic Claude 3 Haiku** | Commercial API | YES | $0.25/1M input (~$0.0004/image) | 1-2s | Multi |
| **OCR.space** | Freemium | YES (with limits) | Free: 500/day, PRO: Contact | 1-3s | 25+ |
| **ABBYY Cloud** | Commercial API | YES | Custom pricing | 2-5s | 200+ |
| **Mindee** | Freemium | YES | Free: 25/month, $0.10/page after | 1-3s | Multi |

### 1.3 Pricing Comparison (Estimated per 1,000 pages)

| Provider | Basic OCR | With Structure/Tables | Notes |
|----------|-----------|----------------------|-------|
| **Local (Tesseract, etc.)** | $0 | $0 | Compute costs only |
| **Google Cloud Vision** | $1.50 | $1.50 | Document text detection |
| **Azure Document Intelligence** | ~$1.50 | $30-65 | Read vs Custom models |
| **AWS Textract** | $1.50 | $15-65 | DetectText vs AnalyzeDocument |
| **Gemini 2.0 Flash** | ~$1.00 | ~$1.00 | Best price/performance for LLM |
| **Claude 3 Haiku** | ~$0.40 | ~$0.40 | Cheapest LLM option |
| **GPT-4o-mini** | ~$8.00 | ~$8.00 | Higher token costs |
| **OCR.space** | Free-$$ | Contact | 500 free/day |

---

## Phase 2: Commercial License Summary

### APPROVED FOR COMMERCIAL USE

| Provider | License | Requirements |
|----------|---------|--------------|
| Tesseract | Apache 2.0 | Include license copy |
| EasyOCR | Apache 2.0 | Include license copy |
| PaddleOCR | Apache 2.0 | Include license, check dependencies |
| docTR | Apache 2.0 | Include license copy |
| TrOCR | MIT | Include copyright notice |
| Florence-2 | MIT | Include copyright notice |
| Qwen2-VL (2B/7B) | Apache 2.0 | Include license copy |
| Windows OCR | Proprietary | Windows license covers use |
| All Cloud APIs | Commercial | Follow API terms of service |

### RESTRICTED / NOT FOR COMMERCIAL USE

| Provider | Restriction | Alternative |
|----------|-------------|-------------|
| **GOT-OCR 2.0** | Research only | Use Qwen2-VL or Florence-2 |
| **Surya OCR** | <$2M revenue or need license | Contact Datalab for commercial license |
| **Qwen2-VL-72B** | Qwen license (review terms) | Use 2B or 7B versions |
| **DeepSeek-OCR** | License TBD (new release) | Wait for clarification |

### DEPENDENCY WARNINGS

| Provider | Dependency | Issue | Solution |
|----------|------------|-------|----------|
| PaddleOCR | PyMuPDF | AGPL-3.0 may require open-sourcing | Use pdf2image instead, or avoid PDF features |

---

## Phase 3: Provider Interface Design

### 3.1 Base Provider Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum

class ProviderType(Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    LLM = "llm"

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float

@dataclass
class OCRResult:
    success: bool
    provider: str
    provider_type: ProviderType
    text: str
    word_count: int
    confidence: float
    processing_time_ms: float

    # Optional detailed data
    bounding_boxes: Optional[List[BoundingBox]] = None
    language_detected: Optional[str] = None
    pages: Optional[List[Dict]] = None

    # Error info
    error: Optional[str] = None

    # Cost tracking (cloud providers)
    estimated_cost_usd: Optional[float] = None
    tokens_used: Optional[int] = None

class OCRProvider(ABC):
    """Base interface for all OCR providers."""

    PROVIDER_NAME: str
    PROVIDER_TYPE: ProviderType
    COMMERCIAL_USE: bool
    LICENSE: str

    @abstractmethod
    def __init__(self, **config):
        """Initialize with provider-specific configuration."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and ready."""
        pass

    @abstractmethod
    async def process_image(self, image_path: str, **options) -> OCRResult:
        """Process image and return OCR result."""
        pass

    def process_image_sync(self, image_path: str, **options) -> OCRResult:
        """Synchronous wrapper."""
        import asyncio
        return asyncio.run(self.process_image(image_path, **options))

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        pass

    def estimate_cost(self, num_pages: int = 1) -> Optional[float]:
        """Estimate cost for processing (USD). None for local providers."""
        return None
```

### 3.2 Provider Registry

```python
class ProviderRegistry:
    """Registry for managing OCR providers."""

    _providers: Dict[str, type] = {}
    _instances: Dict[str, OCRProvider] = {}

    @classmethod
    def register(cls, provider_class: type) -> None:
        cls._providers[provider_class.PROVIDER_NAME] = provider_class

    @classmethod
    def get(cls, name: str, **config) -> OCRProvider:
        if name not in cls._instances:
            cls._instances[name] = cls._providers[name](**config)
        return cls._instances[name]

    @classmethod
    def list_all(cls) -> List[str]:
        return list(cls._providers.keys())

    @classmethod
    def list_available(cls) -> List[str]:
        return [n for n, p in cls._providers.items()
                if cls.get(n).is_available()]

    @classmethod
    def list_commercial(cls) -> List[str]:
        return [n for n, p in cls._providers.items()
                if p.COMMERCIAL_USE]
```

---

## Phase 4: Implementation Priority

### Priority 1: Essential Providers (Implement First)

| # | Provider | Type | Why Priority |
|---|----------|------|--------------|
| 1 | **Google Cloud Vision** | Cloud | DONE - Already implemented |
| 2 | **Tesseract** | Local | Most widely used local OCR |
| 3 | **Gemini 2.0 Flash** | LLM | Best price/performance, shares API key |

### Priority 2: Key Alternatives

| # | Provider | Type | Why Include |
|---|----------|------|-------------|
| 4 | **Azure Document Intelligence** | Cloud | Enterprise standard, good forms |
| 5 | **AWS Textract** | Cloud | AWS ecosystem, tables/queries |
| 6 | **Claude 3 Haiku** | LLM | Cheapest LLM, good accuracy |

### Priority 3: High-Accuracy Options

| # | Provider | Type | Why Include |
|---|----------|------|-------------|
| 7 | **EasyOCR** | Local | Good accuracy, multilingual |
| 8 | **PaddleOCR** | Local | Excellent for CJK |
| 9 | **GPT-4o** | LLM | Highest accuracy for complex |

### Priority 4: Specialized/Experimental

| # | Provider | Type | Why Include |
|---|----------|------|-------------|
| 10 | **Florence-2** | Local | Lightweight VLM |
| 11 | **Qwen2-VL** | Local | Open weights, strong OCR |
| 12 | **docTR** | Local | Document-focused |
| 13 | **Windows OCR** | Local | Zero-install on Windows |
| 14 | **OCR.space** | Cloud | Free tier for testing |

### NOT Implementing (License Issues)

| Provider | Reason |
|----------|--------|
| GOT-OCR 2.0 | Research only license |
| Surya OCR | GPL + revenue limits |
| DeepSeek-OCR | License unclear (new) |

---

## Phase 5: Implementation Details

### 5.1 Tesseract Implementation

**File**: `ocr/providers/tesseract_ocr.py`

```python
class TesseractOCR(OCRProvider):
    PROVIDER_NAME = "tesseract"
    PROVIDER_TYPE = ProviderType.LOCAL
    COMMERCIAL_USE = True
    LICENSE = "Apache-2.0"

    def __init__(self, lang: str = "eng", psm: int = 3, oem: int = 3):
        self.lang = lang
        self.psm = psm  # Page segmentation mode
        self.oem = oem  # OCR Engine mode (3 = LSTM + legacy)
```

**Dependencies**: `pytesseract`, Tesseract binary installed

**Tasks**:
- [ ] Basic implementation
- [ ] Multi-language support
- [ ] PSM mode configuration
- [ ] Confidence extraction
- [ ] Bounding box extraction
- [ ] Hebrew testing

### 5.2 Gemini Vision Implementation

**File**: `ocr/providers/gemini_vision.py`

```python
class GeminiVisionOCR(OCRProvider):
    PROVIDER_NAME = "gemini_vision"
    PROVIDER_TYPE = ProviderType.LLM
    COMMERCIAL_USE = True
    LICENSE = "Commercial API"

    # Shares API key with existing gemini/api_key in keyring
```

**Dependencies**: `google-generativeai`

**Pricing**: ~$0.10/1M tokens, ~$0.001/image

**Tasks**:
- [ ] Share API key from keyring (gemini/api_key)
- [ ] Image encoding
- [ ] OCR-optimized prompting
- [ ] Markdown output parsing
- [ ] Cost tracking

### 5.3 Azure Document Intelligence

**File**: `ocr/providers/azure_doc_intel.py`

**Dependencies**: `azure-ai-documentintelligence`

**Pricing**: ~$1.50/1K pages (Read API)

**Tasks**:
- [ ] API key configuration
- [ ] Read API (general OCR)
- [ ] Layout API (tables, structure)
- [ ] Async processing
- [ ] Cost tracking

### 5.4 AWS Textract

**File**: `ocr/providers/aws_textract.py`

**Dependencies**: `boto3`

**Pricing**: $1.50/1K pages (DetectDocumentText)

**Tasks**:
- [ ] AWS credentials setup
- [ ] DetectDocumentText API
- [ ] AnalyzeDocument for tables
- [ ] Async operations
- [ ] Cost tracking

### 5.5 Claude Vision

**File**: `ocr/providers/claude_vision.py`

**Dependencies**: `anthropic`

**Pricing**: $0.25/1M tokens (Haiku), $3/1M (Sonnet)

**Tasks**:
- [ ] API key configuration
- [ ] Model selection (Haiku vs Sonnet)
- [ ] Image encoding
- [ ] OCR prompting
- [ ] Cost tracking

### 5.6 EasyOCR

**File**: `ocr/providers/easyocr_provider.py`

**Dependencies**: `easyocr`, PyTorch

**Tasks**:
- [ ] Basic implementation
- [ ] GPU/CPU configuration
- [ ] Language model management
- [ ] Hebrew support
- [ ] Batch processing

### 5.7 PaddleOCR

**File**: `ocr/providers/paddleocr_provider.py`

**Dependencies**: `paddleocr`, PaddlePaddle

**Notes**: Avoid PyMuPDF dependency for commercial use

**Tasks**:
- [ ] Installation handling
- [ ] PP-OCRv4 configuration
- [ ] Multilingual support
- [ ] Angle detection

---

## Phase 6: Testing Infrastructure

### 6.1 Test Image Categories

```
samples/
|-- english/
|   |-- printed_clean.jpg       # Clean printed text
|   |-- printed_noisy.jpg       # Low quality scan
|   |-- handwritten.jpg         # Handwritten text
|   |-- mixed_fonts.jpg         # Multiple fonts/sizes
|
|-- hebrew/
|   |-- printed_modern.jpg      # Modern Hebrew print
|   |-- with_nikud.jpg          # With vowel points
|   |-- mixed_heb_eng.jpg       # Bilingual document
|
|-- documents/
|   |-- invoice.jpg             # Invoice/receipt
|   |-- form_filled.jpg         # Filled form
|   |-- table_data.jpg          # Table with data
|   |-- multi_column.jpg        # Multi-column layout
|
|-- challenging/
|   |-- rotated_15deg.jpg       # Slightly rotated
|   |-- low_contrast.jpg        # Low contrast
|   |-- curved_page.jpg         # Book spine curve
|   |-- partial_occlusion.jpg   # Partially hidden
```

### 6.2 Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **CER** | Character Error Rate | Levenshtein(pred, truth) / len(truth) |
| **WER** | Word Error Rate | Word edits / total words |
| **Latency** | Processing time | End-to-end milliseconds |
| **Cost** | USD per page | API pricing calculation |
| **Throughput** | Pages per minute | For batch processing |

### 6.3 Comparison Script

**File**: `compare_providers.py`

```bash
# Single image, all providers
python compare_providers.py image.jpg

# Specific providers
python compare_providers.py image.jpg --providers tesseract,google_vision,gemini

# Full test suite
python compare_providers.py samples/ --output results/report.json

# With TTS playback
python compare_providers.py image.jpg --with-tts
```

---

## Phase 7: Integration with TTS

### 7.1 OCR-to-TTS Pipeline

```python
async def ocr_to_speech(
    image_path: str,
    ocr_provider: str = "google_vision",
    cleanup_config: CleanupConfig = None,
    play_audio: bool = True
) -> dict:
    """
    Complete OCR to speech pipeline.

    1. Run OCR on image
    2. Apply text cleanup (configurable)
    3. Send to TTS
    4. Stream audio playback
    """
```

### 7.2 Quality Comparison

Compare TTS output quality based on OCR source:
- Same image through multiple OCR providers
- Same cleanup settings
- Rate intelligibility of spoken output

---

## Research Sources

### Local Providers
- [Tesseract License](https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE)
- [EasyOCR License](https://github.com/JaidedAI/EasyOCR/blob/master/LICENSE)
- [PaddleOCR License](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)
- [docTR GitHub](https://github.com/mindee/doctr)
- [Florence-2 on HuggingFace](https://huggingface.co/microsoft/Florence-2-large)
- [Qwen2-VL on HuggingFace](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [Surya OCR License](https://github.com/VikParuchuri/surya/blob/master/LICENSE)
- [GOT-OCR 2.0 GitHub](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)

### Cloud Provider Pricing
- [Google Cloud Vision Pricing](https://cloud.google.com/vision/pricing)
- [Azure Document Intelligence Pricing](https://azure.microsoft.com/en-us/pricing/details/ai-document-intelligence/)
- [AWS Textract Pricing](https://aws.amazon.com/textract/pricing/)
- [OpenAI Pricing](https://openai.com/api/pricing/)
- [Gemini Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Anthropic Pricing](https://www.anthropic.com/pricing)
- [OCR.space](https://ocr.space/)

### Comparison Articles
- [Best OCR APIs 2025 - Klippa](https://www.klippa.com/en/blog/information/best-ocr-api/)
- [OCR API Guide - Docsumo](https://www.docsumo.com/blogs/ocr/api)
- [OCR Benchmarks - Nanonets](https://nanonets.com/blog/identifying-the-best-ocr-api/)

---

## Implementation Checklist

### Infrastructure
- [ ] Create base `OCRProvider` class
- [ ] Create `OCRResult` dataclass
- [ ] Create `ProviderRegistry`
- [ ] Set up test image collection
- [ ] Create ground truth files
- [ ] Build comparison script

### Providers - Local
- [x] Google Cloud Vision (implemented as cloud)
- [ ] Tesseract OCR
- [ ] EasyOCR
- [ ] PaddleOCR
- [ ] docTR
- [ ] Florence-2
- [ ] Qwen2-VL
- [ ] Windows OCR

### Providers - Cloud
- [x] Google Cloud Vision
- [ ] Azure Document Intelligence
- [ ] AWS Textract
- [ ] OCR.space

### Providers - LLM Vision
- [ ] Google Gemini 2.0 Flash
- [ ] OpenAI GPT-4o / GPT-4o-mini
- [ ] Anthropic Claude (Haiku / Sonnet)

### Testing & Comparison
- [ ] Collect test images (5+ per category)
- [ ] Create ground truth transcriptions
- [ ] Implement CER/WER metrics
- [ ] Run benchmarks
- [ ] Generate comparison report

### Integration
- [ ] OCR-to-TTS pipeline
- [ ] Provider fallback chain
- [ ] Cost optimization logic

---

## Next Steps

1. **Implement base classes** (OCRProvider, OCRResult, Registry)
2. **Implement Tesseract** (most common local)
3. **Implement Gemini Vision** (shares API key, cheap)
4. **Build comparison infrastructure**
5. **Add providers incrementally, testing as we go**
6. **Generate final comparison report**
