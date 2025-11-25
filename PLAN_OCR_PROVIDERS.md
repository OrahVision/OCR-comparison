# OCR Providers Research and Implementation Plan

## Overview

This document outlines the plan for researching, implementing, and comparing OCR providers for the OCR-comparison tool. The goal is to evaluate multiple OCR solutions across accuracy, speed, cost, and language support to determine optimal providers for different use cases.

---

## Phase 1: Research and Discovery

### 1.1 Local/Offline OCR Providers

| Provider | Type | Languages | License | Notes |
|----------|------|-----------|---------|-------|
| **Tesseract OCR** | Open source | 100+ | Apache 2.0 | Industry standard, maintained by Google |
| **EasyOCR** | Python library | 80+ | Apache 2.0 | PyTorch-based, good accuracy |
| **PaddleOCR** | Python library | 80+ | Apache 2.0 | From Baidu, excellent for CJK |
| **Windows OCR** | System API | 25+ | Proprietary | Built into Windows 10/11 |
| **Doctr** | Python library | Latin scripts | Apache 2.0 | Document-focused, TensorFlow/PyTorch |
| **Surya** | Python library | 90+ | GPL 3.0 | New, high accuracy claims |
| **TrOCR** | ML Model | Multi | MIT | Microsoft transformer-based |
| **Kraken** | Python library | Multi | Apache 2.0 | Historical documents focus |

### 1.2 Cloud/API OCR Providers

| Provider | Pricing Model | Languages | Strengths |
|----------|---------------|-----------|-----------|
| **Google Cloud Vision** | Per image | 100+ | High accuracy, document detection |
| **Azure AI Vision** | Per transaction | 100+ | Good handwriting, forms |
| **Azure Document Intelligence** | Per page | 100+ | Structured document extraction |
| **AWS Textract** | Per page | 6 | Tables, forms, queries |
| **OpenAI GPT-4 Vision** | Per token | Multi | Context understanding |
| **Google Gemini Vision** | Per token | Multi | Multimodal reasoning |
| **Anthropic Claude Vision** | Per token | Multi | Reasoning, context |
| **ABBYY Cloud** | Per page | 200+ | Enterprise-grade accuracy |
| **OCR.space** | Per call/free tier | 25+ | Simple API, free tier |
| **Mathpix** | Per request | Math focus | LaTeX output, equations |

### 1.3 Research Tasks

- [ ] **Tesseract**: Test v4 vs v5, LSTM vs legacy modes, custom training
- [ ] **EasyOCR**: Benchmark GPU vs CPU, test Hebrew support
- [ ] **PaddleOCR**: Test PP-OCRv4, multilingual performance
- [ ] **Windows OCR**: Test via Python winrt bindings
- [ ] **Cloud Vision APIs**: Compare structured output formats
- [ ] **LLM Vision**: Test GPT-4V, Gemini, Claude for OCR tasks
- [ ] **Specialized**: Research Hebrew-specific OCR options

---

## Phase 2: Provider Interface Design

### 2.1 Base Provider Interface

All providers must implement this common interface:

```python
class OCRProvider:
    """Base interface for all OCR providers."""

    PROVIDER_NAME: str  # Unique identifier
    PROVIDER_TYPE: str  # "local" or "cloud"

    def __init__(self, **config):
        """Initialize with provider-specific configuration."""
        pass

    def is_available(self) -> bool:
        """Check if provider is configured and ready."""
        pass

    async def process_image(self, image_path: str) -> OCRResult:
        """Process image and return OCR result."""
        pass

    def process_image_sync(self, image_path: str) -> OCRResult:
        """Synchronous wrapper for process_image."""
        pass

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        pass

    def estimate_cost(self, image_path: str) -> Optional[float]:
        """Estimate cost for processing (cloud providers)."""
        pass
```

### 2.2 OCR Result Structure

```python
@dataclass
class OCRResult:
    success: bool
    provider: str
    text: str
    word_count: int
    confidence: float  # 0.0 to 1.0
    processing_time_ms: float

    # Optional detailed data
    pages: Optional[List[PageData]] = None
    bounding_boxes: Optional[List[BoundingBox]] = None
    language_detected: Optional[str] = None

    # Error info
    error: Optional[str] = None

    # Cost tracking (cloud providers)
    estimated_cost: Optional[float] = None
```

### 2.3 Provider Registry

```python
class ProviderRegistry:
    """Registry for managing OCR providers."""

    def register(self, provider_class: Type[OCRProvider]) -> None:
        """Register a provider class."""
        pass

    def get_provider(self, name: str, **config) -> OCRProvider:
        """Get initialized provider by name."""
        pass

    def list_available(self) -> List[str]:
        """List all available (configured) providers."""
        pass

    def list_local(self) -> List[str]:
        """List local providers only."""
        pass

    def list_cloud(self) -> List[str]:
        """List cloud providers only."""
        pass
```

---

## Phase 3: Implementation Order

### Priority 1: Core Providers (Week 1)

These are the most commonly used and should be implemented first.

#### 3.1 Tesseract OCR
- **File**: `ocr/providers/tesseract.py`
- **Dependencies**: `pytesseract`, Tesseract binary
- **Tasks**:
  - [ ] Basic implementation with pytesseract
  - [ ] Language pack configuration
  - [ ] Page segmentation modes (PSM)
  - [ ] OEM engine modes (LSTM vs legacy)
  - [ ] Confidence extraction
  - [ ] Bounding box support
  - [ ] Hebrew language testing

#### 3.2 Google Cloud Vision (DONE)
- **File**: `ocr/providers/google_vision.py`
- **Status**: Implemented
- **Tasks**:
  - [x] SDK authentication
  - [x] API key authentication (REST)
  - [x] Keyring integration
  - [x] Document text detection
  - [ ] Add bounding box extraction
  - [ ] Add language detection

### Priority 2: Alternative Cloud Providers (Week 2)

#### 3.3 Azure Computer Vision
- **File**: `ocr/providers/azure_vision.py`
- **Dependencies**: `azure-cognitiveservices-vision-computervision`
- **Tasks**:
  - [ ] API key authentication
  - [ ] Read API (async OCR)
  - [ ] Handwriting support
  - [ ] Language detection
  - [ ] Cost tracking

#### 3.4 AWS Textract
- **File**: `ocr/providers/aws_textract.py`
- **Dependencies**: `boto3`
- **Tasks**:
  - [ ] IAM/credential configuration
  - [ ] DetectDocumentText API
  - [ ] AnalyzeDocument API (tables, forms)
  - [ ] Async operations for large docs
  - [ ] Cost tracking

### Priority 3: LLM Vision Providers (Week 3)

#### 3.5 OpenAI GPT-4 Vision
- **File**: `ocr/providers/openai_vision.py`
- **Dependencies**: `openai`
- **Tasks**:
  - [ ] API key from keyring/env
  - [ ] Image encoding (base64)
  - [ ] OCR-specific prompting
  - [ ] Token/cost estimation
  - [ ] Compare gpt-4o vs gpt-4o-mini

#### 3.6 Google Gemini Vision
- **File**: `ocr/providers/gemini_vision.py`
- **Dependencies**: `google-generativeai`
- **Tasks**:
  - [ ] Share API key with existing setup
  - [ ] Image processing
  - [ ] OCR-optimized prompts
  - [ ] Compare models (flash vs pro)

#### 3.7 Anthropic Claude Vision
- **File**: `ocr/providers/claude_vision.py`
- **Dependencies**: `anthropic`
- **Tasks**:
  - [ ] API key configuration
  - [ ] Image encoding
  - [ ] OCR prompting strategy
  - [ ] Cost tracking

### Priority 4: Local Alternatives (Week 4)

#### 3.8 EasyOCR
- **File**: `ocr/providers/easyocr_provider.py`
- **Dependencies**: `easyocr`, PyTorch
- **Tasks**:
  - [ ] Basic implementation
  - [ ] GPU vs CPU configuration
  - [ ] Language model downloads
  - [ ] Hebrew support testing
  - [ ] Batch processing

#### 3.9 PaddleOCR
- **File**: `ocr/providers/paddleocr_provider.py`
- **Dependencies**: `paddleocr`, PaddlePaddle
- **Tasks**:
  - [ ] Installation (can be tricky)
  - [ ] PP-OCRv4 configuration
  - [ ] Multilingual support
  - [ ] Direction detection

#### 3.10 Windows OCR
- **File**: `ocr/providers/windows_ocr.py`
- **Dependencies**: `winrt` (Windows only)
- **Tasks**:
  - [ ] WinRT bindings
  - [ ] Language pack detection
  - [ ] Async processing
  - [ ] Platform detection (skip on non-Windows)

### Priority 5: Specialized Providers (Week 5)

#### 3.11 Surya OCR
- **File**: `ocr/providers/surya_provider.py`
- **Dependencies**: `surya-ocr`
- **Tasks**:
  - [ ] Model download/caching
  - [ ] Line detection
  - [ ] Text recognition
  - [ ] Layout analysis

#### 3.12 OCR.space API
- **File**: `ocr/providers/ocrspace.py`
- **Dependencies**: `requests`
- **Tasks**:
  - [ ] Free tier implementation
  - [ ] API key for premium
  - [ ] Engine selection (1, 2, 3)
  - [ ] Table detection

---

## Phase 4: Testing Infrastructure

### 4.1 Test Image Collection

Create a diverse test set in `samples/`:

```
samples/
|-- english/
|   |-- printed_clean.jpg      # Clean printed text
|   |-- printed_noisy.jpg      # Noisy/low quality
|   |-- handwritten.jpg        # Handwritten text
|   |-- mixed_fonts.jpg        # Multiple fonts
|
|-- hebrew/
|   |-- printed_modern.jpg     # Modern Hebrew print
|   |-- printed_classic.jpg    # Classic Hebrew text
|   |-- mixed_heb_eng.jpg      # Hebrew + English
|   |-- nikud.jpg              # With vowel points
|
|-- documents/
|   |-- invoice.jpg            # Invoice/receipt
|   |-- form.jpg               # Form with fields
|   |-- table.jpg              # Table data
|   |-- multi_column.jpg       # Multi-column layout
|
|-- challenging/
|   |-- rotated.jpg            # Rotated text
|   |-- curved.jpg             # Curved/warped
|   |-- low_contrast.jpg       # Low contrast
|   |-- partial.jpg            # Partially visible
```

### 4.2 Ground Truth Files

For each test image, create a `.txt` file with the expected text:

```
samples/english/printed_clean.txt
samples/hebrew/printed_modern.txt
...
```

### 4.3 Comparison Script

Create `compare_providers.py`:

```python
# Usage:
# python compare_providers.py samples/english/printed_clean.jpg
# python compare_providers.py samples/ --all-providers --output results/

Features:
- Run single image through all providers
- Run test suite through selected providers
- Generate comparison report (accuracy, speed, cost)
- Export results to JSON/CSV
- Side-by-side text comparison
- Character Error Rate (CER) calculation
- Word Error Rate (WER) calculation
```

### 4.4 Metrics to Track

| Metric | Description | Calculation |
|--------|-------------|-------------|
| **CER** | Character Error Rate | Levenshtein distance / total chars |
| **WER** | Word Error Rate | Word-level edit distance / total words |
| **Processing Time** | End-to-end latency | Milliseconds |
| **Confidence** | Provider's confidence score | 0.0 - 1.0 |
| **Cost** | API cost per image | USD |
| **Language Accuracy** | Correct language detection | Boolean |

---

## Phase 5: Integration with TTS

### 5.1 OCR-to-TTS Pipeline

```python
# Full pipeline: Image -> OCR -> Cleanup -> TTS -> Audio

async def ocr_to_speech(
    image_path: str,
    ocr_provider: str = "google_vision",
    cleanup_config: CleanupConfig = None,
    play_audio: bool = True
) -> dict:
    """
    Complete OCR to speech pipeline.

    Returns:
        {
            "ocr_result": OCRResult,
            "cleaned_text": str,
            "tts_success": bool,
            "total_time_ms": float
        }
    """
```

### 5.2 Provider Comparison with TTS

Compare how OCR quality affects TTS output:
- Same image through different OCR providers
- Same cleanup config
- Listen to TTS output for each
- Rate intelligibility

---

## Phase 6: Documentation and Reporting

### 6.1 Provider Documentation

For each provider, document:
- Installation instructions
- Configuration options
- Supported languages
- Known limitations
- Cost structure
- Performance characteristics

### 6.2 Comparison Report

Generate a final report with:
- Accuracy rankings by document type
- Speed benchmarks
- Cost analysis
- Recommendations by use case:
  - Best for English printed text
  - Best for Hebrew
  - Best for handwriting
  - Best for forms/tables
  - Best offline option
  - Best cost/accuracy ratio

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
- [x] Google Cloud Vision (implemented)
- [ ] Tesseract OCR
- [ ] EasyOCR
- [ ] PaddleOCR
- [ ] Windows OCR
- [ ] Surya OCR

### Providers - Cloud
- [x] Google Cloud Vision
- [ ] Azure Computer Vision
- [ ] AWS Textract
- [ ] OpenAI GPT-4 Vision
- [ ] Google Gemini Vision
- [ ] Anthropic Claude Vision
- [ ] OCR.space

### Testing
- [ ] Collect test images (10+ per category)
- [ ] Create ground truth transcriptions
- [ ] Run initial benchmarks
- [ ] Generate comparison report

### Integration
- [ ] OCR-to-TTS pipeline
- [ ] Provider selection logic
- [ ] Fallback chain implementation

---

## Notes

### Hebrew OCR Considerations
- Tesseract Hebrew (heb) language pack quality varies
- Google Cloud Vision has good Hebrew support
- Right-to-left text handling
- Nikud (vowel points) recognition
- Mixed Hebrew/English documents

### Cost Optimization
- Use local providers for bulk processing
- Cache OCR results
- Batch requests where possible
- Monitor API usage

### Performance Tips
- Pre-warm ML models (EasyOCR, PaddleOCR)
- Use GPU acceleration where available
- Async processing for cloud providers
- Image preprocessing can improve accuracy

---

## Timeline Estimate

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Research | 2-3 days | Provider comparison matrix |
| Interface Design | 1 day | Base classes, registry |
| Priority 1 Providers | 3-4 days | Tesseract + Google Vision |
| Priority 2 Providers | 3-4 days | Azure + AWS |
| Priority 3 Providers | 3-4 days | LLM Vision providers |
| Priority 4 Providers | 3-4 days | EasyOCR, PaddleOCR, Windows |
| Testing Infrastructure | 2-3 days | Test suite, metrics |
| Comparison Report | 1-2 days | Final analysis |

**Total: ~3-4 weeks for comprehensive implementation**

---

## Next Steps

1. **Immediate**: Implement Tesseract provider (most common local option)
2. **Then**: Build comparison infrastructure
3. **Then**: Add providers one by one, testing as we go
4. **Finally**: Generate comprehensive comparison report
