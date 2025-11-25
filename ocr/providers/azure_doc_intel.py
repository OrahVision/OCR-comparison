"""
Azure Document Intelligence OCR Provider

Enterprise-grade OCR from Microsoft Azure.
https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence

License: Commercial API
Pricing: ~$1.50/1K pages (Read API), $10-65/1K pages (Custom models)
"""

import logging
import os
from typing import List, Optional

from ..base import OCRProvider, OCRResult, BoundingBox, ProviderType, Timer
from ..registry import register_provider

# Check for keyring
try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

# Check for Azure SDK
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
    AZURE_DOC_AVAILABLE = True
except ImportError:
    AZURE_DOC_AVAILABLE = False
    logging.info("azure-ai-documentintelligence not installed. Install with: pip install azure-ai-documentintelligence")


def get_azure_credentials() -> tuple:
    """Get Azure endpoint and key from keyring or environment."""
    endpoint = None
    key = None

    # Try keyring first
    if KEYRING_AVAILABLE:
        try:
            endpoint = keyring.get_password("azure_doc_intel", "endpoint")
            key = keyring.get_password("azure_doc_intel", "api_key")
        except Exception:
            pass

    # Try environment variables
    if not endpoint:
        endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    if not key:
        key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    return endpoint, key


AZURE_AVAILABLE = AZURE_DOC_AVAILABLE and all(get_azure_credentials())


@register_provider
class AzureDocIntelOCR(OCRProvider):
    """
    Azure Document Intelligence OCR Provider

    Features:
    - Enterprise-grade document processing
    - Multiple models: Read (OCR), Layout (structure), Custom
    - 100+ language support
    - Table extraction
    - Handwriting recognition
    """

    PROVIDER_NAME = "azure_doc_intel"
    PROVIDER_TYPE = ProviderType.CLOUD
    COMMERCIAL_USE = True
    LICENSE = "Commercial API"

    # Pricing per 1K pages
    COST_PER_1K_PAGES = 1.50  # Read API

    def __init__(
        self,
        endpoint: str = None,
        api_key: str = None,
        model: str = "prebuilt-read",
        **kwargs
    ):
        """
        Initialize Azure Document Intelligence provider.

        Args:
            endpoint: Azure endpoint URL
            api_key: Azure API key
            model: Model to use (prebuilt-read, prebuilt-layout, etc.)
        """
        self._endpoint = endpoint
        self._api_key = api_key
        self.model = model
        self._initialized = False
        self._init_error = None
        self._client = None

        if AZURE_DOC_AVAILABLE:
            self._initialize()

    def _initialize(self):
        """Initialize the Azure client."""
        if self._initialized:
            return

        try:
            endpoint = self._endpoint
            key = self._api_key

            if not endpoint or not key:
                env_endpoint, env_key = get_azure_credentials()
                endpoint = endpoint or env_endpoint
                key = key or env_key

            if not endpoint or not key:
                self._init_error = (
                    "Azure credentials not found. Set via keyring (azure_doc_intel/endpoint, azure_doc_intel/api_key) "
                    "or AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY env vars."
                )
                return

            self._client = DocumentIntelligenceClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(key)
            )
            self._initialized = True
            logging.info(f"Azure Document Intelligence initialized with model: {self.model}")

        except Exception as e:
            self._init_error = str(e)
            logging.error(f"Failed to initialize Azure Document Intelligence: {e}")

    def is_available(self) -> bool:
        """Check if Azure Document Intelligence is available."""
        if not AZURE_DOC_AVAILABLE:
            return False
        if not self._initialized:
            self._initialize()
        return self._initialized and self._client is not None

    async def process_image(self, image_path: str, **options) -> OCRResult:
        """
        Process an image with Azure Document Intelligence.

        Args:
            image_path: Path to the image file
            **options:
                model: Override model (prebuilt-read, prebuilt-layout)

        Returns:
            OCRResult with extracted text
        """
        if not os.path.exists(image_path):
            return self._create_error_result(f"Image not found: {image_path}")

        if not self.is_available():
            error = self._init_error or "Azure Document Intelligence not available"
            return self._create_error_result(error)

        with Timer() as timer:
            try:
                # Read image file
                with open(image_path, "rb") as f:
                    image_data = f.read()

                # Get model
                model = options.get("model", self.model)

                # Analyze document
                poller = self._client.begin_analyze_document(
                    model,
                    body=image_data,
                    content_type="application/octet-stream"
                )
                result = poller.result()

                # Extract text and bounding boxes
                text_parts = []
                bounding_boxes = []
                confidences = []

                if result.content:
                    text_parts.append(result.content)

                # Process pages for detailed info
                if result.pages:
                    for page in result.pages:
                        if page.words:
                            for word in page.words:
                                conf = word.confidence or 0.0
                                confidences.append(conf)

                                if word.polygon and len(word.polygon) >= 4:
                                    # polygon is list of points
                                    xs = [p.x for p in word.polygon]
                                    ys = [p.y for p in word.polygon]
                                    x = int(min(xs))
                                    y = int(min(ys))
                                    width = int(max(xs) - x)
                                    height = int(max(ys) - y)

                                    bounding_boxes.append(BoundingBox(
                                        x=x, y=y, width=width, height=height,
                                        text=word.content,
                                        confidence=conf
                                    ))

                text = result.content or ""
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                # Estimate cost
                estimated_cost = self.COST_PER_1K_PAGES / 1000  # per page

                return self._create_success_result(
                    text=text.strip(),
                    processing_time_ms=timer.elapsed_ms,
                    confidence=avg_confidence,
                    bounding_boxes=bounding_boxes if bounding_boxes else None,
                    estimated_cost_usd=estimated_cost,
                    raw_response={"model": model}
                )

            except Exception as e:
                logging.error(f"Azure Document Intelligence processing error: {e}")
                return self._create_error_result(str(e), timer.elapsed_ms)

    def get_supported_languages(self) -> List[str]:
        """Return list of supported languages."""
        return [
            "en", "he", "ar", "zh-Hans", "zh-Hant", "ja", "ko",
            "fr", "de", "es", "it", "pt", "ru", "nl", "pl", "sv"
        ]

    def estimate_cost(self, num_pages: int = 1) -> float:
        """Estimate cost for processing pages."""
        return (num_pages / 1000) * self.COST_PER_1K_PAGES


# CLI interface for testing
if __name__ == "__main__":
    import sys

    def print_help():
        print("Azure Document Intelligence OCR Provider")
        print("")
        print("Usage:")
        print("  python -m ocr.providers.azure_doc_intel test <image>  - Test OCR")
        print("  python -m ocr.providers.azure_doc_intel check         - Check availability")
        print("")
        print("Environment:")
        print("  AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT - Azure endpoint URL")
        print("  AZURE_DOCUMENT_INTELLIGENCE_KEY - Azure API key")
        print("  Or use keyring:")
        print("    keyring set azure_doc_intel endpoint")
        print("    keyring set azure_doc_intel api_key")

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "check":
        print(f"azure-ai-documentintelligence installed: {AZURE_DOC_AVAILABLE}")
        endpoint, key = get_azure_credentials()
        print(f"Endpoint found: {endpoint is not None}")
        print(f"API key found: {key is not None}")
        if AZURE_DOC_AVAILABLE:
            provider = AzureDocIntelOCR()
            print(f"Provider available: {provider.is_available()}")
            if provider._init_error:
                print(f"Init error: {provider._init_error}")
        sys.exit(0)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Error: Please provide an image path")
            sys.exit(1)

        image_path = sys.argv[2]

        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        print(f"Testing Azure Document Intelligence on: {image_path}")
        print("")

        provider = AzureDocIntelOCR()

        if not provider.is_available():
            print(f"Error: Azure Document Intelligence not available")
            if provider._init_error:
                print(f"Reason: {provider._init_error}")
            sys.exit(1)

        result = provider.process_image_sync(image_path)

        if result.success:
            print("SUCCESS!")
            print(f"Processing time: {result.processing_time_ms:.0f}ms")
            print(f"Words detected: {result.word_count}")
            print(f"Confidence: {result.confidence:.1%}")
            print(f"Est. cost: ${result.estimated_cost_usd:.6f}")
            print("")
            print("Extracted text:")
            print("-" * 50)
            print(result.text[:2000] + ("..." if len(result.text) > 2000 else ""))
            sys.exit(0)
        else:
            print(f"FAILED: {result.error}")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        print_help()
        sys.exit(1)
