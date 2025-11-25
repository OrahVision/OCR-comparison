"""
AWS Textract OCR Provider

Amazon's OCR service for document text extraction.
https://aws.amazon.com/textract/

License: Commercial API
Pricing: $1.50/1K pages (DetectDocumentText), $15/1K (AnalyzeDocument)
"""

import logging
import os
from typing import List, Optional

from ..base import OCRProvider, OCRResult, BoundingBox, ProviderType, Timer
from ..registry import register_provider

# Check for boto3
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logging.info("boto3 not installed. Install with: pip install boto3")


def check_aws_credentials() -> bool:
    """Check if AWS credentials are configured."""
    if not BOTO3_AVAILABLE:
        return False

    try:
        # Try to create a session and check for credentials
        session = boto3.Session()
        credentials = session.get_credentials()
        return credentials is not None
    except Exception:
        return False


TEXTRACT_AVAILABLE = BOTO3_AVAILABLE and check_aws_credentials()


@register_provider
class AWSTextractOCR(OCRProvider):
    """
    AWS Textract OCR Provider

    Features:
    - Synchronous API for single-page documents
    - DetectDocumentText for basic OCR
    - AnalyzeDocument for tables, forms, queries
    - Supports 6 languages (English, Spanish, German, French, Italian, Portuguese)
    """

    PROVIDER_NAME = "aws_textract"
    PROVIDER_TYPE = ProviderType.CLOUD
    COMMERCIAL_USE = True
    LICENSE = "Commercial API"

    # Pricing per 1K pages
    COST_DETECT_TEXT = 1.50  # DetectDocumentText
    COST_ANALYZE_DOC = 15.00  # AnalyzeDocument with tables/forms

    SUPPORTED_LANGUAGES = ["en", "es", "de", "fr", "it", "pt"]

    def __init__(
        self,
        region_name: str = None,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        **kwargs
    ):
        """
        Initialize AWS Textract provider.

        Args:
            region_name: AWS region (default: from env/config)
            aws_access_key_id: AWS access key (default: from env/config)
            aws_secret_access_key: AWS secret key (default: from env/config)
        """
        self._region = region_name
        self._access_key = aws_access_key_id
        self._secret_key = aws_secret_access_key
        self._initialized = False
        self._init_error = None
        self._client = None

        if BOTO3_AVAILABLE:
            self._initialize()

    def _initialize(self):
        """Initialize the Textract client."""
        if self._initialized:
            return

        try:
            # Build kwargs for client
            kwargs = {}
            if self._region:
                kwargs['region_name'] = self._region
            if self._access_key and self._secret_key:
                kwargs['aws_access_key_id'] = self._access_key
                kwargs['aws_secret_access_key'] = self._secret_key

            self._client = boto3.client('textract', **kwargs)

            # Test connection by checking if we can make a simple call
            # (We'll verify on first actual use)
            self._initialized = True
            logging.info(f"AWS Textract initialized")

        except NoCredentialsError:
            self._init_error = (
                "AWS credentials not found. Configure via:\n"
                "- Environment: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\n"
                "- AWS CLI: aws configure\n"
                "- IAM role (for EC2/Lambda)"
            )
            logging.error(self._init_error)
        except Exception as e:
            self._init_error = str(e)
            logging.error(f"Failed to initialize AWS Textract: {e}")

    def is_available(self) -> bool:
        """Check if AWS Textract is available."""
        if not BOTO3_AVAILABLE:
            return False
        if not self._initialized:
            self._initialize()
        return self._initialized and self._client is not None

    async def process_image(self, image_path: str, **options) -> OCRResult:
        """
        Process an image with AWS Textract.

        Args:
            image_path: Path to the image file
            **options:
                analyze: Use AnalyzeDocument instead of DetectDocumentText
                features: List of features for AnalyzeDocument (TABLES, FORMS, etc.)

        Returns:
            OCRResult with extracted text
        """
        if not os.path.exists(image_path):
            return self._create_error_result(f"Image not found: {image_path}")

        if not self.is_available():
            error = self._init_error or "AWS Textract not available"
            return self._create_error_result(error)

        with Timer() as timer:
            try:
                # Read image file
                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                # Check file size (Textract has 10MB limit for sync API)
                if len(image_bytes) > 10 * 1024 * 1024:
                    return self._create_error_result(
                        "Image too large for Textract sync API (max 10MB)"
                    )

                # Determine which API to use
                use_analyze = options.get('analyze', False)
                features = options.get('features', ['TABLES', 'FORMS'])

                if use_analyze:
                    response = self._client.analyze_document(
                        Document={'Bytes': image_bytes},
                        FeatureTypes=features
                    )
                    cost_per_page = self.COST_ANALYZE_DOC / 1000
                else:
                    response = self._client.detect_document_text(
                        Document={'Bytes': image_bytes}
                    )
                    cost_per_page = self.COST_DETECT_TEXT / 1000

                # Process response
                text_parts = []
                bounding_boxes = []
                confidences = []

                for block in response.get('Blocks', []):
                    block_type = block.get('BlockType')

                    if block_type == 'LINE':
                        text = block.get('Text', '')
                        if text:
                            text_parts.append(text)

                    if block_type == 'WORD':
                        text = block.get('Text', '')
                        confidence = block.get('Confidence', 0) / 100.0
                        confidences.append(confidence)

                        # Get bounding box
                        bbox = block.get('Geometry', {}).get('BoundingBox', {})
                        if bbox:
                            # Textract returns normalized coordinates (0-1)
                            # We'll store them as-is; caller can scale to image dimensions
                            bounding_boxes.append(BoundingBox(
                                x=int(bbox.get('Left', 0) * 1000),
                                y=int(bbox.get('Top', 0) * 1000),
                                width=int(bbox.get('Width', 0) * 1000),
                                height=int(bbox.get('Height', 0) * 1000),
                                text=text,
                                confidence=confidence
                            ))

                full_text = '\n'.join(text_parts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                return self._create_success_result(
                    text=full_text.strip(),
                    processing_time_ms=timer.elapsed_ms,
                    confidence=avg_confidence,
                    bounding_boxes=bounding_boxes if bounding_boxes else None,
                    estimated_cost_usd=cost_per_page,
                    raw_response=response
                )

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_msg = e.response.get('Error', {}).get('Message', str(e))
                logging.error(f"AWS Textract error ({error_code}): {error_msg}")
                return self._create_error_result(f"{error_code}: {error_msg}", timer.elapsed_ms)
            except Exception as e:
                logging.error(f"AWS Textract processing error: {e}")
                return self._create_error_result(str(e), timer.elapsed_ms)

    def get_supported_languages(self) -> List[str]:
        """Return list of supported languages (limited for Textract)."""
        return self.SUPPORTED_LANGUAGES

    def estimate_cost(self, num_pages: int = 1, analyze: bool = False) -> float:
        """Estimate cost for processing pages."""
        cost_per_1k = self.COST_ANALYZE_DOC if analyze else self.COST_DETECT_TEXT
        return (num_pages / 1000) * cost_per_1k


# CLI interface for testing
if __name__ == "__main__":
    import sys

    def print_help():
        print("AWS Textract OCR Provider")
        print("")
        print("Usage:")
        print("  python -m ocr.providers.aws_textract test <image>  - Test OCR")
        print("  python -m ocr.providers.aws_textract check         - Check availability")
        print("")
        print("Options:")
        print("  --analyze    Use AnalyzeDocument API (tables/forms)")
        print("")
        print("Environment:")
        print("  AWS_ACCESS_KEY_ID - AWS access key")
        print("  AWS_SECRET_ACCESS_KEY - AWS secret key")
        print("  AWS_DEFAULT_REGION - AWS region")
        print("  Or configure via: aws configure")

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "check":
        print(f"boto3 installed: {BOTO3_AVAILABLE}")
        print(f"AWS credentials found: {check_aws_credentials()}")
        if BOTO3_AVAILABLE:
            provider = AWSTextractOCR()
            print(f"Provider available: {provider.is_available()}")
            if provider._init_error:
                print(f"Init error: {provider._init_error}")
        sys.exit(0)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Error: Please provide an image path")
            sys.exit(1)

        image_path = sys.argv[2]

        # Parse optional arguments
        use_analyze = "--analyze" in sys.argv

        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        print(f"Testing AWS Textract on: {image_path}")
        print(f"API: {'AnalyzeDocument' if use_analyze else 'DetectDocumentText'}")
        print("")

        provider = AWSTextractOCR()

        if not provider.is_available():
            print(f"Error: AWS Textract not available")
            if provider._init_error:
                print(f"Reason: {provider._init_error}")
            sys.exit(1)

        result = provider.process_image_sync(image_path, analyze=use_analyze)

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
