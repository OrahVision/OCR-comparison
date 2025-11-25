#!/usr/bin/env python3
"""
Google Cloud Vision OCR Provider

This module provides OCR capabilities using Google Cloud Vision API.
Supports multiple authentication methods:
- Keyring (secure credential storage)
- Environment variable (GOOGLE_APPLICATION_CREDENTIALS)
- Application default credentials (gcloud auth)
- API key (REST API)
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any, List
import base64
import json

# For secure credential storage
try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    logging.info("keyring not available for secure credential storage")

# Google Cloud Vision for OCR
try:
    from google.cloud import vision
    from google.oauth2 import service_account
    VISION_SDK_AVAILABLE = True
except ImportError:
    VISION_SDK_AVAILABLE = False
    logging.info("google-cloud-vision SDK not installed. Will use REST API if API key provided.")

# For REST API calls
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logging.warning("aiohttp not installed - required for API key authentication")


class GoogleVisionOCR:
    """
    Google Cloud Vision OCR Provider

    Handles OCR operations using Google Cloud Vision API with multiple
    authentication methods.
    """

    PROVIDER_NAME = "google_vision"

    def __init__(self, use_keyring: bool = True, api_key: str = None):
        """
        Initialize the Google Vision OCR provider.

        Args:
            use_keyring: Whether to try loading credentials from keyring first
            api_key: Optional API key to use directly (REST API mode)
        """
        self.client = None
        self.api_key = api_key
        self.use_api_key = False
        self.initialized = False

        # If no API key provided, try to get from keyring or environment
        if not api_key:
            api_key = self._get_api_key_from_keyring_or_env()
            self.api_key = api_key

        # If API key is available, use REST API mode (simpler, no SDK needed)
        if self.api_key:
            self.use_api_key = True
            self.initialized = True
            logging.info("Google Vision OCR initialized with API key (REST mode)")
            return

        # Otherwise, try SDK with various credential sources
        if VISION_SDK_AVAILABLE:
            try:
                credentials = None

                # 1. Try keyring first if available
                if use_keyring and KEYRING_AVAILABLE:
                    credentials = self._get_credentials_from_keyring()

                # 2. Try environment variable
                if not credentials:
                    credentials = self._get_credentials_from_env()

                # 3. Try application default credentials
                if not credentials:
                    self.client = vision.ImageAnnotatorClient()
                    logging.info("Google Vision OCR initialized with default credentials")
                else:
                    self.client = vision.ImageAnnotatorClient(credentials=credentials)
                    logging.info("Google Vision OCR initialized with custom credentials")

                self.initialized = True

            except Exception as e:
                logging.error(f"Failed to initialize Google Cloud Vision client: {e}")
                logging.error("Configure credentials using one of these methods:")
                logging.error("1. Store in keyring: python -m ocr.providers.google_vision store")
                logging.error("2. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
                logging.error("3. Run: gcloud auth application-default login")
        else:
            logging.warning("Google Vision SDK not available and no API key provided")

    def _get_credentials_from_keyring(self):
        """Load Google Cloud credentials from keyring."""
        try:
            stored_creds = keyring.get_password("google_cloud_vision", "service_account_json")
            if stored_creds:
                creds_dict = json.loads(stored_creds)
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                logging.info("Loaded Google Cloud credentials from keyring")
                return credentials
        except Exception as e:
            logging.debug(f"Could not load credentials from keyring: {e}")
        return None

    def _get_credentials_from_env(self):
        """Load Google Cloud credentials from environment variable."""
        try:
            creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if creds_path and os.path.exists(creds_path):
                credentials = service_account.Credentials.from_service_account_file(creds_path)
                logging.info(f"Loaded Google Cloud credentials from {creds_path}")
                return credentials
        except Exception as e:
            logging.debug(f"Could not load credentials from environment: {e}")
        return None

    def _get_api_key_from_keyring_or_env(self) -> Optional[str]:
        """
        Try to get Google API key from keyring or environment.

        Checks in order:
        1. Keyring: gemini/api_key (shared with Gemini/TTS)
        2. Keyring: google/api_key
        3. Environment: GEMINI_API_KEY
        4. Environment: GOOGLE_API_KEY

        Returns:
            API key string or None
        """
        # Try keyring first
        if KEYRING_AVAILABLE:
            try:
                # Check gemini keyring (shared API key)
                api_key = keyring.get_password("gemini", "api_key")
                if api_key:
                    logging.info("Found API key in keyring (gemini/api_key)")
                    return api_key

                # Check google keyring
                api_key = keyring.get_password("google", "api_key")
                if api_key:
                    logging.info("Found API key in keyring (google/api_key)")
                    return api_key
            except Exception as e:
                logging.debug(f"Could not get API key from keyring: {e}")

        # Try environment variables
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            logging.info("Found API key in GEMINI_API_KEY environment variable")
            return api_key

        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            logging.info("Found API key in GOOGLE_API_KEY environment variable")
            return api_key

        return None

    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        return self.initialized

    async def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image for OCR.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing:
                - success: bool
                - text: extracted text
                - word_count: number of words
                - confidence: average confidence score
                - pages: structured page data (optional)
                - error: error message (if failed)
                - provider: provider name
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "Google Vision OCR not initialized",
                "text": "",
                "provider": self.PROVIDER_NAME
            }

        # Use appropriate method based on auth mode
        if self.use_api_key:
            return await self._process_with_api_key(image_path)
        else:
            return await self._process_with_sdk(image_path)

    async def _process_with_sdk(self, image_path: str) -> Dict[str, Any]:
        """Process image using the Google Cloud Vision SDK."""
        if not self.client:
            return {
                "success": False,
                "error": "Vision client not initialized",
                "text": "",
                "provider": self.PROVIDER_NAME
            }

        try:
            # Read the image file
            with open(image_path, 'rb') as image_file:
                content = image_file.read()

            # Create Vision API image object
            image = vision.Image(content=content)

            # Perform document text detection
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.document_text_detection,
                image
            )

            # Extract the full text
            full_text = response.full_text_annotation.text if response.full_text_annotation else ""

            # Process into structured format
            pages = []
            if response.full_text_annotation:
                for page in response.full_text_annotation.pages:
                    page_data = {
                        "blocks": [],
                        "confidence": page.confidence if hasattr(page, 'confidence') else 1.0
                    }

                    for block in page.blocks:
                        block_text = []
                        for paragraph in block.paragraphs:
                            para_text = []
                            for word in paragraph.words:
                                word_text = ''.join([
                                    symbol.text for symbol in word.symbols
                                ])
                                para_text.append(word_text)
                            block_text.append(' '.join(para_text))

                        page_data["blocks"].append({
                            "text": '\n'.join(block_text),
                            "confidence": block.confidence if hasattr(block, 'confidence') else 1.0
                        })

                    pages.append(page_data)

            # Check for errors
            if response.error.message:
                logging.error(f"Vision API error: {response.error.message}")
                return {
                    "success": False,
                    "error": response.error.message,
                    "text": "",
                    "provider": self.PROVIDER_NAME
                }

            return {
                "success": True,
                "text": full_text,
                "pages": pages,
                "word_count": len(full_text.split()) if full_text else 0,
                "confidence": sum(p.get("confidence", 1.0) for p in pages) / len(pages) if pages else 0,
                "provider": self.PROVIDER_NAME
            }

        except Exception as e:
            logging.error(f"Error performing OCR with SDK: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "provider": self.PROVIDER_NAME
            }

    async def _process_with_api_key(self, image_path: str) -> Dict[str, Any]:
        """Process image using REST API with API key authentication."""
        if not AIOHTTP_AVAILABLE:
            return {
                "success": False,
                "error": "aiohttp not installed - required for API key mode",
                "text": "",
                "provider": self.PROVIDER_NAME
            }

        try:
            # Read and encode the image
            with open(image_path, 'rb') as image_file:
                content = image_file.read()

            image_base64 = base64.b64encode(content).decode('utf-8')

            # Prepare the request
            url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"

            payload = {
                "requests": [{
                    "image": {"content": image_base64},
                    "features": [{
                        "type": "DOCUMENT_TEXT_DETECTION",
                        "maxResults": 1
                    }]
                }]
            }

            # Make the request
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logging.error(f"Vision API error: {error_text}")
                        return {
                            "success": False,
                            "error": f"API request failed: {response.status}",
                            "text": "",
                            "provider": self.PROVIDER_NAME
                        }

                    result = await response.json()

            # Parse the response
            if "responses" in result and len(result["responses"]) > 0:
                response_data = result["responses"][0]

                if "error" in response_data:
                    return {
                        "success": False,
                        "error": response_data["error"].get("message", "Unknown error"),
                        "text": "",
                        "provider": self.PROVIDER_NAME
                    }

                # Extract text
                full_text = ""
                pages = []

                if "fullTextAnnotation" in response_data:
                    annotation = response_data["fullTextAnnotation"]
                    full_text = annotation.get("text", "")

                    # Process pages
                    for page in annotation.get("pages", []):
                        page_data = {
                            "blocks": [],
                            "confidence": page.get("confidence", 1.0)
                        }

                        for block in page.get("blocks", []):
                            block_text = []
                            for paragraph in block.get("paragraphs", []):
                                para_words = []
                                for word in paragraph.get("words", []):
                                    word_text = ''.join([
                                        symbol.get("text", "") for symbol in word.get("symbols", [])
                                    ])
                                    para_words.append(word_text)
                                block_text.append(' '.join(para_words))

                            if block_text:
                                page_data["blocks"].append({
                                    "text": '\n'.join(block_text),
                                    "confidence": block.get("confidence", 1.0)
                                })

                        pages.append(page_data)

                return {
                    "success": True,
                    "text": full_text,
                    "pages": pages,
                    "word_count": len(full_text.split()) if full_text else 0,
                    "confidence": sum(p.get("confidence", 1.0) for p in pages) / len(pages) if pages else 0,
                    "provider": self.PROVIDER_NAME
                }
            else:
                return {
                    "success": False,
                    "error": "No response from Vision API",
                    "text": "",
                    "provider": self.PROVIDER_NAME
                }

        except Exception as e:
            logging.error(f"Error in API key OCR processing: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "provider": self.PROVIDER_NAME
            }

    def process_image_sync(self, image_path: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for process_image.

        Args:
            image_path: Path to the image file

        Returns:
            OCR result dictionary
        """
        return asyncio.run(self.process_image(image_path))


# Credential management functions

def store_gcp_credentials(json_path: Optional[str] = None) -> bool:
    """
    Store Google Cloud service account credentials in the system keyring.

    Args:
        json_path: Path to the service account JSON file. If not provided, will prompt.

    Returns:
        True if successful, False otherwise
    """
    if not KEYRING_AVAILABLE:
        print("Error: keyring module not available. Install with: pip install keyring")
        return False

    try:
        # Get the JSON file path
        if not json_path:
            json_path = input("Enter path to Google Cloud service account JSON file: ").strip()
            if json_path.startswith('"') and json_path.endswith('"'):
                json_path = json_path[1:-1]

        # Verify file exists
        if not os.path.exists(json_path):
            print(f"Error: File not found: {json_path}")
            return False

        # Read and validate the JSON
        with open(json_path, 'r') as f:
            creds_dict = json.load(f)

        # Basic validation
        required_fields = ['type', 'project_id', 'private_key', 'client_email']
        missing_fields = [field for field in required_fields if field not in creds_dict]
        if missing_fields:
            print(f"Error: Invalid service account JSON. Missing fields: {missing_fields}")
            return False

        if creds_dict.get('type') != 'service_account':
            print(f"Error: JSON file is not a service account key (type: {creds_dict.get('type')})")
            return False

        # Store in keyring
        creds_json = json.dumps(creds_dict)
        keyring.set_password("google_cloud_vision", "service_account_json", creds_json)

        print(f"SUCCESS: Stored Google Cloud credentials in keyring")
        print(f"   Project: {creds_dict.get('project_id')}")
        print(f"   Service Account: {creds_dict.get('client_email')}")
        print("\nThe original JSON file can now be deleted or moved to a secure location.")

        return True

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        return False
    except Exception as e:
        print(f"Error storing credentials: {e}")
        return False


def remove_gcp_credentials() -> bool:
    """Remove Google Cloud credentials from keyring."""
    if not KEYRING_AVAILABLE:
        print("Error: keyring module not available")
        return False

    try:
        keyring.delete_password("google_cloud_vision", "service_account_json")
        print("SUCCESS: Google Cloud credentials removed from keyring")
        return True
    except Exception as e:
        print(f"Error removing credentials: {e}")
        return False


def check_google_cloud_auth() -> bool:
    """
    Check if Google Cloud authentication is configured.

    Returns:
        True if authentication is configured, False otherwise
    """
    # Check keyring
    if KEYRING_AVAILABLE:
        try:
            stored_creds = keyring.get_password("google_cloud_vision", "service_account_json")
            if stored_creds:
                print("Google Cloud credentials found in keyring")
                return True
        except:
            pass

    # Check environment variable
    adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if adc_path and os.path.exists(adc_path):
        print(f"Google Cloud credentials found at: {adc_path}")
        return True

    # Check gcloud auth
    try:
        import subprocess
        result = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("Google Cloud authentication configured via gcloud")
            return True
    except:
        pass

    print("Google Cloud authentication not configured")
    print("\nTo configure, use one of these methods:")
    print("1. python -m ocr.providers.google_vision store <service-account.json>")
    print("2. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
    print("3. Run: gcloud auth application-default login")

    return False


# Exports
__all__ = [
    'GoogleVisionOCR',
    'store_gcp_credentials',
    'remove_gcp_credentials',
    'check_google_cloud_auth',
    'VISION_SDK_AVAILABLE',
    'KEYRING_AVAILABLE',
    'AIOHTTP_AVAILABLE'
]


# CLI interface
if __name__ == "__main__":
    import sys

    def print_help():
        print("Google Cloud Vision OCR Provider")
        print("")
        print("Usage:")
        print("  python -m ocr.providers.google_vision store [json_file]  - Store credentials")
        print("  python -m ocr.providers.google_vision remove             - Remove credentials")
        print("  python -m ocr.providers.google_vision check              - Check auth status")
        print("  python -m ocr.providers.google_vision test <image>       - Test OCR")
        print("")
        print("Examples:")
        print("  python -m ocr.providers.google_vision store ~/service-account.json")
        print("  python -m ocr.providers.google_vision test document.jpg")

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "store":
        json_path = sys.argv[2] if len(sys.argv) > 2 else None
        success = store_gcp_credentials(json_path)
        sys.exit(0 if success else 1)

    elif command == "remove":
        success = remove_gcp_credentials()
        sys.exit(0 if success else 1)

    elif command == "check":
        success = check_google_cloud_auth()
        sys.exit(0 if success else 1)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Error: Please provide an image path")
            print("Usage: python -m ocr.providers.google_vision test <image_path>")
            sys.exit(1)

        image_path = sys.argv[2]
        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        print(f"Testing OCR on: {image_path}")

        provider = GoogleVisionOCR()
        if not provider.is_available():
            print("Error: Google Vision OCR not available")
            sys.exit(1)

        result = provider.process_image_sync(image_path)

        if result.get("success"):
            print("\nSUCCESS: OCR Test Passed!")
            print(f"Words detected: {result.get('word_count', 0)}")
            print(f"Confidence: {result.get('confidence', 0):.2%}")
            print("\nExtracted text (first 500 chars):")
            print("-" * 50)
            text = result.get('text', '')
            print(text[:500] + ("..." if len(text) > 500 else ""))
            sys.exit(0)
        else:
            print(f"\nFAIL: OCR Test Failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        print()
        print_help()
        sys.exit(1)
