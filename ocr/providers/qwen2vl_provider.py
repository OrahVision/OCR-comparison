"""
Qwen2-VL Provider

Alibaba's vision-language model with strong OCR capabilities.
https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

License: Apache 2.0 for 2B/7B models - Commercial use allowed
Note: 72B model uses different license (Qwen License)

Requirements:
- 2B: ~8GB VRAM
- 7B: ~16GB VRAM
- Supports 4-bit/8-bit quantization
"""

import logging
import os
from typing import List, Optional, Dict, Any

from ..base import OCRProvider, OCRResult, ProviderType, Timer
from ..registry import register_provider

# Check dependencies
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
    QWEN2VL_CLASS_AVAILABLE = True
except ImportError:
    QWEN2VL_CLASS_AVAILABLE = False
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        logging.info("transformers not installed")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Check for qwen-vl-utils (optional, for better image handling)
try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False

QWEN2VL_AVAILABLE = TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE and PIL_AVAILABLE


@register_provider
class Qwen2VLProvider(OCRProvider):
    """
    Qwen2-VL Provider

    Features:
    - Strong OCR capabilities
    - Multilingual support
    - Multiple model sizes (2B, 7B)
    - Apache 2.0 license for commercial use
    """

    PROVIDER_NAME = "qwen2vl"
    PROVIDER_TYPE = ProviderType.LOCAL
    COMMERCIAL_USE = True  # True for 2B/7B, check license for 72B
    LICENSE = "Apache-2.0"

    # Model variants
    MODEL_VARIANTS = {
        "2b": "Qwen/Qwen2-VL-2B-Instruct",
        "7b": "Qwen/Qwen2-VL-7B-Instruct",
        # "72b": "Qwen/Qwen2-VL-72B-Instruct",  # Different license
    }

    # Supports many languages
    SUPPORTED_LANGUAGES = [
        "en",    # English
        "zh",    # Chinese
        "ja",    # Japanese
        "ko",    # Korean
        "ar",    # Arabic
        "he",    # Hebrew
        "ru",    # Russian
        "de",    # German
        "fr",    # French
        "es",    # Spanish
        "pt",    # Portuguese
        "it",    # Italian
        # Many more supported
    ]

    def __init__(
        self,
        model_variant: str = "7b",
        device: str = "auto",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        **config
    ):
        """
        Initialize Qwen2-VL provider.

        Args:
            model_variant: Model size ('2b', '7b')
            device: Device to use ('auto', 'cuda', 'cpu')
            torch_dtype: Data type ('auto', 'float16', 'bfloat16', 'float32')
            load_in_4bit: Use 4-bit quantization
            load_in_8bit: Use 8-bit quantization
            min_pixels: Minimum image resolution (tokens)
            max_pixels: Maximum image resolution (tokens)
        """
        self.model_variant = model_variant
        self.device_config = device
        self.torch_dtype_config = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.model = None
        self.processor = None
        self.device = None
        self._initialized = False
        self._init_error = None

        if model_variant not in self.MODEL_VARIANTS:
            self._init_error = f"Invalid model variant: {model_variant}. Choose from: {list(self.MODEL_VARIANTS.keys())}"

    def _get_model_name(self) -> str:
        """Get the HuggingFace model name for current variant."""
        return self.MODEL_VARIANTS.get(self.model_variant, self.MODEL_VARIANTS["7b"])

    def _initialize_model(self):
        """Initialize the model (lazy loading)."""
        if self._initialized or self._init_error:
            return

        if not QWEN2VL_AVAILABLE:
            self._init_error = "Qwen2-VL dependencies not available"
            return

        try:
            import torch
            from transformers import AutoProcessor

            model_name = self._get_model_name()

            # Determine device
            if self.device_config == "auto":
                self.device = "cuda" if CUDA_AVAILABLE else "cpu"
            else:
                self.device = self.device_config

            # Determine dtype
            if self.torch_dtype_config == "auto":
                if self.device == "cuda":
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float32
            elif self.torch_dtype_config == "float16":
                torch_dtype = torch.float16
            elif self.torch_dtype_config == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

            # Build model kwargs
            model_kwargs = {
                "torch_dtype": torch_dtype,
            }

            # Add quantization if requested
            if self.load_in_4bit or self.load_in_8bit:
                try:
                    from transformers import BitsAndBytesConfig

                    if self.load_in_4bit:
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch_dtype,
                        )
                    elif self.load_in_8bit:
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_8bit=True,
                        )
                except ImportError:
                    logging.warning("bitsandbytes not installed, skipping quantization")

            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"

            logging.info(f"Loading Qwen2-VL model: {model_name}")

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )

            # Load model - try Qwen2VL specific class first
            if QWEN2VL_CLASS_AVAILABLE:
                from transformers import Qwen2VLForConditionalGeneration
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            else:
                from transformers import AutoModelForVision2Seq
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    **model_kwargs
                )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()
            self._initialized = True

            logging.info(
                f"Qwen2-VL initialized: {model_name}, device: {self.device}"
            )

        except Exception as e:
            self._init_error = str(e)
            logging.error(f"Failed to initialize Qwen2-VL: {e}")

    def is_available(self) -> bool:
        """Check if Qwen2-VL is available."""
        if not QWEN2VL_AVAILABLE:
            return False
        if self._init_error:
            return False
        return True

    async def process_image(self, image_path: str, **options) -> OCRResult:
        """
        Process an image with Qwen2-VL.

        Args:
            image_path: Path to the image file
            **options:
                prompt: Custom prompt (default: OCR prompt)
                max_new_tokens: Maximum tokens to generate (default: 2048)
                language: Target language hint

        Returns:
            OCRResult with extracted text
        """
        if not os.path.exists(image_path):
            return self._create_error_result(f"Image not found: {image_path}")

        if not self.is_available():
            error = self._init_error or "Qwen2-VL not available"
            return self._create_error_result(error)

        # Lazy load model
        if not self._initialized:
            self._initialize_model()
            if self._init_error:
                return self._create_error_result(self._init_error)

        with Timer() as timer:
            try:
                import torch
                from PIL import Image

                # Load image
                image = Image.open(image_path).convert("RGB")

                # Options
                max_new_tokens = options.get("max_new_tokens", 2048)
                language = options.get("language", "")
                prompt = options.get("prompt")

                if not prompt:
                    if language:
                        prompt = f"Please perform OCR on this image and extract all text. Output the text in {language}."
                    else:
                        prompt = "Please perform OCR on this image and extract all the text content. Preserve the layout as much as possible."

                # Build messages in Qwen2-VL chat format
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Process inputs
                if QWEN_VL_UTILS_AVAILABLE:
                    from qwen_vl_utils import process_vision_info
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                else:
                    # Fallback without qwen_vl_utils
                    inputs = self.processor(
                        text=[text],
                        images=[image],
                        padding=True,
                        return_tensors="pt",
                    )

                # Move to device
                if self.device == "cuda":
                    inputs = inputs.to(self.model.device)

                # Generate
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                    )

                # Trim input tokens from output
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                # Decode
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

                return self._create_success_result(
                    text=output_text.strip(),
                    processing_time_ms=timer.elapsed_ms,
                    confidence=1.0,
                    tokens_used=generated_ids.shape[1],
                )

            except Exception as e:
                logging.error(f"Qwen2-VL processing error: {e}")
                return self._create_error_result(str(e), timer.elapsed_ms)

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return self.SUPPORTED_LANGUAGES

    def unload_model(self):
        """Unload model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._initialized = False

        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            import torch
            torch.cuda.empty_cache()

        logging.info("Qwen2-VL model unloaded")


# CLI interface for testing
if __name__ == "__main__":
    import sys

    def print_help():
        print("Qwen2-VL Provider")
        print("")
        print("Usage:")
        print("  python -m ocr.providers.qwen2vl_provider test <image>  - Test OCR")
        print("  python -m ocr.providers.qwen2vl_provider check         - Check availability")
        print("")
        print("Options:")
        print("  --variant <name>  Model variant (2b/7b)")
        print("  --4bit            Use 4-bit quantization")
        print("  --8bit            Use 8-bit quantization")
        print("  --cpu             Force CPU mode")
        print("  --lang <code>     Language hint (en, zh, he, etc.)")
        print("")
        print("Examples:")
        print("  python -m ocr.providers.qwen2vl_provider check")
        print("  python -m ocr.providers.qwen2vl_provider test doc.jpg --variant 2b --4bit")

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "check":
        print(f"PyTorch installed: {TORCH_AVAILABLE}")
        print(f"Transformers installed: {TRANSFORMERS_AVAILABLE}")
        print(f"Qwen2VL class available: {QWEN2VL_CLASS_AVAILABLE}")
        print(f"qwen_vl_utils available: {QWEN_VL_UTILS_AVAILABLE}")
        print(f"PIL installed: {PIL_AVAILABLE}")
        print(f"CUDA available: {CUDA_AVAILABLE}")
        print("")

        if QWEN2VL_AVAILABLE:
            print("Qwen2-VL dependencies OK")
            print("")
            print("Recommended for your hardware:")
            if CUDA_AVAILABLE:
                import torch
                mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if mem >= 16:
                    print("  7B model should work (16GB+ VRAM)")
                elif mem >= 8:
                    print("  2B model recommended, or 7B with --4bit")
                else:
                    print("  Try 2B with --4bit quantization")
            else:
                print("  CPU mode will be slow. Consider 2B model.")
        else:
            print("Install with: pip install torch transformers pillow")
            print("Optional: pip install qwen-vl-utils")
        sys.exit(0)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Error: Please provide an image path")
            sys.exit(1)

        image_path = sys.argv[2]

        # Parse optional arguments
        variant = "7b"
        load_4bit = False
        load_8bit = False
        device = "auto"
        language = ""

        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--variant" and i + 1 < len(sys.argv):
                variant = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--4bit":
                load_4bit = True
                i += 1
            elif sys.argv[i] == "--8bit":
                load_8bit = True
                i += 1
            elif sys.argv[i] == "--cpu":
                device = "cpu"
                i += 1
            elif sys.argv[i] == "--lang" and i + 1 < len(sys.argv):
                language = sys.argv[i + 1]
                i += 2
            else:
                i += 1

        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        print(f"Testing Qwen2-VL on: {image_path}")
        print(f"Variant: {variant}")
        print(f"4-bit: {load_4bit}, 8-bit: {load_8bit}")
        print(f"Device: {device}")
        if language:
            print(f"Language: {language}")
        print("")

        provider = Qwen2VLProvider(
            model_variant=variant,
            load_in_4bit=load_4bit,
            load_in_8bit=load_8bit,
            device=device
        )

        if not provider.is_available():
            print("Error: Qwen2-VL not available")
            sys.exit(1)

        print("Loading model (this may take a while)...")
        result = provider.process_image_sync(image_path, language=language)

        if result.success:
            print("\nSUCCESS!")
            print(f"Processing time: {result.processing_time_ms:.0f}ms")
            print(f"Words detected: {result.word_count}")
            if result.tokens_used:
                print(f"Tokens generated: {result.tokens_used}")
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
