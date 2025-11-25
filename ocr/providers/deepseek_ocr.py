"""
DeepSeek-OCR Provider

Vision-language model specialized for OCR with high accuracy and compression.
https://github.com/deepseek-ai/DeepSeek-OCR

License: MIT - Commercial use allowed

Requirements:
- ~16GB VRAM for full precision (BF16)
- Can run 4-bit quantized on ~8GB VRAM
- torch 2.6.0+, transformers 4.46+
"""

import logging
import os
from typing import List, Optional, Dict, Any

from ..base import OCRProvider, OCRResult, ProviderType, Timer
from ..registry import register_provider

# Check dependencies
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
DEEPSEEK_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        GPU_NAME = None
        GPU_MEMORY = 0
except ImportError:
    CUDA_AVAILABLE = False
    GPU_NAME = None
    GPU_MEMORY = 0
    logging.info("PyTorch not installed")

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.info("transformers not installed. Install with: pip install transformers>=4.46")

# PIL for image loading
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

DEEPSEEK_AVAILABLE = TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE and PIL_AVAILABLE


@register_provider
class DeepSeekOCRProvider(OCRProvider):
    """
    DeepSeek-OCR Provider

    Features:
    - 97% OCR accuracy at 10x compression
    - ~100 language support
    - Outputs clean, structured Markdown
    - Multiple resolution variants
    - Built on DeepSeek-VL2 vision-language model
    """

    PROVIDER_NAME = "deepseek_ocr"
    PROVIDER_TYPE = ProviderType.LOCAL
    COMMERCIAL_USE = True
    LICENSE = "MIT"

    # Model variants by resolution
    MODEL_VARIANTS = {
        "tiny": "deepseek-ai/DeepSeek-OCR-Tiny",     # 512x512, 64 tokens
        "small": "deepseek-ai/DeepSeek-OCR-Small",   # 640x640, 100 tokens
        "base": "deepseek-ai/DeepSeek-OCR-Base",     # 1024x1024, 256 tokens
        "large": "deepseek-ai/DeepSeek-OCR-Large",   # 1280x1280, 400 tokens
    }

    # Extensive language support
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
        # ~100 languages total
    ]

    def __init__(
        self,
        model_variant: str = "base",
        device: str = "auto",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        trust_remote_code: bool = True,
        **config
    ):
        """
        Initialize DeepSeek-OCR provider.

        Args:
            model_variant: Model size ('tiny', 'small', 'base', 'large')
            device: Device to use ('auto', 'cuda', 'cpu')
            torch_dtype: Data type ('auto', 'float16', 'bfloat16', 'float32')
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
            load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
            trust_remote_code: Trust remote code from HuggingFace
        """
        self.model_variant = model_variant
        self.device_config = device
        self.torch_dtype_config = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.trust_remote_code = trust_remote_code

        self.model = None
        self.processor = None
        self.device = None
        self._initialized = False
        self._init_error = None

        if model_variant not in self.MODEL_VARIANTS:
            self._init_error = f"Invalid model variant: {model_variant}. Choose from: {list(self.MODEL_VARIANTS.keys())}"

    def _get_model_name(self) -> str:
        """Get the HuggingFace model name for current variant."""
        return self.MODEL_VARIANTS.get(self.model_variant, self.MODEL_VARIANTS["base"])

    def _initialize_model(self):
        """Initialize the model (lazy loading)."""
        if self._initialized or self._init_error:
            return

        if not DEEPSEEK_AVAILABLE:
            self._init_error = "DeepSeek-OCR dependencies not available"
            return

        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            model_name = self._get_model_name()

            # Determine device
            if self.device_config == "auto":
                self.device = "cuda" if CUDA_AVAILABLE else "cpu"
            else:
                self.device = self.device_config

            # Determine dtype
            if self.torch_dtype_config == "auto":
                if CUDA_AVAILABLE:
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
                "trust_remote_code": self.trust_remote_code,
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

            # Set device_map for auto placement
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"

            logging.info(f"Loading DeepSeek-OCR model: {model_name}")

            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=self.trust_remote_code
            )

            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                **model_kwargs
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()
            self._initialized = True

            logging.info(
                f"DeepSeek-OCR initialized: {model_name}, device: {self.device}, "
                f"dtype: {torch_dtype}"
            )

        except Exception as e:
            self._init_error = str(e)
            logging.error(f"Failed to initialize DeepSeek-OCR: {e}")

    def is_available(self) -> bool:
        """Check if DeepSeek-OCR is available."""
        if not DEEPSEEK_AVAILABLE:
            return False
        if self._init_error:
            return False
        return True

    async def process_image(self, image_path: str, **options) -> OCRResult:
        """
        Process an image with DeepSeek-OCR.

        Args:
            image_path: Path to the image file
            **options:
                max_new_tokens: Maximum tokens to generate (default: 4096)
                do_sample: Use sampling (default: False)
                prompt: Custom prompt (default: OCR prompt)

        Returns:
            OCRResult with extracted text (Markdown format)
        """
        if not os.path.exists(image_path):
            return self._create_error_result(f"Image not found: {image_path}")

        if not self.is_available():
            error = self._init_error or "DeepSeek-OCR not available"
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
                max_new_tokens = options.get("max_new_tokens", 4096)
                do_sample = options.get("do_sample", False)
                prompt = options.get("prompt", "OCR the text in this image.")

                # Prepare inputs
                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt"
                )

                # Move to device
                if self.device == "cuda":
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                    )

                # Decode output
                generated_text = self.processor.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

                # Remove the prompt from output if present
                if prompt in generated_text:
                    generated_text = generated_text.split(prompt, 1)[-1].strip()

                return self._create_success_result(
                    text=generated_text,
                    processing_time_ms=timer.elapsed_ms,
                    confidence=1.0,  # Model doesn't provide confidence
                    tokens_used=outputs.shape[1],
                )

            except Exception as e:
                logging.error(f"DeepSeek-OCR processing error: {e}")
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

        logging.info("DeepSeek-OCR model unloaded")


# Convenience function for checking GPU requirements
def check_gpu_requirements():
    """Check if system meets GPU requirements for DeepSeek-OCR."""
    print("DeepSeek-OCR GPU Requirements Check")
    print("=" * 50)
    print(f"PyTorch installed: {TORCH_AVAILABLE}")
    print(f"Transformers installed: {TRANSFORMERS_AVAILABLE}")
    print(f"PIL installed: {PIL_AVAILABLE}")
    print("")

    if TORCH_AVAILABLE:
        print(f"CUDA available: {CUDA_AVAILABLE}")
        if CUDA_AVAILABLE:
            print(f"GPU: {GPU_NAME}")
            print(f"GPU Memory: {GPU_MEMORY:.1f} GB")
            print("")

            if GPU_MEMORY >= 16:
                print("OK: Full precision (BF16) should work")
                print("   Recommended: model_variant='large'")
            elif GPU_MEMORY >= 10:
                print("OK: May need 8-bit quantization")
                print("   Recommended: load_in_8bit=True")
            elif GPU_MEMORY >= 6:
                print("OK: Needs 4-bit quantization")
                print("   Recommended: load_in_4bit=True, model_variant='small'")
            else:
                print("WARNING: GPU memory may be insufficient")
                print("   Try: load_in_4bit=True, model_variant='tiny'")
        else:
            print("No GPU available - will use CPU (slow)")


# CLI interface for testing
if __name__ == "__main__":
    import sys

    def print_help():
        print("DeepSeek-OCR Provider")
        print("")
        print("Usage:")
        print("  python -m ocr.providers.deepseek_ocr test <image>  - Test OCR")
        print("  python -m ocr.providers.deepseek_ocr check         - Check GPU requirements")
        print("")
        print("Options:")
        print("  --variant <name>  Model variant (tiny/small/base/large)")
        print("  --4bit            Use 4-bit quantization")
        print("  --8bit            Use 8-bit quantization")
        print("  --cpu             Force CPU mode")
        print("")
        print("Examples:")
        print("  python -m ocr.providers.deepseek_ocr check")
        print("  python -m ocr.providers.deepseek_ocr test doc.jpg --variant small --4bit")

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "check":
        check_gpu_requirements()
        sys.exit(0)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Error: Please provide an image path")
            sys.exit(1)

        image_path = sys.argv[2]

        # Parse optional arguments
        variant = "base"
        load_4bit = False
        load_8bit = False
        device = "auto"

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
            else:
                i += 1

        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        print(f"Testing DeepSeek-OCR on: {image_path}")
        print(f"Variant: {variant}")
        print(f"4-bit: {load_4bit}, 8-bit: {load_8bit}")
        print(f"Device: {device}")
        print("")

        provider = DeepSeekOCRProvider(
            model_variant=variant,
            load_in_4bit=load_4bit,
            load_in_8bit=load_8bit,
            device=device
        )

        if not provider.is_available():
            print("Error: DeepSeek-OCR not available")
            print("Install with: pip install torch transformers pillow")
            sys.exit(1)

        print("Loading model (this may take a while on first run)...")
        result = provider.process_image_sync(image_path)

        if result.success:
            print("\nSUCCESS!")
            print(f"Processing time: {result.processing_time_ms:.0f}ms")
            print(f"Words detected: {result.word_count}")
            if result.tokens_used:
                print(f"Tokens generated: {result.tokens_used}")
            print("")
            print("Extracted text (Markdown):")
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
