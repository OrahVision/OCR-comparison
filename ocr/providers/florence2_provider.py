"""
Florence-2 Provider

Microsoft's lightweight vision-language model with OCR capabilities.
https://huggingface.co/microsoft/Florence-2-large

License: MIT - Commercial use allowed

Requirements:
- Lightweight: 0.23B (base) to 0.77B (large) parameters
- Works on consumer GPUs and CPU
"""

import logging
import os
from typing import List, Optional, Dict, Any

from ..base import OCRProvider, OCRResult, BoundingBox, ProviderType, Timer
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
    from transformers import AutoModelForCausalLM, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.info("transformers not installed")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

FLORENCE_AVAILABLE = TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE and PIL_AVAILABLE


@register_provider
class Florence2Provider(OCRProvider):
    """
    Florence-2 Provider

    Features:
    - Lightweight VLM (0.23B-0.77B parameters)
    - Multiple task modes (OCR, caption, detection, etc.)
    - Works on consumer hardware
    - Good accuracy for document OCR
    """

    PROVIDER_NAME = "florence2"
    PROVIDER_TYPE = ProviderType.LOCAL
    COMMERCIAL_USE = True
    LICENSE = "MIT"

    # Model variants
    MODEL_VARIANTS = {
        "base": "microsoft/Florence-2-base",
        "large": "microsoft/Florence-2-large",
        "base-ft": "microsoft/Florence-2-base-ft",      # Fine-tuned
        "large-ft": "microsoft/Florence-2-large-ft",    # Fine-tuned
    }

    # Task prompts for different OCR modes
    TASK_PROMPTS = {
        "ocr": "<OCR>",                      # Basic OCR
        "ocr_with_region": "<OCR_WITH_REGION>",  # OCR with bounding boxes
        "caption": "<CAPTION>",              # Image caption
        "detailed_caption": "<DETAILED_CAPTION>",
        "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    }

    SUPPORTED_LANGUAGES = [
        "en",    # English (primary)
        # Florence-2 is primarily English but handles Latin scripts
    ]

    def __init__(
        self,
        model_variant: str = "large",
        device: str = "auto",
        torch_dtype: str = "auto",
        task: str = "ocr",
        trust_remote_code: bool = True,
        **config
    ):
        """
        Initialize Florence-2 provider.

        Args:
            model_variant: Model size ('base', 'large', 'base-ft', 'large-ft')
            device: Device to use ('auto', 'cuda', 'cpu')
            torch_dtype: Data type ('auto', 'float16', 'bfloat16', 'float32')
            task: OCR task type ('ocr', 'ocr_with_region', 'caption')
            trust_remote_code: Trust remote code from HuggingFace
        """
        self.model_variant = model_variant
        self.device_config = device
        self.torch_dtype_config = torch_dtype
        self.task = task
        self.trust_remote_code = trust_remote_code

        self.model = None
        self.processor = None
        self.device = None
        self._initialized = False
        self._init_error = None

        if model_variant not in self.MODEL_VARIANTS:
            self._init_error = f"Invalid model variant: {model_variant}"

        if task not in self.TASK_PROMPTS:
            self._init_error = f"Invalid task: {task}. Choose from: {list(self.TASK_PROMPTS.keys())}"

    def _get_model_name(self) -> str:
        """Get the HuggingFace model name for current variant."""
        return self.MODEL_VARIANTS.get(self.model_variant, self.MODEL_VARIANTS["large"])

    def _initialize_model(self):
        """Initialize the model (lazy loading)."""
        if self._initialized or self._init_error:
            return

        if not FLORENCE_AVAILABLE:
            self._init_error = "Florence-2 dependencies not available"
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            model_name = self._get_model_name()

            # Determine device
            if self.device_config == "auto":
                self.device = "cuda" if CUDA_AVAILABLE else "cpu"
            else:
                self.device = self.device_config

            # Determine dtype
            if self.torch_dtype_config == "auto":
                if self.device == "cuda":
                    torch_dtype = torch.float16
                else:
                    torch_dtype = torch.float32
            elif self.torch_dtype_config == "float16":
                torch_dtype = torch.float16
            elif self.torch_dtype_config == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

            logging.info(f"Loading Florence-2 model: {model_name}")

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=self.trust_remote_code
            )

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=self.trust_remote_code
            ).to(self.device)

            self.model.eval()
            self._initialized = True

            logging.info(
                f"Florence-2 initialized: {model_name}, device: {self.device}"
            )

        except Exception as e:
            self._init_error = str(e)
            logging.error(f"Failed to initialize Florence-2: {e}")

    def is_available(self) -> bool:
        """Check if Florence-2 is available."""
        if not FLORENCE_AVAILABLE:
            return False
        if self._init_error:
            return False
        return True

    async def process_image(self, image_path: str, **options) -> OCRResult:
        """
        Process an image with Florence-2.

        Args:
            image_path: Path to the image file
            **options:
                task: Override default task
                max_new_tokens: Maximum tokens to generate (default: 1024)

        Returns:
            OCRResult with extracted text
        """
        if not os.path.exists(image_path):
            return self._create_error_result(f"Image not found: {image_path}")

        if not self.is_available():
            error = self._init_error or "Florence-2 not available"
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

                # Get task
                task = options.get("task", self.task)
                task_prompt = self.TASK_PROMPTS.get(task, self.TASK_PROMPTS["ocr"])
                max_new_tokens = options.get("max_new_tokens", 1024)

                # Prepare inputs
                inputs = self.processor(
                    text=task_prompt,
                    images=image,
                    return_tensors="pt"
                ).to(self.device)

                # Generate
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=max_new_tokens,
                        num_beams=3,
                        do_sample=False,
                    )

                # Decode
                generated_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=False
                )[0]

                # Post-process based on task
                result_text, bounding_boxes = self._parse_output(
                    generated_text, task, image.size
                )

                return self._create_success_result(
                    text=result_text,
                    processing_time_ms=timer.elapsed_ms,
                    confidence=1.0,
                    bounding_boxes=bounding_boxes,
                    tokens_used=generated_ids.shape[1],
                )

            except Exception as e:
                logging.error(f"Florence-2 processing error: {e}")
                return self._create_error_result(str(e), timer.elapsed_ms)

    def _parse_output(self, raw_output: str, task: str, image_size: tuple):
        """Parse Florence-2 output based on task type."""
        bounding_boxes = None

        # Post-process the output
        try:
            result = self.processor.post_process_generation(
                raw_output,
                task=self.TASK_PROMPTS[task],
                image_size=image_size
            )

            if task == "ocr":
                # Simple OCR returns text
                text = result.get(self.TASK_PROMPTS[task], raw_output)

            elif task == "ocr_with_region":
                # OCR with regions returns structured data
                ocr_data = result.get(self.TASK_PROMPTS[task], {})

                if isinstance(ocr_data, dict):
                    labels = ocr_data.get("labels", [])
                    quad_boxes = ocr_data.get("quad_boxes", [])

                    text_parts = []
                    bounding_boxes = []

                    for i, label in enumerate(labels):
                        text_parts.append(label)

                        if i < len(quad_boxes):
                            # quad_boxes is [x1,y1,x2,y2,x3,y3,x4,y4]
                            box = quad_boxes[i]
                            if len(box) >= 8:
                                xs = [box[j] for j in range(0, 8, 2)]
                                ys = [box[j] for j in range(1, 8, 2)]
                                x = int(min(xs))
                                y = int(min(ys))
                                width = int(max(xs) - x)
                                height = int(max(ys) - y)

                                bounding_boxes.append(BoundingBox(
                                    x=x,
                                    y=y,
                                    width=width,
                                    height=height,
                                    text=label,
                                    confidence=1.0
                                ))

                    text = " ".join(text_parts)
                else:
                    text = str(ocr_data)

            else:
                # Caption tasks return descriptive text
                text = result.get(self.TASK_PROMPTS[task], raw_output)

        except Exception as e:
            logging.debug(f"Output parsing fallback: {e}")
            # Fallback: clean up raw output
            text = raw_output
            # Remove special tokens
            for token in ["<s>", "</s>", "<pad>", self.TASK_PROMPTS.get(task, "")]:
                text = text.replace(token, "")
            text = text.strip()

        return text, bounding_boxes

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

        logging.info("Florence-2 model unloaded")


# CLI interface for testing
if __name__ == "__main__":
    import sys

    def print_help():
        print("Florence-2 Provider")
        print("")
        print("Usage:")
        print("  python -m ocr.providers.florence2_provider test <image>  - Test OCR")
        print("  python -m ocr.providers.florence2_provider check         - Check availability")
        print("")
        print("Options:")
        print("  --variant <name>  Model variant (base/large/base-ft/large-ft)")
        print("  --task <name>     Task type (ocr/ocr_with_region/caption)")
        print("  --cpu             Force CPU mode")
        print("")
        print("Examples:")
        print("  python -m ocr.providers.florence2_provider test doc.jpg")
        print("  python -m ocr.providers.florence2_provider test doc.jpg --task ocr_with_region")

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "check":
        print(f"PyTorch installed: {TORCH_AVAILABLE}")
        print(f"Transformers installed: {TRANSFORMERS_AVAILABLE}")
        print(f"PIL installed: {PIL_AVAILABLE}")
        print(f"CUDA available: {CUDA_AVAILABLE}")
        print("")

        if FLORENCE_AVAILABLE:
            print("Florence-2 dependencies OK")
        else:
            print("Install with: pip install torch transformers pillow")
        sys.exit(0)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Error: Please provide an image path")
            sys.exit(1)

        image_path = sys.argv[2]

        # Parse optional arguments
        variant = "large"
        task = "ocr"
        device = "auto"

        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--variant" and i + 1 < len(sys.argv):
                variant = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--task" and i + 1 < len(sys.argv):
                task = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--cpu":
                device = "cpu"
                i += 1
            else:
                i += 1

        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        print(f"Testing Florence-2 on: {image_path}")
        print(f"Variant: {variant}")
        print(f"Task: {task}")
        print(f"Device: {device}")
        print("")

        provider = Florence2Provider(
            model_variant=variant,
            task=task,
            device=device
        )

        if not provider.is_available():
            print("Error: Florence-2 not available")
            sys.exit(1)

        print("Loading model...")
        result = provider.process_image_sync(image_path)

        if result.success:
            print("\nSUCCESS!")
            print(f"Processing time: {result.processing_time_ms:.0f}ms")
            print(f"Words detected: {result.word_count}")
            if result.bounding_boxes:
                print(f"Regions detected: {len(result.bounding_boxes)}")
            print("")
            print("Extracted text:")
            print("-" * 50)
            print(result.text[:1500] + ("..." if len(result.text) > 1500 else ""))
            sys.exit(0)
        else:
            print(f"FAILED: {result.error}")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        print_help()
        sys.exit(1)
