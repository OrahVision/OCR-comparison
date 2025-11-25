"""
OCR Latency Test Script

Tests each provider individually with accurate timing.
Timer starts right when image is sent to OCR, ends when OCR completes.
"""

import time
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGE_PATH = r"Image Files\2025_03_23_17_48_30\Processed\image00002.jpg"


def test_easyocr(image_path: str) -> dict:
    """Test EasyOCR latency."""
    print("=" * 60)
    print("Testing EasyOCR")
    print("=" * 60)

    from ocr.providers.easyocr_provider import EasyOCRProvider, EASYOCR_AVAILABLE

    if not EASYOCR_AVAILABLE:
        print("EasyOCR not available")
        return {"provider": "easyocr", "available": False}

    # Initialize (not timed - one-time setup)
    print("Initializing EasyOCR reader...")
    init_start = time.perf_counter()
    provider = EasyOCRProvider(languages=["en"], gpu=False)
    init_time = (time.perf_counter() - init_start) * 1000
    print(f"Initialization time: {init_time:.0f}ms")

    if not provider.is_available():
        print("Provider not available after init")
        return {"provider": "easyocr", "available": False}

    # OCR timing (this is what we care about)
    print(f"\nProcessing: {image_path}")
    print("Starting OCR...")

    ocr_start = time.perf_counter()
    # Direct call to reader.readtext for accurate timing
    results = provider.reader.readtext(image_path)
    ocr_end = time.perf_counter()

    ocr_time_ms = (ocr_end - ocr_start) * 1000

    # Process results
    text_parts = [item[1] for item in results]
    text = "\n".join(text_parts)
    word_count = len(text.split())
    avg_conf = sum(item[2] for item in results) / len(results) if results else 0

    print(f"\nOCR Time: {ocr_time_ms:.0f}ms")
    print(f"Words: {word_count}")
    print(f"Confidence: {avg_conf:.1%}")
    print(f"Preview: {text[:200]}...")

    return {
        "provider": "easyocr",
        "available": True,
        "init_time_ms": init_time,
        "ocr_time_ms": ocr_time_ms,
        "word_count": word_count,
        "confidence": avg_conf
    }


def test_doctr(image_path: str) -> dict:
    """Test docTR latency."""
    print("\n" + "=" * 60)
    print("Testing docTR")
    print("=" * 60)

    from ocr.providers.doctr_provider import DocTRProvider, DOCTR_AVAILABLE

    if not DOCTR_AVAILABLE:
        print("docTR not available")
        return {"provider": "doctr", "available": False}

    from doctr.io import DocumentFile

    # Initialize (not timed)
    print("Initializing docTR predictor...")
    init_start = time.perf_counter()
    provider = DocTRProvider()
    init_time = (time.perf_counter() - init_start) * 1000
    print(f"Initialization time: {init_time:.0f}ms")

    if not provider.is_available():
        print("Provider not available after init")
        return {"provider": "doctr", "available": False}

    # OCR timing
    print(f"\nProcessing: {image_path}")
    print("Starting OCR...")

    # Load document (part of OCR pipeline)
    ocr_start = time.perf_counter()
    doc = DocumentFile.from_images(image_path)
    result = provider.predictor(doc)
    ocr_end = time.perf_counter()

    ocr_time_ms = (ocr_end - ocr_start) * 1000

    # Process results
    text_parts = []
    confidences = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join(w.value for w in line.words)
                text_parts.append(line_text)
                confidences.extend(w.confidence for w in line.words)

    text = "\n".join(text_parts)
    word_count = len(text.split())
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    print(f"\nOCR Time: {ocr_time_ms:.0f}ms")
    print(f"Words: {word_count}")
    print(f"Confidence: {avg_conf:.1%}")
    print(f"Preview: {text[:200]}...")

    return {
        "provider": "doctr",
        "available": True,
        "init_time_ms": init_time,
        "ocr_time_ms": ocr_time_ms,
        "word_count": word_count,
        "confidence": avg_conf
    }


def test_windows_ocr(image_path: str) -> dict:
    """Test Windows OCR latency."""
    print("\n" + "=" * 60)
    print("Testing Windows OCR")
    print("=" * 60)

    from ocr.providers.windows_ocr import WindowsOCRProvider, IS_WINDOWS, WINOCR_AVAILABLE

    if not IS_WINDOWS:
        print("Windows OCR only available on Windows")
        return {"provider": "windows_ocr", "available": False}

    if not WINOCR_AVAILABLE:
        print("winocr package not available")
        return {"provider": "windows_ocr", "available": False}

    import winocr
    from PIL import Image
    import asyncio

    # Initialize (minimal for Windows OCR)
    print("Initializing Windows OCR...")
    init_start = time.perf_counter()
    provider = WindowsOCRProvider()
    init_time = (time.perf_counter() - init_start) * 1000
    print(f"Initialization time: {init_time:.0f}ms")

    # OCR timing
    print(f"\nProcessing: {image_path}")
    print("Starting OCR...")

    async def do_ocr():
        image = Image.open(image_path)
        return await winocr.recognize_pil(image)

    ocr_start = time.perf_counter()
    result = asyncio.run(do_ocr())
    ocr_end = time.perf_counter()

    ocr_time_ms = (ocr_end - ocr_start) * 1000

    # Process results
    text_parts = []
    if hasattr(result, 'lines'):
        for line in result.lines:
            text_parts.append(line.text)

    text = "\n".join(text_parts)
    word_count = len(text.split())

    print(f"\nOCR Time: {ocr_time_ms:.0f}ms")
    print(f"Words: {word_count}")
    print(f"Preview: {text[:200]}...")

    return {
        "provider": "windows_ocr",
        "available": True,
        "init_time_ms": init_time,
        "ocr_time_ms": ocr_time_ms,
        "word_count": word_count,
        "confidence": 1.0  # Windows OCR doesn't provide confidence
    }


def test_paddleocr(image_path: str) -> dict:
    """Test PaddleOCR latency."""
    print("\n" + "=" * 60)
    print("Testing PaddleOCR")
    print("=" * 60)

    from ocr.providers.paddleocr_provider import PaddleOCRProvider, PADDLEOCR_AVAILABLE

    if not PADDLEOCR_AVAILABLE:
        print("PaddleOCR not available")
        return {"provider": "paddleocr", "available": False}

    from paddleocr import PaddleOCR

    # Initialize (not timed - one-time model loading)
    print("Initializing PaddleOCR (this loads models)...")
    init_start = time.perf_counter()
    ocr = PaddleOCR(lang='en', use_textline_orientation=True)
    init_time = (time.perf_counter() - init_start) * 1000
    print(f"Initialization time: {init_time:.0f}ms")

    # OCR timing
    print(f"\nProcessing: {image_path}")
    print("Starting OCR...")

    ocr_start = time.perf_counter()
    results = ocr.predict(image_path)
    ocr_end = time.perf_counter()

    ocr_time_ms = (ocr_end - ocr_start) * 1000

    # Process results
    text_parts = []
    confidences = []

    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'get'):
            rec_texts = result.get('rec_texts', [])
            rec_scores = result.get('rec_scores', [])
            text_parts = list(rec_texts)
            confidences = [float(s) for s in rec_scores]

    text = "\n".join(text_parts)
    word_count = len(text.split())
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    print(f"\nOCR Time: {ocr_time_ms:.0f}ms")
    print(f"Words: {word_count}")
    print(f"Confidence: {avg_conf:.1%}")
    print(f"Preview: {text[:200]}...")

    return {
        "provider": "paddleocr",
        "available": True,
        "init_time_ms": init_time,
        "ocr_time_ms": ocr_time_ms,
        "word_count": word_count,
        "confidence": avg_conf
    }


def print_summary(results: list):
    """Print a summary table of results."""
    print("\n" + "=" * 60)
    print("LATENCY SUMMARY")
    print("=" * 60)
    print(f"{'Provider':<15} {'Init(ms)':<12} {'OCR(ms)':<12} {'Words':<8} {'Conf':<8}")
    print("-" * 60)

    available_results = [r for r in results if r.get('available')]
    available_results.sort(key=lambda x: x.get('ocr_time_ms', float('inf')))

    for r in available_results:
        init_ms = f"{r.get('init_time_ms', 0):.0f}"
        ocr_ms = f"{r.get('ocr_time_ms', 0):.0f}"
        words = str(r.get('word_count', 0))
        conf = f"{r.get('confidence', 0):.1%}"
        print(f"{r['provider']:<15} {init_ms:<12} {ocr_ms:<12} {words:<8} {conf:<8}")

    print("-" * 60)

    if available_results:
        fastest = available_results[0]
        print(f"\nFastest OCR: {fastest['provider']} ({fastest['ocr_time_ms']:.0f}ms)")


if __name__ == "__main__":
    # Check image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found: {IMAGE_PATH}")
        sys.exit(1)

    print(f"Testing OCR latency on: {IMAGE_PATH}")
    print(f"Image size: {os.path.getsize(IMAGE_PATH) / 1024:.1f} KB")

    results = []

    # Test each provider one at a time
    if len(sys.argv) > 1:
        provider = sys.argv[1].lower()
        if provider == "easyocr":
            results.append(test_easyocr(IMAGE_PATH))
        elif provider == "doctr":
            results.append(test_doctr(IMAGE_PATH))
        elif provider == "windows_ocr":
            results.append(test_windows_ocr(IMAGE_PATH))
        elif provider == "paddleocr":
            results.append(test_paddleocr(IMAGE_PATH))
        else:
            print(f"Unknown provider: {provider}")
            print("Available: easyocr, doctr, windows_ocr, paddleocr")
            sys.exit(1)
    else:
        # Test all
        results.append(test_easyocr(IMAGE_PATH))
        results.append(test_doctr(IMAGE_PATH))
        results.append(test_windows_ocr(IMAGE_PATH))
        results.append(test_paddleocr(IMAGE_PATH))

    print_summary(results)
