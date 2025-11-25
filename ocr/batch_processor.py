"""
Batch OCR Processor

Scans directories for images and runs OCR with multiple providers.
Stores results in SQLite database.
"""

import glob
import os
import sys
import time
import logging
from typing import List, Optional, Callable
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr.database import OCRDatabase


# Provider imports - done lazily to avoid loading all models at startup
PROVIDERS = {
    'windows_ocr': None,
    'tesseract': None,
    'doctr': None,
    'easyocr': None,
    'paddleocr': None
}


def get_provider(name: str):
    """Lazy-load and return a provider instance."""
    global PROVIDERS

    if PROVIDERS.get(name) is not None:
        return PROVIDERS[name]

    if name == 'windows_ocr':
        from ocr.providers.windows_ocr import WindowsOCRProvider, IS_WINDOWS, WINOCR_AVAILABLE
        if IS_WINDOWS and WINOCR_AVAILABLE:
            PROVIDERS[name] = WindowsOCRProvider()
        else:
            PROVIDERS[name] = False

    elif name == 'tesseract':
        from ocr.providers.tesseract_provider import TesseractProvider, TESSERACT_AVAILABLE
        if TESSERACT_AVAILABLE:
            PROVIDERS[name] = TesseractProvider(lang='eng')
        else:
            PROVIDERS[name] = False

    elif name == 'doctr':
        from ocr.providers.doctr_provider import DocTRProvider, DOCTR_AVAILABLE
        if DOCTR_AVAILABLE:
            PROVIDERS[name] = DocTRProvider()
        else:
            PROVIDERS[name] = False

    elif name == 'easyocr':
        from ocr.providers.easyocr_provider import EasyOCRProvider, EASYOCR_AVAILABLE
        if EASYOCR_AVAILABLE:
            PROVIDERS[name] = EasyOCRProvider(languages=['en'], gpu=False)
        else:
            PROVIDERS[name] = False

    elif name == 'paddleocr':
        from ocr.providers.paddleocr_provider import PaddleOCRProvider, PADDLEOCR_AVAILABLE
        if PADDLEOCR_AVAILABLE:
            PROVIDERS[name] = PaddleOCRProvider(lang='en', use_gpu=False)
        else:
            PROVIDERS[name] = False

    return PROVIDERS.get(name)


def find_images(
    base_dir: str,
    pattern: str = "*/Processed/*.jpg"
) -> List[str]:
    """
    Find all images matching pattern in base directory.

    Args:
        base_dir: Base directory to search
        pattern: Glob pattern for images

    Returns:
        List of image paths
    """
    images = []

    # Support multiple extensions
    extensions = ['jpg', 'jpeg', 'png']

    for ext in extensions:
        full_pattern = os.path.join(base_dir, pattern.replace('.jpg', f'.{ext}'))
        found = glob.glob(full_pattern)
        images.extend(found)

    # Sort by folder then filename
    images.sort()
    return images


def process_image_with_provider(
    image_path: str,
    provider_name: str,
    db: OCRDatabase,
    progress_callback: Callable = None
) -> bool:
    """
    Process a single image with a single provider.

    Args:
        image_path: Path to image
        provider_name: Name of provider to use
        db: Database instance
        progress_callback: Optional callback for progress updates

    Returns:
        True if successful
    """
    # Get or create image record
    image_id = db.add_image(image_path)

    # Check if already processed
    if db.has_ocr_result(image_id, provider_name):
        if progress_callback:
            progress_callback(f"  {provider_name}: Already processed (skipped)")
        return True

    # Get provider
    provider = get_provider(provider_name)
    if provider is False or provider is None:
        if progress_callback:
            progress_callback(f"  {provider_name}: Not available (skipped)")
        return False

    if not provider.is_available():
        if progress_callback:
            progress_callback(f"  {provider_name}: Not available (skipped)")
        return False

    # Run OCR
    try:
        start_time = time.perf_counter()
        result = provider.process_image_sync(image_path)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Store result
        db.add_ocr_result(
            image_id=image_id,
            provider=provider_name,
            text=result.text if result.success else "",
            confidence=result.confidence if result.success else 0.0,
            word_count=result.word_count if result.success else 0,
            char_count=result.char_count if result.success else 0,
            processing_time_ms=elapsed_ms,
            success=result.success,
            error=result.error if not result.success else None
        )

        if progress_callback:
            if result.success:
                progress_callback(f"  {provider_name}: {elapsed_ms:.0f}ms, {result.word_count} words")
            else:
                progress_callback(f"  {provider_name}: FAILED - {result.error}")

        return result.success

    except Exception as e:
        logging.error(f"Error processing {image_path} with {provider_name}: {e}")
        if progress_callback:
            progress_callback(f"  {provider_name}: ERROR - {e}")
        return False


def batch_process(
    image_dir: str,
    db_path: str = "ocr_results.db",
    providers: List[str] = None,
    pattern: str = "*/Processed/*.jpg",
    limit: int = None,
    skip_existing: bool = True,
    progress_callback: Callable = None
):
    """
    Process all images in directory with specified providers.

    Args:
        image_dir: Directory containing images
        db_path: Path to database file
        providers: List of provider names (default: all)
        pattern: Glob pattern for finding images
        limit: Maximum number of images to process
        skip_existing: Skip images that already have results
        progress_callback: Optional callback for progress updates
    """
    if providers is None:
        providers = ['windows_ocr', 'tesseract', 'doctr', 'easyocr', 'paddleocr']

    # Find all images
    print(f"Scanning for images in: {image_dir}")
    print(f"Pattern: {pattern}")

    images = find_images(image_dir, pattern)
    total_images = len(images)

    print(f"Found {total_images} images")

    if limit:
        images = images[:limit]
        print(f"Limited to first {limit} images")

    if not images:
        print("No images to process")
        return

    # Initialize database
    db = OCRDatabase(db_path)

    print(f"\nProviders to run: {', '.join(providers)}")
    print(f"Database: {db_path}")
    print("-" * 60)

    # Process each image
    processed = 0
    failed = 0

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {os.path.basename(image_path)}")

        if progress_callback:
            progress_callback(f"Processing {i}/{len(images)}: {image_path}")

        for provider_name in providers:
            success = process_image_with_provider(
                image_path=image_path,
                provider_name=provider_name,
                db=db,
                progress_callback=print
            )
            if success:
                processed += 1
            else:
                failed += 1

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)

    stats = db.get_stats()
    print(f"Total images in DB: {stats['image_count']}")
    print(f"Total OCR results: {stats['result_count']}")
    print("\nProvider statistics:")
    for provider, pstats in stats['providers'].items():
        print(f"  {provider}: {pstats['count']} results, avg {pstats['avg_time_ms']:.0f}ms")

    db.close()


def process_single_provider(
    image_dir: str,
    provider_name: str,
    db_path: str = "ocr_results.db",
    pattern: str = "*/Processed/*.jpg",
    limit: int = None
):
    """
    Process all images with a single provider (for parallel processing).

    Args:
        image_dir: Directory containing images
        provider_name: Provider to use
        db_path: Path to database file
        pattern: Glob pattern for finding images
        limit: Maximum number of images to process
    """
    print(f"Processing with {provider_name}")
    print("=" * 60)

    # Find images
    images = find_images(image_dir, pattern)
    if limit:
        images = images[:limit]

    print(f"Found {len(images)} images")

    # Initialize database and provider
    db = OCRDatabase(db_path)

    # Pre-initialize provider (load models once)
    print(f"Initializing {provider_name}...")
    provider = get_provider(provider_name)
    if provider is False or provider is None or not provider.is_available():
        print(f"Provider {provider_name} not available!")
        return

    print(f"Provider ready. Processing images...")
    print("-" * 60)

    processed = 0
    skipped = 0
    failed = 0

    for i, image_path in enumerate(images, 1):
        image_id = db.add_image(image_path)

        # Check if already processed
        if db.has_ocr_result(image_id, provider_name):
            skipped += 1
            continue

        # Process
        try:
            start_time = time.perf_counter()
            result = provider.process_image_sync(image_path)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            db.add_ocr_result(
                image_id=image_id,
                provider=provider_name,
                text=result.text if result.success else "",
                confidence=result.confidence if result.success else 0.0,
                word_count=result.word_count if result.success else 0,
                char_count=result.char_count if result.success else 0,
                processing_time_ms=elapsed_ms,
                success=result.success,
                error=result.error if not result.success else None
            )

            if result.success:
                processed += 1
                print(f"[{i}/{len(images)}] {os.path.basename(image_path)}: {elapsed_ms:.0f}ms, {result.word_count} words")
            else:
                failed += 1
                print(f"[{i}/{len(images)}] {os.path.basename(image_path)}: FAILED - {result.error}")

        except Exception as e:
            failed += 1
            print(f"[{i}/{len(images)}] {os.path.basename(image_path)}: ERROR - {e}")

    print("\n" + "=" * 60)
    print(f"{provider_name} COMPLETE")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {failed}")

    db.close()


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch OCR Processor")
    parser.add_argument("image_dir", nargs="?", default="Image Files",
                        help="Directory containing images")
    parser.add_argument("--db", default="ocr_results.db",
                        help="Database file path")
    parser.add_argument("--providers", "-p", nargs="+",
                        choices=['windows_ocr', 'tesseract', 'doctr', 'easyocr', 'paddleocr'],
                        help="Providers to use (default: all)")
    parser.add_argument("--single", "-s",
                        choices=['windows_ocr', 'tesseract', 'doctr', 'easyocr', 'paddleocr'],
                        help="Run single provider only")
    parser.add_argument("--pattern", default="*/Processed/*.jpg",
                        help="Glob pattern for images")
    parser.add_argument("--limit", "-n", type=int,
                        help="Limit number of images to process")
    parser.add_argument("--stats", action="store_true",
                        help="Show database statistics and exit")

    args = parser.parse_args()

    if args.stats:
        # Just show stats
        if os.path.exists(args.db):
            db = OCRDatabase(args.db)
            stats = db.get_stats()
            print(f"Database: {args.db}")
            print(f"Images: {stats['image_count']}")
            print(f"OCR Results: {stats['result_count']}")
            print("\nProvider stats:")
            for provider, pstats in stats['providers'].items():
                print(f"  {provider}: {pstats['count']} results, avg {pstats['avg_time_ms']:.0f}ms")
            db.close()
        else:
            print(f"Database not found: {args.db}")
        sys.exit(0)

    if args.single:
        # Single provider mode
        process_single_provider(
            image_dir=args.image_dir,
            provider_name=args.single,
            db_path=args.db,
            pattern=args.pattern,
            limit=args.limit
        )
    else:
        # Multi-provider mode
        batch_process(
            image_dir=args.image_dir,
            db_path=args.db,
            providers=args.providers,
            pattern=args.pattern,
            limit=args.limit
        )
