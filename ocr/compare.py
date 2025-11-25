"""
OCR Provider Comparison Tool

Compares multiple OCR providers on the same image(s) with accuracy metrics.
Calculates CER (Character Error Rate) and WER (Word Error Rate) against ground truth.
"""

import asyncio
import os
import sys
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .registry import ProviderRegistry
from .base import OCRResult


@dataclass
class AccuracyMetrics:
    """Accuracy metrics comparing OCR output to ground truth."""
    cer: float = 0.0  # Character Error Rate (lower is better)
    wer: float = 0.0  # Word Error Rate (lower is better)
    accuracy: float = 0.0  # 1 - CER (higher is better)
    ground_truth_chars: int = 0
    ground_truth_words: int = 0
    ocr_chars: int = 0
    ocr_words: int = 0


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_cer(ocr_text: str, ground_truth: str) -> float:
    """
    Calculate Character Error Rate.

    CER = (insertions + deletions + substitutions) / total_reference_chars
    Lower is better (0.0 = perfect match)
    """
    if not ground_truth:
        return 1.0 if ocr_text else 0.0

    distance = levenshtein_distance(ocr_text, ground_truth)
    return distance / len(ground_truth)


def calculate_wer(ocr_text: str, ground_truth: str) -> float:
    """
    Calculate Word Error Rate.

    WER = (word insertions + deletions + substitutions) / total_reference_words
    Lower is better (0.0 = perfect match)
    """
    ocr_words = ocr_text.split()
    gt_words = ground_truth.split()

    if not gt_words:
        return 1.0 if ocr_words else 0.0

    # Use word-level edit distance
    distance = levenshtein_distance_words(ocr_words, gt_words)
    return distance / len(gt_words)


def levenshtein_distance_words(words1: List[str], words2: List[str]) -> int:
    """Calculate Levenshtein distance between two word lists."""
    if len(words1) < len(words2):
        return levenshtein_distance_words(words2, words1)

    if len(words2) == 0:
        return len(words1)

    previous_row = range(len(words2) + 1)
    for i, w1 in enumerate(words1):
        current_row = [i + 1]
        for j, w2 in enumerate(words2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (w1.lower() != w2.lower())
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, single spaces, strip)."""
    import re
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip and lowercase
    return text.strip().lower()


def calculate_metrics(ocr_text: str, ground_truth: str, normalize: bool = True) -> AccuracyMetrics:
    """
    Calculate accuracy metrics comparing OCR output to ground truth.

    Args:
        ocr_text: Text from OCR provider
        ground_truth: Reference text
        normalize: Whether to normalize text before comparison

    Returns:
        AccuracyMetrics with CER, WER, and accuracy
    """
    if normalize:
        ocr_text = normalize_text(ocr_text)
        ground_truth = normalize_text(ground_truth)

    cer = calculate_cer(ocr_text, ground_truth)
    wer = calculate_wer(ocr_text, ground_truth)

    return AccuracyMetrics(
        cer=cer,
        wer=wer,
        accuracy=max(0.0, 1.0 - cer),  # Clamp to 0
        ground_truth_chars=len(ground_truth),
        ground_truth_words=len(ground_truth.split()),
        ocr_chars=len(ocr_text),
        ocr_words=len(ocr_text.split())
    )


@dataclass
class ComparisonResult:
    """Result of comparing multiple providers on an image."""
    image_path: str
    ground_truth: Optional[str] = None
    results: Dict[str, OCRResult] = field(default_factory=dict)
    metrics: Dict[str, AccuracyMetrics] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "image_path": self.image_path,
            "ground_truth_available": self.ground_truth is not None,
            "providers": {
                name: {
                    **result.to_dict(),
                    "metrics": {
                        "cer": self.metrics.get(name, AccuracyMetrics()).cer,
                        "wer": self.metrics.get(name, AccuracyMetrics()).wer,
                        "accuracy": self.metrics.get(name, AccuracyMetrics()).accuracy
                    } if name in self.metrics else None
                }
                for name, result in self.results.items()
            }
        }


async def compare_providers(
    image_path: str,
    providers: Optional[List[str]] = None,
    ground_truth: Optional[str] = None,
    skip_unavailable: bool = True,
    verbose: bool = True
) -> ComparisonResult:
    """
    Compare multiple OCR providers on an image.

    Args:
        image_path: Path to image file
        providers: List of provider names (None = all available)
        ground_truth: Reference text for accuracy calculation
        skip_unavailable: Skip providers that aren't available
        verbose: Print progress

    Returns:
        ComparisonResult with all provider outputs and metrics
    """
    # Get providers to test
    if providers is None:
        providers = ProviderRegistry.list_available() if skip_unavailable else ProviderRegistry.list_all()

    if verbose:
        print(f"Testing {len(providers)} providers on: {image_path}")
        print("-" * 60)

    comparison = ComparisonResult(image_path=image_path, ground_truth=ground_truth)

    for name in providers:
        provider = ProviderRegistry.get(name)

        if provider is None:
            if verbose:
                print(f"  {name}: Not found")
            continue

        if not provider.is_available():
            if skip_unavailable:
                if verbose:
                    print(f"  {name}: Not available (skipped)")
                continue

        if verbose:
            print(f"  {name}: Processing...", end="", flush=True)

        try:
            result = await provider.process_image(image_path)
            comparison.results[name] = result

            if result.success and ground_truth:
                metrics = calculate_metrics(result.text, ground_truth)
                comparison.metrics[name] = metrics

                if verbose:
                    print(f" OK - {result.processing_time_ms:.0f}ms, "
                          f"CER: {metrics.cer:.1%}, WER: {metrics.wer:.1%}")
            elif result.success:
                if verbose:
                    print(f" OK - {result.processing_time_ms:.0f}ms, "
                          f"{result.word_count} words")
            else:
                if verbose:
                    print(f" FAILED: {result.error}")

        except Exception as e:
            if verbose:
                print(f" ERROR: {e}")

    return comparison


def print_comparison_table(comparison: ComparisonResult):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    has_ground_truth = comparison.ground_truth is not None

    # Header
    if has_ground_truth:
        print(f"{'Provider':<20} {'Status':<8} {'Time(ms)':<10} {'Words':<8} {'CER':<8} {'WER':<8} {'Accuracy':<8}")
    else:
        print(f"{'Provider':<20} {'Status':<8} {'Time(ms)':<10} {'Words':<8} {'Confidence':<10}")
    print("-" * 80)

    # Sort by accuracy (if available) or processing time
    def sort_key(item):
        name, result = item
        if has_ground_truth and name in comparison.metrics:
            return comparison.metrics[name].cer  # Lower CER is better
        return result.processing_time_ms if result.success else float('inf')

    sorted_results = sorted(comparison.results.items(), key=sort_key)

    for name, result in sorted_results:
        status = "OK" if result.success else "FAIL"
        time_ms = f"{result.processing_time_ms:.0f}" if result.success else "-"
        words = str(result.word_count) if result.success else "-"

        if has_ground_truth and name in comparison.metrics:
            metrics = comparison.metrics[name]
            print(f"{name:<20} {status:<8} {time_ms:<10} {words:<8} "
                  f"{metrics.cer:<8.1%} {metrics.wer:<8.1%} {metrics.accuracy:<8.1%}")
        else:
            conf = f"{result.confidence:.1%}" if result.success else "-"
            print(f"{name:<20} {status:<8} {time_ms:<10} {words:<8} {conf:<10}")

    print("-" * 80)

    # Best provider summary
    if has_ground_truth and comparison.metrics:
        best = min(comparison.metrics.items(), key=lambda x: x[1].cer)
        print(f"\nBest accuracy: {best[0]} (CER: {best[1].cer:.1%}, Accuracy: {best[1].accuracy:.1%})")

    successful = [name for name, r in comparison.results.items() if r.success]
    if successful:
        fastest = min(successful, key=lambda n: comparison.results[n].processing_time_ms)
        print(f"Fastest: {fastest} ({comparison.results[fastest].processing_time_ms:.0f}ms)")


def run_comparison(
    image_path: str,
    providers: Optional[List[str]] = None,
    ground_truth: Optional[str] = None,
    ground_truth_file: Optional[str] = None,
    output_json: Optional[str] = None
) -> ComparisonResult:
    """
    Synchronous entry point for running comparisons.

    Args:
        image_path: Path to image
        providers: Provider names (None = all available)
        ground_truth: Ground truth text
        ground_truth_file: Path to file containing ground truth
        output_json: Path to save JSON results

    Returns:
        ComparisonResult
    """
    # Load ground truth from file if specified
    if ground_truth_file and os.path.exists(ground_truth_file):
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = f.read()

    # Run comparison
    comparison = asyncio.run(compare_providers(
        image_path=image_path,
        providers=providers,
        ground_truth=ground_truth
    ))

    # Print results
    print_comparison_table(comparison)

    # Save JSON if requested
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(comparison.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_json}")

    return comparison


# CLI interface
if __name__ == "__main__":
    def print_help():
        print("OCR Provider Comparison Tool")
        print("")
        print("Usage:")
        print("  python -m ocr.compare <image> [options]")
        print("")
        print("Options:")
        print("  --providers <list>    Comma-separated provider names")
        print("  --ground-truth <text> Reference text for accuracy")
        print("  --ground-truth-file <path>  File with reference text")
        print("  --output <path>       Save JSON results to file")
        print("  --list                List available providers")
        print("")
        print("Examples:")
        print("  python -m ocr.compare document.jpg")
        print("  python -m ocr.compare scan.png --providers easyocr,paddleocr,doctr")
        print("  python -m ocr.compare page.jpg --ground-truth-file page.txt")

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    if sys.argv[1] == "--list":
        # Import providers to register them
        from . import providers

        print("Registered providers:")
        for name in ProviderRegistry.list_all():
            info = ProviderRegistry.get_info(name)
            available = "available" if name in ProviderRegistry.list_available() else "not available"
            print(f"  {name:<20} ({info['type']:<6}) - {available}, {info['license']}")
        sys.exit(0)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Parse arguments
    providers_arg = None
    ground_truth = None
    ground_truth_file = None
    output_json = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--providers" and i + 1 < len(sys.argv):
            providers_arg = [p.strip() for p in sys.argv[i + 1].split(",")]
            i += 2
        elif sys.argv[i] == "--ground-truth" and i + 1 < len(sys.argv):
            ground_truth = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--ground-truth-file" and i + 1 < len(sys.argv):
            ground_truth_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_json = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    # Import providers to register them
    from . import providers

    # Run comparison
    run_comparison(
        image_path=image_path,
        providers=providers_arg,
        ground_truth=ground_truth,
        ground_truth_file=ground_truth_file,
        output_json=output_json
    )
