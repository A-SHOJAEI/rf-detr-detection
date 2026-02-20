#!/usr/bin/env python3
"""Benchmark RF-DETR inference latency on RTX 4090."""

import sys
sys.path.insert(0, ".")

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WARMUP_ITERS = 50
BENCH_ITERS = 300


def benchmark_model(model_class, model_name, resolution, optimize=False, dtype=torch.float32):
    """Benchmark a single model's inference latency."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {model_name} (optimize={optimize}, dtype={dtype})")
    logger.info(f"{'='*60}")

    model = model_class()

    if optimize:
        logger.info("  Optimizing for inference...")
        model.optimize_for_inference(batch_size=1, dtype=dtype)

    # Create a realistic test image
    dummy = Image.fromarray(np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8))

    # Warmup
    logger.info(f"  Warming up ({WARMUP_ITERS} iters)...")
    for _ in range(WARMUP_ITERS):
        model.predict(dummy, threshold=0.5)
    torch.cuda.synchronize()

    # Benchmark
    logger.info(f"  Benchmarking ({BENCH_ITERS} iters)...")
    latencies = []
    for _ in range(BENCH_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(dummy, threshold=0.5)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    latencies = np.array(latencies)
    results = {
        "model": model_name,
        "optimized": optimize,
        "dtype": str(dtype),
        "resolution": resolution,
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "p5_ms": float(np.percentile(latencies, 5)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "fps": float(1000 / np.mean(latencies)),
    }

    logger.info(f"  Mean:   {results['mean_ms']:.2f} ms ({results['fps']:.1f} FPS)")
    logger.info(f"  Median: {results['median_ms']:.2f} ms")
    logger.info(f"  P5-P95: {results['p5_ms']:.2f} - {results['p95_ms']:.2f} ms")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    from rfdetr import RFDETRBase, RFDETRLarge

    all_results = []

    configs = [
        (RFDETRBase, "RF-DETR-Base", 560, False, torch.float32),
        (RFDETRBase, "RF-DETR-Base", 560, True, torch.float32),
        (RFDETRBase, "RF-DETR-Base", 560, True, torch.float16),
        (RFDETRLarge, "RF-DETR-Large", 704, False, torch.float32),
        (RFDETRLarge, "RF-DETR-Large", 704, True, torch.float32),
        (RFDETRLarge, "RF-DETR-Large", 704, True, torch.float16),
    ]

    for model_class, name, res, opt, dtype in configs:
        try:
            result = benchmark_model(model_class, name, res, optimize=opt, dtype=dtype)
            all_results.append(result)
        except Exception as e:
            logger.error(f"  Failed: {e}")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("LATENCY BENCHMARK SUMMARY (RTX 4090)")
    logger.info(f"{'='*80}")
    logger.info(f"{'Model':<20} {'Opt':>5} {'Dtype':>8} {'Mean(ms)':>10} {'P95(ms)':>10} {'FPS':>8}")
    logger.info("-" * 65)
    for r in all_results:
        opt_str = "Yes" if r["optimized"] else "No"
        dtype_str = "fp16" if "float16" in r["dtype"] else "fp32"
        logger.info(f"{r['model']:<20} {opt_str:>5} {dtype_str:>8} "
                    f"{r['mean_ms']:>10.2f} {r['p95_ms']:>10.2f} {r['fps']:>8.1f}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "latency_benchmark.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
