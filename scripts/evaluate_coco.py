#!/usr/bin/env python3
"""Evaluate RF-DETR models on COCO val2017 with official COCO metrics."""

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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(model_class, model_name, coco_dir, results_dir, threshold=0.01):
    """Evaluate a single model on COCO val2017."""
    from rfdetr import RFDETRBase, RFDETRLarge

    ann_file = Path(coco_dir) / "annotations" / "instances_val2017.json"
    img_dir = Path(coco_dir) / "val2017"

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"{'='*60}")

    # Load COCO annotations
    coco_gt = COCO(str(ann_file))
    img_ids = coco_gt.getImgIds()
    logger.info(f"  Images: {len(img_ids)}")

    # Load model
    logger.info(f"  Loading {model_name}...")
    model = model_class()
    logger.info(f"  Model loaded.")

    # Run predictions
    coco_results = []
    latencies = []
    warmup = 50

    for idx, img_id in enumerate(tqdm(img_ids, desc=f"  {model_name}")):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = img_dir / img_info["file_name"]
        image = Image.open(str(img_path)).convert("RGB")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        detections = model.predict(image, threshold=threshold)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if idx >= warmup:
            latencies.append((t1 - t0) * 1000)

        # Convert to COCO format
        if hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
            boxes = detections.xyxy  # [N, 4] in xyxy format
            scores = detections.confidence
            class_ids = detections.class_id

            for box, score, cls_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1

                # RF-DETR outputs actual COCO category IDs (1-90, non-contiguous)
                # No remapping needed — the model's classification head has 91 neurons
                # (indices 0-90) trained directly with COCO category_id labels
                coco_results.append({
                    "image_id": img_id,
                    "category_id": int(cls_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                })

    # Save predictions
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pred_file = results_dir / f"{model_name}_predictions.json"
    with open(pred_file, "w") as f:
        json.dump(coco_results, f)
    logger.info(f"  Saved {len(coco_results)} predictions to {pred_file}")

    # Run COCO evaluation
    if len(coco_results) > 0:
        coco_dt = coco_gt.loadRes(str(pred_file))
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            "AP": float(coco_eval.stats[0]),      # AP @[IoU=0.50:0.95]
            "AP50": float(coco_eval.stats[1]),     # AP @IoU=0.50
            "AP75": float(coco_eval.stats[2]),     # AP @IoU=0.75
            "AP_small": float(coco_eval.stats[3]),
            "AP_medium": float(coco_eval.stats[4]),
            "AP_large": float(coco_eval.stats[5]),
            "AR_1": float(coco_eval.stats[6]),
            "AR_10": float(coco_eval.stats[7]),
            "AR_100": float(coco_eval.stats[8]),
        }
    else:
        logger.warning("  No detections!")
        metrics = {}

    # Latency stats
    if latencies:
        metrics["latency_mean_ms"] = float(np.mean(latencies))
        metrics["latency_median_ms"] = float(np.median(latencies))
        metrics["latency_p95_ms"] = float(np.percentile(latencies, 95))
        metrics["fps"] = float(1000 / np.mean(latencies))
        logger.info(f"  Latency: {metrics['latency_mean_ms']:.1f}ms mean, "
                    f"{metrics['fps']:.1f} FPS")

    metrics["model"] = model_name
    metrics["n_predictions"] = len(coco_results)

    with open(results_dir / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-dir", default="data/coco")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models", nargs="+", default=["base", "large"],
                        choices=["base", "large"])
    args = parser.parse_args()

    from rfdetr import RFDETRBase, RFDETRLarge

    model_map = {
        "base": (RFDETRBase, "RF-DETR-Base"),
        "large": (RFDETRLarge, "RF-DETR-Large"),
    }

    all_metrics = {}
    for model_key in args.models:
        model_class, model_name = model_map[model_key]
        metrics = evaluate_model(model_class, model_name, args.coco_dir, args.results_dir)
        all_metrics[model_name] = metrics
        torch.cuda.empty_cache()

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("EVALUATION SUMMARY (COCO val2017)")
    logger.info(f"{'='*70}")
    logger.info(f"{'Model':<20} {'AP':>8} {'AP50':>8} {'AP75':>8} {'FPS':>8}")
    logger.info("-" * 50)
    for name, m in all_metrics.items():
        logger.info(f"{name:<20} {m.get('AP', 0):>8.3f} {m.get('AP50', 0):>8.3f} "
                    f"{m.get('AP75', 0):>8.3f} {m.get('fps', 0):>8.1f}")

    with open(Path(args.results_dir) / "coco_eval_summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
