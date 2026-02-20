#!/usr/bin/env python3
"""Generate publication-quality visualizations for RF-DETR evaluation."""

import sys
sys.path.insert(0, ".")

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
})

COLORS = {
    "RF-DETR-Base": "#2196F3",
    "RF-DETR-Large": "#FF5722",
}


def load_results(results_dir):
    """Load all result JSON files."""
    results_dir = Path(results_dir)
    data = {}

    for name in ["RF-DETR-Base", "RF-DETR-Large"]:
        metrics_file = results_dir / f"{name}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data[name] = json.load(f)

    latency_file = results_dir / "latency_benchmark.json"
    if latency_file.exists():
        with open(latency_file) as f:
            data["latency"] = json.load(f)

    ft_metrics_file = results_dir / "finetune_metrics.json"
    if ft_metrics_file.exists():
        with open(ft_metrics_file) as f:
            data["finetune"] = json.load(f)

    return data


def plot_coco_metrics(data, save_dir):
    """Bar chart comparing Base vs Large on COCO metrics."""
    metrics = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
    labels = ["AP", "AP50", "AP75", "AP$_S$", "AP$_M$", "AP$_L$"]

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(metrics))
    width = 0.35

    for i, (name, color) in enumerate(COLORS.items()):
        if name in data:
            vals = [data[name].get(m, 0) * 100 for m in metrics]
            bars = ax.bar(x + (i - 0.5) * width, vals, width, label=name,
                         color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                       f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Average Precision (%)", fontsize=12)
    ax.set_title("RF-DETR COCO val2017 Evaluation", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 85)
    ax.legend(fontsize=11, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_dir / "coco_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved coco_metrics.png")


def plot_ar_metrics(data, save_dir):
    """Bar chart for recall metrics."""
    metrics = ["AR_1", "AR_10", "AR_100"]
    labels = ["AR$_1$", "AR$_{10}$", "AR$_{100}$"]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(metrics))
    width = 0.35

    for i, (name, color) in enumerate(COLORS.items()):
        if name in data:
            vals = [data[name].get(m, 0) * 100 for m in metrics]
            bars = ax.bar(x + (i - 0.5) * width, vals, width, label=name,
                         color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                       f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Average Recall (%)", fontsize=12)
    ax.set_title("RF-DETR Average Recall by Max Detections", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_dir / "recall_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved recall_metrics.png")


def plot_latency(data, save_dir):
    """Latency comparison across configurations."""
    if "latency" not in data:
        logger.warning("  No latency data found, skipping latency plot")
        return

    latency_data = data["latency"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Group by model
    for ax, model_name in zip([ax1, ax2], ["RF-DETR-Base", "RF-DETR-Large"]):
        model_data = [d for d in latency_data if d["model"] == model_name]
        if not model_data:
            continue

        labels = []
        means = []
        p5s = []
        p95s = []

        for d in model_data:
            opt = "Optimized" if d["optimized"] else "Default"
            dtype = "FP16" if "float16" in d["dtype"] else "FP32"
            labels.append(f"{opt}\n{dtype}")
            means.append(d["mean_ms"])
            p5s.append(d["p5_ms"])
            p95s.append(d["p95_ms"])

        x = np.arange(len(labels))
        color = COLORS[model_name]
        bars = ax.bar(x, means, 0.5, color=color, alpha=0.85,
                     edgecolor="white", linewidth=0.5)

        # Error bars for p5-p95 range
        for xi, m, p5, p95 in zip(x, means, p5s, p95s):
            ax.plot([xi, xi], [p5, p95], color="black", linewidth=1.5, zorder=3)
            ax.plot([xi - 0.1, xi + 0.1], [p5, p5], color="black", linewidth=1.5, zorder=3)
            ax.plot([xi - 0.1, xi + 0.1], [p95, p95], color="black", linewidth=1.5, zorder=3)

        for bar, m, d in zip(bars, means, model_data):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                   f"{m:.1f}ms\n({d['fps']:.0f} FPS)", ha="center", va="bottom",
                   fontsize=9, fontweight="bold")

        ax.set_title(model_name, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Latency (ms)", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("RF-DETR Inference Latency (RTX 4090)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "latency_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved latency_benchmark.png")


def plot_size_vs_accuracy(data, save_dir):
    """Scatter plot of model size vs AP, with latency as bubble size."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Published model sizes (approximate parameter counts)
    model_info = {
        "RF-DETR-Base": {"params": 29, "res": 560},
        "RF-DETR-Large": {"params": 128, "res": 704},
    }

    # Add comparison models from literature
    comparisons = [
        ("YOLOv11-L", 25.3, 49.8, "#9E9E9E"),
        ("YOLOv11-X", 56.9, 50.5, "#9E9E9E"),
        ("DEIM-D-SAVP-R50", 33, 51.7, "#9E9E9E"),
        ("RT-DETRv2-L", 42, 53.4, "#9E9E9E"),
        ("D-FINE-L", 31, 54.0, "#9E9E9E"),
    ]

    for name, params, ap, color in comparisons:
        ax.scatter(params, ap, s=100, c=color, alpha=0.5, edgecolors="gray", linewidths=0.5)
        ax.annotate(name, (params, ap), textcoords="offset points",
                   xytext=(5, 5), fontsize=8, color="gray")

    for name, color in COLORS.items():
        if name in data:
            ap = data[name]["AP"] * 100
            params = model_info[name]["params"]
            ax.scatter(params, ap, s=200, c=color, alpha=0.9,
                      edgecolors="black", linewidths=1, zorder=5)
            ax.annotate(name, (params, ap), textcoords="offset points",
                       xytext=(8, -5), fontsize=10, fontweight="bold", color=color)

    ax.set_xlabel("Parameters (M)", fontsize=12)
    ax.set_ylabel("AP (COCO val2017)", fontsize=12)
    ax.set_title("Accuracy vs Model Size", fontsize=14, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_dir / "accuracy_vs_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved accuracy_vs_size.png")


def plot_detection_samples(save_dir):
    """Generate sample detection visualizations on COCO val images."""
    try:
        import supervision as sv
        from rfdetr import RFDETRLarge
        from rfdetr.util.coco_classes import COCO_CLASSES
        from PIL import Image
    except ImportError:
        logger.warning("  Cannot import supervision/rfdetr, skipping detection samples")
        return

    coco_dir = Path("data/coco/val2017")
    if not coco_dir.exists():
        logger.warning("  COCO val2017 images not found, skipping detection samples")
        return

    # Pick diverse sample images
    sample_images = [
        "000000000139.jpg",  # People on bench
        "000000001503.jpg",  # Cars/street
        "000000002299.jpg",  # Kitchen
        "000000005037.jpg",  # Outdoor scene
        "000000007816.jpg",  # Animals
        "000000009448.jpg",  # Sports
    ]

    # Filter to existing images
    available = [img for img in sample_images if (coco_dir / img).exists()]
    if not available:
        # Fall back to first 6 images
        available = sorted(coco_dir.glob("*.jpg"))[:6]
        available = [p.name for p in available]

    if len(available) < 2:
        logger.warning("  Not enough sample images, skipping detection samples")
        return

    logger.info(f"  Generating detections on {len(available)} sample images...")
    model = RFDETRLarge()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1,
                                         text_padding=3)

    for idx, (ax, img_name) in enumerate(zip(axes, available[:6])):
        img_path = coco_dir / img_name
        image = np.array(Image.open(str(img_path)).convert("RGB"))
        detections = model.predict(Image.open(str(img_path)).convert("RGB"), threshold=0.35)

        labels = [
            f"{COCO_CLASSES.get(cls_id, 'unk')} {conf:.2f}"
            for cls_id, conf in zip(detections.class_id, detections.confidence)
        ]

        annotated = box_annotator.annotate(image.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels)

        ax.imshow(annotated)
        ax.set_title(f"{img_name} ({len(detections)} dets)", fontsize=10)
        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(available[:6]):]:
        ax.axis("off")

    fig.suptitle("RF-DETR-Large Detections (threshold=0.35)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "detection_samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved detection_samples.png")

    del model
    import torch
    torch.cuda.empty_cache()


def plot_finetune_results(data, save_dir):
    """Plot fine-tuning training curves if available."""
    if "finetune" not in data:
        return

    ft = data["finetune"]
    if "epochs" not in ft:
        return

    epochs = ft["epochs"]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    if "train_loss" in ft:
        ax1.plot(epochs, ft["train_loss"], color="#2196F3", linewidth=2, label="Train")
    if "val_loss" in ft:
        ax1.plot(epochs, ft["val_loss"], color="#FF5722", linewidth=2, label="Val")
    ax1.set_title("Training Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # AP
    if "ap50" in ft:
        ax2.plot(epochs, ft["ap50"], color="#4CAF50", linewidth=2, label="AP50")
    if "ap" in ft:
        ax2.plot(epochs, ft["ap"], color="#FF9800", linewidth=2, label="AP50:95")
    ax2.set_title("Average Precision", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AP")
    ax2.legend()

    # AR
    if "ar100" in ft:
        ax3.plot(epochs, ft["ar100"], color="#9C27B0", linewidth=2)
    ax3.set_title("Average Recall (AR100)", fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("AR")

    # LR
    if "lr" in ft:
        ax4.plot(epochs, ft["lr"], color="#607D8B", linewidth=2)
    ax4.set_title("Learning Rate", fontweight="bold")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("LR")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("RF-DETR Fine-tuning on Aquarium Dataset", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "finetune_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved finetune_curves.png")


def main():
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading results...")
    data = load_results(results_dir)

    if not data:
        logger.error("No results found. Run evaluation first.")
        return

    logger.info("Generating visualizations...")
    plot_coco_metrics(data, results_dir)
    plot_ar_metrics(data, results_dir)
    plot_latency(data, results_dir)
    plot_size_vs_accuracy(data, results_dir)
    plot_detection_samples(results_dir)
    plot_finetune_results(data, results_dir)

    logger.info("\nAll visualizations saved to results/")


if __name__ == "__main__":
    main()
