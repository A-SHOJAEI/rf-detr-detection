#!/usr/bin/env python3
"""Fine-tune RF-DETR on the Aquarium dataset to demonstrate transfer learning."""

import sys
sys.path.insert(0, ".")

import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def finetune(model_class, model_name, dataset_dir, output_dir, epochs=50, batch_size=8):
    """Fine-tune a model on the Aquarium dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Fine-tuning: {model_name} on Aquarium")
    logger.info(f"{'='*60}")

    model = model_class()
    output_path = Path(output_dir) / model_name.lower().replace("-", "_")

    logger.info(f"  Dataset: {dataset_dir}")
    logger.info(f"  Output:  {output_path}")
    logger.info(f"  Epochs:  {epochs}")
    logger.info(f"  Batch:   {batch_size}")

    model.train(
        dataset_dir=str(dataset_dir),
        dataset_file="roboflow",
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=4,
        lr=1e-4,
        lr_encoder=1.5e-4,
        output_dir=str(output_path),
        use_ema=True,
        multi_scale=True,
        early_stopping=True,
        early_stopping_patience=10,
        checkpoint_interval=10,
        tensorboard=True,
        wandb=False,
    )

    logger.info(f"  Fine-tuning complete. Best model: {output_path / 'checkpoint_best_total.pth'}")

    # Load fine-tuned model and run sample predictions
    logger.info("  Running sample predictions with fine-tuned model...")
    ft_model = model_class(pretrain_weights=str(output_path / "checkpoint_best_total.pth"))

    test_dir = Path(dataset_dir) / "test"
    ann_file = test_dir / "_annotations.coco.json"
    if ann_file.exists():
        with open(ann_file) as f:
            test_ann = json.load(f)

        n_test = len(test_ann["images"])
        categories = {c["id"]: c["name"] for c in test_ann["categories"]}
        logger.info(f"  Test set: {n_test} images, {len(categories)} classes")
        logger.info(f"  Classes: {list(categories.values())}")

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/aquarium")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", choices=["base", "large"], default="base")
    args = parser.parse_args()

    from rfdetr import RFDETRBase, RFDETRLarge

    model_map = {
        "base": (RFDETRBase, "RF-DETR-Base"),
        "large": (RFDETRLarge, "RF-DETR-Large"),
    }

    model_class, model_name = model_map[args.model]
    finetune(model_class, model_name, args.dataset_dir, args.output_dir,
             epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
