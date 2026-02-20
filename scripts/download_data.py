#!/usr/bin/env python3
"""Download COCO 2017 val set for evaluation and Aquarium dataset for fine-tuning."""

import os
import sys
import json
import logging
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_file(url, dest, desc=None):
    """Download a file with progress."""
    desc = desc or Path(dest).name
    logger.info(f"Downloading {desc}...")

    def reporthook(count, block_size, total_size):
        pct = min(100, count * block_size * 100 // max(total_size, 1))
        if count % 100 == 0:
            print(f"\r  {pct}%", end="", flush=True)

    urlretrieve(url, dest, reporthook)
    print()
    logger.info(f"  Saved to {dest}")


def download_coco_val(data_dir):
    """Download COCO 2017 validation images and annotations."""
    data_dir = Path(data_dir) / "coco"
    data_dir.mkdir(parents=True, exist_ok=True)

    # COCO val2017 images (1GB)
    images_dir = data_dir / "val2017"
    if not images_dir.exists():
        zip_path = data_dir / "val2017.zip"
        if not zip_path.exists():
            download_file(
                "http://images.cocodataset.org/zips/val2017.zip",
                str(zip_path), "COCO val2017 images"
            )
        logger.info("Extracting val2017 images...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(str(data_dir))
        zip_path.unlink()
    else:
        logger.info("COCO val2017 images already exist.")

    # COCO annotations
    ann_dir = data_dir / "annotations"
    if not ann_dir.exists():
        zip_path = data_dir / "annotations_trainval2017.zip"
        if not zip_path.exists():
            download_file(
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                str(zip_path), "COCO annotations"
            )
        logger.info("Extracting annotations...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(str(data_dir))
        zip_path.unlink()
    else:
        logger.info("COCO annotations already exist.")

    logger.info(f"COCO val2017 ready at {data_dir}")
    return data_dir


def download_aquarium(data_dir):
    """Download Aquarium detection dataset from Roboflow (COCO format)."""
    data_dir = Path(data_dir) / "aquarium"

    if (data_dir / "train" / "_annotations.coco.json").exists():
        logger.info("Aquarium dataset already exists.")
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)

    # Aquarium dataset v2 from Roboflow (public, no API key needed)
    url = "https://universe.roboflow.com/ds/qjfrWiqHdf?key=oXAhDVHjy2"

    zip_path = data_dir / "aquarium.zip"
    download_file(url, str(zip_path), "Aquarium dataset")

    logger.info("Extracting Aquarium dataset...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(str(data_dir))
    zip_path.unlink()

    # Verify structure
    for split in ["train", "valid", "test"]:
        ann_file = data_dir / split / "_annotations.coco.json"
        if ann_file.exists():
            with open(ann_file) as f:
                ann = json.load(f)
            logger.info(f"  {split}: {len(ann['images'])} images, "
                       f"{len(ann['annotations'])} annotations, "
                       f"{len(ann['categories'])} categories")

    logger.info(f"Aquarium dataset ready at {data_dir}")
    return data_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", choices=["coco", "aquarium", "all"], default="all")
    args = parser.parse_args()

    if args.dataset in ("coco", "all"):
        download_coco_val(args.data_dir)
    if args.dataset in ("aquarium", "all"):
        download_aquarium(args.data_dir)
