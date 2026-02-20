#!/usr/bin/env python3
"""Prepare a COCO subset in Roboflow format for fine-tuning demonstration.

Creates a focused 'animals' detection dataset from COCO val2017 by extracting
images containing animal categories and converting to Roboflow COCO format.
"""

import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path

from pycocotools.coco import COCO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Animal categories from COCO (supercategory='animal')
ANIMAL_CATS = {
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
}


def prepare_animal_subset(coco_dir, output_dir, train_ratio=0.75, max_images=500):
    """Extract animal images from COCO and create Roboflow-format dataset."""
    coco_dir = Path(coco_dir)
    output_dir = Path(output_dir)

    ann_file = coco_dir / "annotations" / "instances_val2017.json"
    img_dir = coco_dir / "val2017"

    logger.info("Loading COCO annotations...")
    coco = COCO(str(ann_file))

    # Get all images with animal annotations
    animal_cat_ids = list(ANIMAL_CATS.keys())
    animal_img_ids = set()
    for cat_id in animal_cat_ids:
        animal_img_ids.update(coco.getImgIds(catIds=[cat_id]))
    animal_img_ids = sorted(animal_img_ids)[:max_images]

    logger.info(f"Found {len(animal_img_ids)} images with animals")

    # Split into train/valid/test (75/15/10)
    n_train = int(len(animal_img_ids) * train_ratio)
    n_valid = int(len(animal_img_ids) * 0.15)

    splits = {
        "train": animal_img_ids[:n_train],
        "valid": animal_img_ids[n_train:n_train + n_valid],
        "test": animal_img_ids[n_train + n_valid:],
    }

    # Remap category IDs to 0-indexed contiguous IDs for Roboflow format
    new_categories = []
    old_to_new = {}
    for i, (old_id, name) in enumerate(sorted(ANIMAL_CATS.items())):
        new_id = i
        old_to_new[old_id] = new_id
        new_categories.append({
            "id": new_id,
            "name": name,
            "supercategory": "animal",
        })

    for split_name, img_ids in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        coco_ann = {
            "images": [],
            "annotations": [],
            "categories": new_categories,
        }

        ann_id = 0
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            src_path = img_dir / img_info["file_name"]
            dst_path = split_dir / img_info["file_name"]

            if not src_path.exists():
                continue

            shutil.copy2(str(src_path), str(dst_path))

            coco_ann["images"].append({
                "id": img_id,
                "file_name": img_info["file_name"],
                "width": img_info["width"],
                "height": img_info["height"],
            })

            # Get annotations for this image (only animal categories)
            ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=animal_cat_ids)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                if ann["category_id"] in old_to_new:
                    coco_ann["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": old_to_new[ann["category_id"]],
                        "bbox": ann["bbox"],
                        "area": ann["area"],
                        "iscrowd": ann.get("iscrowd", 0),
                    })
                    ann_id += 1

        ann_file_out = split_dir / "_annotations.coco.json"
        with open(ann_file_out, "w") as f:
            json.dump(coco_ann, f)

        logger.info(f"  {split_name}: {len(coco_ann['images'])} images, "
                    f"{len(coco_ann['annotations'])} annotations")

    # Count per-class annotations
    cat_counts = defaultdict(int)
    for split_name in splits:
        ann_file_path = output_dir / split_name / "_annotations.coco.json"
        with open(ann_file_path) as f:
            anns = json.load(f)
        for a in anns["annotations"]:
            cat_counts[a["category_id"]] += 1

    logger.info("\nClass distribution:")
    for cat in new_categories:
        logger.info(f"  {cat['name']}: {cat_counts.get(cat['id'], 0)} annotations")

    logger.info(f"\nDataset ready at: {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-dir", default="data/coco")
    parser.add_argument("--output-dir", default="data/coco_animals")
    args = parser.parse_args()

    prepare_animal_subset(args.coco_dir, args.output_dir)
