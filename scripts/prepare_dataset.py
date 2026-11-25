#!/usr/bin/env python3
"""Convert one MVTec AD category into a supervised YOLO detection dataset.

MVTec AD ships defect-free images in ``train/good`` and pixel-precise masks in
``ground_truth``. YOLOv8 needs bounding boxes, so this script converts every
mask component to a YOLO box, splits defective images into train/val, and adds
normal images as negative samples (no objects => empty label file).

Example:
    python scripts/prepare_dataset.py \
        --mvtec-root data/mvtec_anomaly_detection \
        --category bottle \
        --out data/yolo_bottle
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def discover_defect_classes(category_dir: Path) -> list[str]:
    """Return sorted defect folder names under ``test/`` (excluding 'good')."""
    test_dir = category_dir / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")
    classes = sorted(
        p.name for p in test_dir.iterdir() if p.is_dir() and p.name != "good"
    )
    if not classes:
        raise ValueError(f"No defect classes found under {test_dir}")
    return classes


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def find_mask_path(category_dir: Path, defect: str, stem: str) -> Path:
    """Locate a mask, supporting both official and HF-mirror naming."""
    base = category_dir / "ground_truth" / defect
    for name in (f"{stem}.png", f"{stem}_mask.png"):
        candidate = base / name
        if candidate.exists():
            return candidate
    return base / f"{stem}.png"


def mask_to_yolo_lines(
    mask_path: Path,
    class_id: int,
    min_area: int = 10,
) -> list[str]:
    """Convert a binary anomaly mask to normalized YOLO bounding-box lines.

    Each connected component becomes one box: ``cls xc yc w h``.
    """
    # imdecode supports Unicode paths on Windows, unlike cv2.imread.
    data = np.fromfile(str(mask_path), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    binary = (mask > 0).astype("uint8")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape

    lines: list[str] = []
    for component in range(1, count):  # 0 is background
        x, y, box_w, box_h, area = stats[component]
        if area < min_area or box_w == 0 or box_h == 0:
            continue
        xc = (x + box_w / 2) / width
        yc = (y + box_h / 2) / height
        w = box_w / width
        h = box_h / height
        lines.append(
            f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        )
    return lines


def _copy_split(
    images: Iterable[Path],
    split: str,
    out_dir: Path,
    class_names: list[str],
    copy_negatives: bool,
) -> dict[str, int]:
    """Copy images and write matching (possibly empty) label files."""
    image_dir = out_dir / "images" / split
    label_dir = out_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name in class_names}
    counts["negative"] = 0

    for image_path in images:
        rel_parts = image_path.parts
        class_name = rel_parts[-2] if len(rel_parts) >= 2 else "unknown"
        is_negative = class_name not in class_names
        if is_negative and not copy_negatives:
            continue

        dst_image = image_dir / image_path.name
        # Defect folders have unique names in MVTec AD, but be defensive.
        if dst_image.exists():
            dst_image = image_dir / f"{image_path.stem}_{class_name}{image_path.suffix}"
        shutil.copy2(image_path, dst_image)

        label_path = label_dir / (dst_image.stem + ".txt")
        if is_negative:
            label_path.write_text("", encoding="utf-8")
            counts["negative"] += 1
            continue

        mask_path = find_mask_path(
            image_path.parents[2], class_name, image_path.stem
        )
        lines = mask_to_yolo_lines(
            mask_path, class_names.index(class_name), min_area=10
        )
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[class_name] += 1

    return counts


def stratified_split(
    items_by_class: dict[str, list[Path]], val_ratio: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Split each defect class independently, keeping at least one val image."""
    rng = random.Random(seed)
    train: list[Path] = []
    val: list[Path] = []
    for class_name, items in items_by_class.items():
        shuffled = sorted(items)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) >= 5 else 0
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val


def prepare_dataset(
    mvtec_root: Path,
    category: str,
    out_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    neg_train: int = 60,
) -> dict:
    """Run the full conversion and return a statistics dictionary."""
    category_dir = mvtec_root / category
    if not category_dir.is_dir():
        raise FileNotFoundError(f"Category directory not found: {category_dir}")

    class_names = discover_defect_classes(category_dir)

    items_by_class: dict[str, list[Path]] = {}
    for name in class_names:
        defect_dir = category_dir / "test" / name
        items_by_class[name] = list_images(defect_dir)
        for image_path in items_by_class[name]:
            mask_path = find_mask_path(category_dir, name, image_path.stem)
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing annotation for {image_path}")

    train_defects, val_defects = stratified_split(items_by_class, val_ratio, seed)

    good_train = list_images(category_dir / "train" / "good")
    good_test = list_images(category_dir / "test" / "good")
    rng = random.Random(seed)
    rng.shuffle(good_train)
    neg_train_images = good_train[:neg_train]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_counts = _copy_split(
        train_defects + neg_train_images, "train", out_dir, class_names, True
    )
    val_counts = _copy_split(val_defects + good_test, "val", out_dir, class_names, True)

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names),
    }
    (out_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    stats = {
        "category": category,
        "classes": class_names,
        "train_images": len(train_defects) + len(neg_train_images),
        "val_images": len(val_defects) + len(good_test),
        "train_counts": train_counts,
        "val_counts": val_counts,
        "seed": seed,
        "val_ratio": val_ratio,
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvtec-root", type=Path, required=True,
                        help="Root folder containing MVTec AD categories")
    parser.add_argument("--category", default="bottle",
                        help="MVTec AD category to convert (default: bottle)")
    parser.add_argument("--out", type=Path, default=Path("data/yolo_bottle"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg-train", type=int, default=60,
                        help="Number of normal images used as train negatives")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    stats = prepare_dataset(
        mvtec_root=args.mvtec_root,
        category=args.category,
        out_dir=args.out,
        val_ratio=args.val_ratio,
        seed=args.seed,
        neg_train=args.neg_train,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
