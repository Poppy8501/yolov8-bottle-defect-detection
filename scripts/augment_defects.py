#!/usr/bin/env python3
"""Expand defect samples with label-preserving augmentation.

The MVTec AD bottle subset only has ~51 defect training images. This script
creates 4 extra views per defect image (horizontal flip, vertical flip, 90°
rotation, brightness shift) while transforming the YOLO boxes accordingly.
Normal images are copied unchanged.

Example:
    python scripts/augment_defects.py \
        --src data/yolo_bottle --out data/yolo_bottle_aug
    python scripts/train.py --data-root data/yolo_bottle_aug --epochs 80 ...
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def write_image(path: Path, img: np.ndarray) -> None:
    ext = ".png" if path.suffix.lower() not in {".jpg", ".jpeg"} else path.suffix.lower()
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    path.with_suffix(ext).write_bytes(encoded.tobytes())


def parse_labels(text: str) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def format_labels(boxes: list[tuple[int, float, float, float, float]]) -> str:
    return "".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for c, xc, yc, w, h in boxes)


def transform_boxes(
    boxes: list[tuple[int, float, float, float, float]], mode: str
) -> list[tuple[int, float, float, float, float]]:
    out = []
    for cls, xc, yc, w, h in boxes:
        if mode == "hflip":
            out.append((cls, 1.0 - xc, yc, w, h))
        elif mode == "vflip":
            out.append((cls, xc, 1.0 - yc, w, h))
        elif mode == "rot90":
            out.append((cls, 1.0 - yc, xc, h, w))
        else:  # brightness, boxes unchanged
            out.append((cls, xc, yc, w, h))
    return out


def augment_image(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "hflip":
        return cv2.flip(img, 1)
    if mode == "vflip":
        return cv2.flip(img, 0)
    if mode == "rot90":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if mode == "bright":
        return cv2.convertScaleAbs(img, alpha=1.2, beta=12)
    raise ValueError(f"Unknown mode: {mode}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=ROOT / "data" / "yolo_bottle")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "yolo_bottle_aug")
    parser.add_argument("--modes", nargs="+", default=["hflip", "vflip", "rot90", "bright"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    src, out = args.src, args.out

    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"original_defect": 0, "augmented": 0, "negative": 0}
    for split in ("train", "val"):
        images = sorted(p for p in (src / "images" / split).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for image_path in images:
            label_path = src / "labels" / split / (image_path.stem + ".txt")
            text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
            boxes = parse_labels(text)
            img = read_image(image_path)

            dst_stem = image_path.stem
            write_image(out / "images" / split / f"{dst_stem}.png", img)
            (out / "labels" / split / f"{dst_stem}.txt").write_text(text, encoding="utf-8")

            if boxes and split == "train":
                counts["original_defect"] += 1
                for mode in args.modes:
                    aug_name = f"{dst_stem}_{mode}"
                    write_image(out / "images" / split / f"{aug_name}.png", augment_image(img, mode))
                    aug_boxes = transform_boxes(boxes, mode)
                    (out / "labels" / split / f"{aug_name}.txt").write_text(
                        format_labels(aug_boxes), encoding="utf-8"
                    )
                    counts["augmented"] += 1
            elif not boxes:
                counts["negative"] += 1

    cfg = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
    cfg["path"] = str(out.resolve())
    (out / "data.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    total_train = len(list((out / "images" / "train").glob("*")))
    total_val = len(list((out / "images" / "val").glob("*")))
    print(f"[done] train={total_train} val={total_val} counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
