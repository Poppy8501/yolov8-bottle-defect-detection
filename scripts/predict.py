#!/usr/bin/env python3
"""Shared inference helpers for CLI and the Streamlit app."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modules import apply_cbam_patch, register_custom_modules  # noqa: E402


def load_model(weights: str | Path) -> Any:
    """Load a trained detector, restoring the CBAM structure if present."""
    register_custom_modules()
    if os.environ.get("DEFECT_NO_CBAM") != "1":
        apply_cbam_patch()
    from ultralytics import YOLO

    return YOLO(str(weights), task="detect")


def predict_image(
    model: Any,
    image: str | Path | np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return an annotated BGR image and a list of detections.

    Each detection dict contains ``class_id``, ``class_name``, ``confidence``
    and ``box`` (xyxy pixel coordinates).
    """
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=False,
        verbose=False,
    )
    result = results[0]
    annotated = result.plot()

    detections: list[dict[str, Any]] = []
    if result.boxes is not None and len(result.boxes) > 0:
        names = result.names or {}
        for box in result.boxes:
            cls_id = int(box.cls.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "confidence": float(box.conf.item()),
                    "box": xyxy,
                }
            )
    return annotated, detections


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run defect detection on an image or folder")
    parser.add_argument("--weights", default="outputs/best.pt")
    parser.add_argument("--source", required=True, help="Image file or folder")
    parser.add_argument("--out", default="outputs/predictions", type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser


def main(argv: list[str] | None = None) -> int:
    import cv2

    args = build_arg_parser().parse_args(argv)
    model = load_model(args.weights)
    args.out.mkdir(parents=True, exist_ok=True)

    source = Path(args.source)
    files = (
        sorted(p for p in source.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if source.is_dir()
        else [source]
    )
    for path in files:
        annotated, detections = predict_image(model, path, conf=args.conf, imgsz=args.imgsz)
        ext = path.suffix.lower() if path.suffix.lower() in {".png", ".jpg", ".jpeg"} else ".png"
        ok, encoded = cv2.imencode(ext, annotated)
        if not ok:
            raise RuntimeError(f"Cannot encode result for {path}")
        (args.out / f"{path.stem}_det{ext}").write_bytes(encoded.tobytes())
        if detections:
            print(f"{path.name}: {len(detections)} defect(s) -> " +
                  ", ".join(f"{d['class_name']}={d['confidence']:.2f}" for d in detections))
        else:
            print(f"{path.name}: 没有缺陷，产品完好")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
