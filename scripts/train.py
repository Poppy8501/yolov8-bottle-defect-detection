#!/usr/bin/env python3
"""Train the improved YOLOv8 (CBAM + WIoU) defect detector.

Example:
    python scripts/train.py --data-root data/yolo_bottle --epochs 120 \
        --imgsz 640 --batch 16 --device 0
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.loss import apply_wiou_patch  # noqa: E402
from src.modules import apply_cbam_patch, register_custom_modules  # noqa: E402
from scripts.build_model_yaml import build_cbam_yaml, find_base_yaml  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/yolo_bottle"),
                        help="Folder produced by prepare_dataset.py (contains data.yaml)")
    parser.add_argument("--data-yaml", type=Path, default=None,
                        help="Override dataset YAML path")
    parser.add_argument("--model-yaml", type=Path,
                        default=ROOT / "configs" / "yolov8n_cbam.yaml")
    parser.add_argument("--weights", default=str(ROOT / "weights" / "yolov8n.pt"),
                        help="Pretrained weights to transfer (default: bundled yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0",
                        help="CUDA device index, 'cpu', or '0,1'")
    parser.add_argument("--workers", type=int,
                        default=0 if os.name == "nt" else 4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--copy-paste", type=float, default=0.0,
                        help="Copy-paste augmentation probability, useful for few defects")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision (useful on CPU)")
    parser.add_argument("--no-cbam", action="store_true",
                        help="Disable CBAM attention (ablation baseline)")
    parser.add_argument("--no-wiou", action="store_true",
                        help="Disable WIoU loss (ablation baseline)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train from random initialization")
    parser.add_argument("--project", type=Path, default=ROOT / "outputs" / "runs")
    parser.add_argument("--name", default="yolov8n_cbam_wiou")
    parser.add_argument("--out-weights", type=Path,
                        default=ROOT / "outputs" / "best.pt")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    data_yaml = args.data_yaml or args.data_root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"{data_yaml} not found. Run scripts/prepare_dataset.py first."
        )

    if not args.model_yaml.is_file():
        base = find_base_yaml("n")
        build_cbam_yaml(base, args.model_yaml, nc=3)

    register_custom_modules()
    if not args.no_cbam:
        apply_cbam_patch()
        print("[info] CBAM attention enabled on all C2f blocks")
    else:
        print("[info] CBAM disabled (original C2f blocks)")
    if not args.no_wiou:
        apply_wiou_patch()
        print("[info] WIoU loss enabled")
    else:
        print("[info] WIoU disabled (original CIoU loss)")

    from ultralytics import YOLO

    model = YOLO(str(args.model_yaml), task="detect")

    if not args.no_pretrained:
        try:
            model.load(args.weights)
            print(f"[info] transferred weights from {args.weights}")
        except Exception as exc:  # noqa: BLE001 - fallback should be explicit
            print(f"[warn] could not load {args.weights}: {exc}")
            print("[warn] continuing from random initialization")

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        lr0=args.lr0,
        mosaic=args.mosaic,
        copy_paste=args.copy_paste,
        close_mosaic=args.close_mosaic,
        amp=not args.no_amp,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        plots=True,
        pretrained=False,
    )

    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        device=args.device,
        project=str(args.project),
        name=f"{args.name}_val",
    )
    print(
        "[result] mAP50={:.4f} mAP50-95={:.4f}".format(
            metrics.box.map50, metrics.box.map
        )
    )

    args.out_weights.parent.mkdir(parents=True, exist_ok=True)
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.is_file():
        shutil.copy2(best, args.out_weights)
        print(f"[info] best weights saved to {args.out_weights}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
