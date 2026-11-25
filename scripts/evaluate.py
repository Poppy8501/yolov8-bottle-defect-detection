#!/usr/bin/env python3
"""Evaluate the trained detector and write a Markdown report.

Example:
    python scripts/evaluate.py --weights outputs/best.pt \
        --data-yaml data/yolo_bottle/data.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modules import apply_cbam_patch, register_custom_modules  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "outputs" / "best.pt")
    parser.add_argument("--data-yaml", type=Path,
                        default=ROOT / "data" / "yolo_bottle" / "data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    register_custom_modules()
    if os.environ.get("DEFECT_NO_CBAM") != "1":
        apply_cbam_patch()

    from ultralytics import YOLO

    model = YOLO(str(args.weights), task="detect")
    metrics = model.val(
        data=str(args.data_yaml),
        imgsz=args.imgsz,
        device=args.device,
        split=args.split,
        project=str(args.out_dir),
        name="val_plots",
        plots=True,
        verbose=True,
    )
    box = metrics.box
    mAP50 = float(box.map50)
    mAP5095 = float(box.map)
    precision = float(box.mp)
    recall = float(box.mr)
    per_class = [float(v) for v in getattr(box, "maps", [])]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "weights": str(args.weights),
        "split": args.split,
        "mAP50": mAP50,
        "mAP50_95": mAP5095,
        "precision": precision,
        "recall": recall,
        "per_class_mAP50_95": per_class,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    names = model.names if hasattr(model, "names") else {}
    class_rows = "\n".join(
        f"| {names.get(i, i)} | {value:.4f} |"
        for i, value in enumerate(per_class)
    )
    report = f"""# 检测结果

权重：`{args.weights}`，评估划分：`{args.split}`

| 指标 | 数值 |
| --- | --- |
| mAP@0.5 | {mAP50:.4f} |
| mAP@0.5:0.95 | {mAP5095:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |

## 各类别 mAP@0.5:0.95

| 类别 | mAP |
| --- | --- |
{class_rows if class_rows else "| - | - |"}

可视化结果见 `{args.out_dir}` 下的 confusion_matrix.png、PR_curve.png 等。
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
