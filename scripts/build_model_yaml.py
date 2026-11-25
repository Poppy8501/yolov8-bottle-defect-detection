#!/usr/bin/env python3
"""Create a local YOLOv8n config with the correct class count.

CBAM is applied by ``src.modules.apply_cbam_patch()`` at runtime rather than by
renaming modules in YAML. This keeps Ultralytics' channel/depth scaling logic
fully intact across versions.

Example:
    python scripts/build_model_yaml.py --out configs/yolov8n_cbam.yaml --nc 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def find_base_yaml(variant: str = "n") -> Path:
    """Locate the official YOLOv8 architecture YAML shipped with Ultralytics."""
    try:
        import ultralytics
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ultralytics must be installed to build the model YAML") from exc

    cfg_dir = Path(ultralytics.__file__).parent / "cfg" / "models" / "v8"
    candidates = [cfg_dir / f"yolov8{variant}.yaml", cfg_dir / "yolov8.yaml"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find official YOLOv8 YAML in {cfg_dir}")


def build_cbam_yaml(
    base_yaml: Path,
    out_yaml: Path,
    nc: int = 3,
    variant: str = "n",
) -> dict:
    """Write a local copy of the official architecture with ``nc`` overridden."""
    cfg = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))

    if base_yaml.stem == "yolov8":
        scale = {"n": (0.33, 0.25), "s": (0.33, 0.5), "m": (0.67, 0.75)}
        if variant in scale:
            cfg["depth_multiple"], cfg["width_multiple"] = scale[variant]
    cfg["nc"] = nc

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="n", choices=["n", "s", "m"])
    parser.add_argument("--base-yaml", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("configs/yolov8n_cbam.yaml"))
    parser.add_argument("--nc", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    base = args.base_yaml or find_base_yaml(args.variant)
    cfg = build_cbam_yaml(base, args.out, args.nc, args.variant)
    c2f_count = sum(
        1 for section in ("backbone", "head") for layer in cfg[section]
        if layer[2] == "C2f"
    )
    print(f"Wrote {args.out}: nc={args.nc}, {c2f_count} C2f blocks (CBAM via runtime patch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
