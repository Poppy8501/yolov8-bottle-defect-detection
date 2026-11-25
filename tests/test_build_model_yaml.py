"""Tests for local model YAML generation from an official base architecture."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_model_yaml import build_cbam_yaml  # noqa: E402


class BuildModelYamlTest(unittest.TestCase):
    def test_keeps_official_architecture_and_sets_nc(self) -> None:
        base_cfg = {
            "nc": 80,
            "depth_multiple": 0.33,
            "width_multiple": 0.25,
            "backbone": [
                [-1, 1, "Conv", [64, 3, 2]],
                [-1, 3, "C2f", [64, True]],
                [-1, 1, "Conv", [128, 3, 2]],
                [-1, 6, "C2f", [128, True]],
            ],
            "head": [
                [-1, 1, "Upsample", [2, "nearest"]],
                [-1, 2, "C2f", [128]],
                [-1, 2, "C2f", [128]],
                [-1, 1, "Detect", [0, 1]],
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.yaml"
            out = Path(tmp) / "out.yaml"
            base.write_text(yaml.safe_dump(base_cfg), encoding="utf-8")
            cfg = build_cbam_yaml(base, out, nc=3)

            self.assertEqual(cfg["nc"], 3)
            self.assertEqual(
                [layer[2] for layer in cfg["backbone"] + cfg["head"]].count("C2f"), 4
            )
            reloaded = yaml.safe_load(out.read_text(encoding="utf-8"))
            self.assertEqual(reloaded["nc"], 3)
            self.assertEqual(reloaded["backbone"][1][2], "C2f")


if __name__ == "__main__":
    unittest.main()
