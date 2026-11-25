"""Integration smoke test: build config, patch CBAM, instantiate YOLOv8."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    import ultralytics  # noqa: F401
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:
    from scripts.build_model_yaml import build_cbam_yaml, find_base_yaml
    from src.modules import apply_cbam_patch


@unittest.skipUnless(torch is not None, "ultralytics/torch not installed")
class IntegrationTest(unittest.TestCase):
    def test_model_builds_with_cbam_and_forward_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "yolov8n_cbam.yaml"
            build_cbam_yaml(find_base_yaml("n"), out, nc=3)
            apply_cbam_patch()

            from ultralytics import YOLO

            model = YOLO(str(out), task="detect")
            module_names = [name for name, _ in model.model.named_modules()]
            self.assertTrue(
                any(name.endswith(".attention") for name in module_names),
                "CBAM attention modules were not injected",
            )

            model.model.train()
            with torch.no_grad():
                outputs = model.model(torch.zeros(1, 3, 64, 64))
            if isinstance(outputs, dict):
                self.assertIn("boxes", outputs)
            else:
                self.assertIsInstance(outputs, (tuple, list))
                self.assertGreaterEqual(len(outputs), 1)


if __name__ == "__main__":
    unittest.main()
