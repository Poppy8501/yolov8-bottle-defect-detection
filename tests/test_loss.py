"""Tests for the WIoU loss (skipped when torch/ultralytics unavailable)."""

from __future__ import annotations

import sys
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
    from src.loss import apply_wiou_patch, wiou_bbox_iou, wiou_quality


@unittest.skipUnless(torch is not None, "torch/ultralytics not installed")
class WIoULossTest(unittest.TestCase):
    def test_identical_boxes_have_perfect_quality(self) -> None:
        box = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
        quality = wiou_quality(box, box.clone())
        self.assertAlmostEqual(float(quality[0]), 1.0, places=5)

    def test_worse_overlap_gives_lower_quality(self) -> None:
        target = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
        partial = torch.tensor([[20.0, 20.0, 40.0, 40.0]])
        disjoint = torch.tensor([[90.0, 90.0, 110.0, 110.0]])
        q_partial = float(wiou_quality(partial, target)[0])
        q_disjoint = float(wiou_quality(disjoint, target)[0])
        self.assertGreater(q_partial, q_disjoint)
        self.assertLess(q_disjoint, 1.0)

    def test_bbox_iou_compat_shape(self) -> None:
        box1 = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
        box2 = torch.tensor([[20.0, 20.0, 40.0, 40.0]])
        quality = wiou_bbox_iou(box1, box2, xywh=False, CIoU=True)
        self.assertEqual(tuple(quality.shape), (1, 1))

    def test_apply_patch(self) -> None:
        import ultralytics.utils.loss as loss_module

        apply_wiou_patch()
        self.assertIs(loss_module.bbox_iou, wiou_bbox_iou)


if __name__ == "__main__":
    unittest.main()
