"""Tests for MVTec AD -> YOLO dataset conversion using synthetic data."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_dataset import find_mask_path, mask_to_yolo_lines, prepare_dataset  # noqa: E402


def make_tree(root: Path) -> None:
    cats = ["broken_large", "contamination"]
    for name in cats:
        (root / "bottle" / "test" / name).mkdir(parents=True, exist_ok=True)
        (root / "bottle" / "ground_truth" / name).mkdir(parents=True, exist_ok=True)
        for i in range(5):
            image = np.full((50, 50, 3), 255, dtype="uint8")
            cv2.imwrite(str(root / "bottle" / "test" / name / f"{name}_{i}.png"), image)
            mask = np.zeros((50, 50), dtype="uint8")
            mask[10:20, 10:20] = 255
            cv2.imwrite(
                str(root / "bottle" / "ground_truth" / name / f"{name}_{i}_mask.png"), mask
            )
    for split in ("train", "test"):
        folder = root / "bottle" / split / "good"
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            image = np.full((50, 50, 3), 255, dtype="uint8")
            cv2.imwrite(str(folder / f"good_{split}_{i}.png"), image)


class PrepareDatasetTest(unittest.TestCase):
    def test_find_mask_path_supports_both_namings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "ground_truth" / "broken_large"
            base.mkdir(parents=True)
            (base / "001_mask.png").write_bytes(b"x")
            self.assertEqual(find_mask_path(root, "broken_large", "001").name, "001_mask.png")
            (base / "002.png").write_bytes(b"x")
            self.assertEqual(find_mask_path(root, "broken_large", "002").name, "002.png")

    def test_mask_to_yolo_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mask_path = Path(tmp) / "mask.png"
            mask = np.zeros((50, 50), dtype="uint8")
            mask[10:20, 10:20] = 255
            cv2.imwrite(str(mask_path), mask)
            lines = mask_to_yolo_lines(mask_path, class_id=1)
            self.assertEqual(len(lines), 1)
            parts = lines[0].split()
            self.assertEqual(int(parts[0]), 1)
            np.testing.assert_allclose(
                [float(v) for v in parts[1:]], [0.3, 0.3, 0.2, 0.2], atol=1e-5
            )

    def test_prepare_dataset_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_tree(root)
            out = root / "yolo_bottle"
            stats = prepare_dataset(
                mvtec_root=root,
                category="bottle",
                out_dir=out,
                val_ratio=0.4,
                seed=7,
                neg_train=2,
            )

            self.assertEqual(stats["classes"], ["broken_large", "contamination"])
            self.assertEqual(stats["train_counts"]["negative"], 2)
            self.assertEqual(stats["val_counts"]["negative"], 6)
            for cls in stats["classes"]:
                self.assertGreaterEqual(stats["train_counts"][cls], 1)
                self.assertGreaterEqual(stats["val_counts"][cls], 1)

            cfg = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
            self.assertEqual(cfg["nc"], 2)
            self.assertEqual(cfg["names"][0], "broken_large")
            self.assertTrue((out / "images" / "train").is_dir())
            self.assertTrue((out / "labels" / "val").is_dir())

            label_files = sorted((out / "labels" / "train").glob("*.txt"))
            class_ids = set()
            for label in label_files:
                text = label.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                for line in text.splitlines():
                    class_ids.add(int(line.split()[0]))
            self.assertEqual(class_ids, {0, 1})

            negative_labels = [
                p for p in (out / "labels" / "train").glob("*.txt")
                if not p.read_text(encoding="utf-8").strip()
            ]
            self.assertEqual(len(negative_labels), 2)


if __name__ == "__main__":
    unittest.main()
