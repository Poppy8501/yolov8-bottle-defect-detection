"""Improvement 2: WIoU-based bounding-box regression loss for YOLOv8.

Rather than subclassing Ultralytics' BboxLoss (whose internals changed across
versions), this module replaces the ``bbox_iou`` function *inside the loss
module* with a WIoU-compatible quality function. Every YOLOv8 BboxLoss version
computes ``(1 - iou) * weight``, so the replacement is version-agnostic.

Usage:
    from src.loss import apply_wiou_patch
    apply_wiou_patch()   # call BEFORE creating the YOLO trainer
"""

from __future__ import annotations

import torch

_EPS = 1e-7


def _to_xyxy(boxes: torch.Tensor, xywh: bool) -> torch.Tensor:
    if not xywh:
        return boxes
    cx, cy, w, h = boxes.unbind(dim=-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)


def wiou_quality(pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor) -> torch.Tensor:
    """Return WIoU 'quality' in the same form Ultralytics expects for IoU.

    Args:
        pred_bboxes: Predicted boxes in xyxy format, shape (N, 4).
        target_bboxes: Ground-truth boxes in xyxy format, shape (N, 4).

    Returns:
        Tensor of shape (N,) or (N, M). Values may be negative for very poor
        matches; Ultralytics computes ``(1 - quality) * weight`` so bad
        predictions still receive a positive, harder penalty.
    """
    if pred_bboxes.shape != target_bboxes.shape:
        pred_bboxes = pred_bboxes.unsqueeze(-2)
        target_bboxes = target_bboxes.unsqueeze(-3)

    px1, py1, px2, py2 = pred_bboxes.unbind(dim=-1)
    gx1, gy1, gx2, gy2 = target_bboxes.unbind(dim=-1)

    inter_w = (torch.minimum(px2, gx2) - torch.maximum(px1, gx1)).clamp(min=0)
    inter_h = (torch.minimum(py2, gy2) - torch.maximum(py1, gy1)).clamp(min=0)
    inter = inter_w * inter_h

    pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    target_area = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union = pred_area + target_area - inter + _EPS
    iou = inter / union

    enclosing_w = torch.maximum(px2, gx2) - torch.minimum(px1, gx1)
    enclosing_h = torch.maximum(py2, gy2) - torch.minimum(py1, gy1)
    centre_dist_sq = (((px1 + px2) - (gx1 + gx2)) ** 2) / 4
    centre_dist_sq = centre_dist_sq + (((py1 + py2) - (gy1 + gy2)) ** 2) / 4
    rho = torch.exp(
        centre_dist_sq / (enclosing_w.square() + enclosing_h.square() + _EPS)
    )

    return 1.0 - (1.0 - iou) * rho


def wiou_bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = _EPS,
) -> torch.Tensor:
    """Drop-in replacement for ``ultralytics.utils.loss.bbox_iou``."""
    quality = wiou_quality(_to_xyxy(box1, xywh), _to_xyxy(box2, xywh))
    if quality.ndim == 1:
        quality = quality.unsqueeze(-1)
    return quality


def apply_wiou_patch() -> None:
    """Replace the IoU function used by Ultralytics' detection loss."""
    import ultralytics.utils.loss as loss_module

    if getattr(loss_module, "_wiou_patched", False):
        return
    loss_module.bbox_iou = wiou_bbox_iou
    loss_module._wiou_patched = True
