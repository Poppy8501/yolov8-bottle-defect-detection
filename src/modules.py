"""Improvement 1: inject CBAM attention into YOLOv8 C2f blocks.

Two ways are provided:

* ``apply_cbam_patch()`` (recommended, used by train.py): monkey-patches the
  official ``C2f`` class in place. Ultralytics' YAML parser recognises ``C2f``
  natively, so channel/repeat scaling keeps working across versions.
* ``C2fCBAM`` + ``register_custom_modules()``: an explicit custom module for
  users who prefer editing the YAML by hand.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn as nn

try:  # only required when actually training/inferencing
    from ultralytics.nn.modules import Bottleneck, Conv
except ImportError:  # pragma: no cover - allows importing package without ultralytics
    Bottleneck = Any  # type: ignore[assignment,misc]
    Conv = Any  # type: ignore[assignment,misc]

from .attention import CBAM


class C2fCBAM(nn.Module):
    """C2f block with channel + spatial attention on its fused output."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        bottleneck_kwargs: dict[str, Any] = {"shortcut": shortcut, "g": g, "e": 1.0}
        # Ultralytics >=8.1 exposes an extra `k` argument on Bottleneck.
        if "k" in inspect.signature(Bottleneck.__init__).parameters:
            bottleneck_kwargs["k"] = (3, 3)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, **bottleneck_kwargs) for _ in range(n)
        )
        self.attention = CBAM(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), dim=1))
        y.extend(block(y[-1]) for block in self.m)
        return self.attention(self.cv2(torch.cat(y, dim=1)))


def register_custom_modules() -> None:
    """Make ``C2fCBAM`` resolvable by the Ultralytics YAML parser."""
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    setattr(modules, "C2fCBAM", C2fCBAM)
    setattr(modules, "CBAM", CBAM)
    tasks.__dict__["C2fCBAM"] = C2fCBAM
    tasks.__dict__["CBAM"] = CBAM


def apply_cbam_patch() -> None:
    """Wrap every official C2f block with CBAM, keeping YAML parsing intact.

    Call this BEFORE ``YOLO(model_yaml)``. Repeating the call is harmless.
    """
    import ultralytics.nn.modules as modules

    c2f_class = modules.C2f
    if getattr(c2f_class, "_cbam_patched", False):
        return

    original_init = c2f_class.__init__
    original_forward = c2f_class.forward

    def patched_init(self: nn.Module, c1: int, c2: int, *args: Any, **kwargs: Any) -> None:
        original_init(self, c1, c2, *args, **kwargs)
        self.attention = CBAM(c2)  # type: ignore[attr-defined]

    def patched_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        output = original_forward(self, x)
        return self.attention(output)  # type: ignore[attr-defined]

    c2f_class.__init__ = patched_init
    c2f_class.forward = patched_forward
    c2f_class._cbam_patched = True
