from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

from .pvt_tiny_detection import pvt_tiny as _build_pvt_tiny

# Originally built via mmdet.models.backbones.pvt.PyramidVisionTransformer,
# which requires mmdet + mmcv. mmcv's full build needs a complete CUDA
# Toolkit (nvcc) to compile its CUDA extensions -- not present in this
# environment (only PyTorch's bundled CUDA runtime is). mmcv-lite (no
# compiled ops) installs but conflicts with mmdet's pinned version range,
# and mmcv 2.x also moved ConfigDict out of the top-level mmcv namespace.
# None of this is pinned in DistillSAM's own repo, so there's no
# authoritative version combination to chase. Switched to a standalone
# PVT-tiny (whai362/PVT's own detection/pvt.py, vendored at
# pvt_tiny_detection.py with mmdet/mmcv stripped out, architecture
# otherwise verbatim) -- this is the exact code the pvt_tiny.pth
# checkpoint was itself trained with, so no vendor/mmdet reimplementation
# gap to worry about.


class PVTAdapterEncoder(nn.Module):
    """PVT-tiny encoder used by the DistillSAM adapter branch."""

    def __init__(
        self,
        pretrained: Optional[str] = None,
        out_indices: Tuple[int, int, int] = (1, 2, 3),
        trainable: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.out_indices = out_indices
        pretrained = str(pretrained) if pretrained else ""

        self.pvt = _build_pvt_tiny()

        if pretrained:
            checkpoint_path = Path(pretrained)
            if not checkpoint_path.is_absolute():
                checkpoint_path = Path(__file__).resolve().parents[2] / checkpoint_path
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"PVT-tiny pretrained checkpoint does not exist: {checkpoint_path}"
                )
            # strict=False: the checkpoint is whai362/PVT's *classification*
            # export (has cls_token/head.* with no counterpart here); every
            # key this detection-style backbone actually owns (pos_embed1-4,
            # patch_embed1-4.*, block1-4.*) matches exactly.
            state_dict = torch.load(str(checkpoint_path), map_location="cpu")
            missing, unexpected = self.pvt.load_state_dict(state_dict, strict=False)
            expected_unexpected = {"cls_token", "head.weight", "head.bias", "norm.weight", "norm.bias"}
            surprising_unexpected = set(unexpected) - expected_unexpected
            if missing or surprising_unexpected:
                raise RuntimeError(
                    f"PVT-tiny checkpoint load looks wrong: missing={missing} "
                    f"unexpected={sorted(surprising_unexpected)}"
                )

        if verbose:
            if pretrained:
                print(f"[PVTAdapterEncoder] Loaded pretrained PVT-tiny from: {pretrained}")
            else:
                print(
                    "[PVTAdapterEncoder] No pretrained checkpoint provided. "
                    "Using random initialization."
                )

        if not trainable:
            self.pvt.eval()
            for param in self.pvt.parameters():
                param.requires_grad_(False)

    def forward(self, image):
        all_stages = self.pvt(image)
        outputs = [all_stages[i] for i in self.out_indices]
        if len(outputs) != 3:
            raise RuntimeError(f"PVTAdapterEncoder expected 3 outputs, got {len(outputs)}.")
        return outputs
