from pathlib import Path
from typing import Optional, Tuple

from torch import nn

try:
    from mmcv import ConfigDict
    from mmdet.models.backbones.pvt import PyramidVisionTransformer
except ModuleNotFoundError as exc:
    missing = exc.name
    raise ModuleNotFoundError(
        "PVT adapter requires mmcv and mmdet to be installed in the current environment."
    ) from exc


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
        pretrained = str(pretrained) if pretrained else ""
        init_cfg = None
        if pretrained:
            checkpoint_path = Path(pretrained)
            if not checkpoint_path.is_absolute():
                checkpoint_path = Path(__file__).resolve().parents[2] / checkpoint_path
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"PVT-tiny pretrained checkpoint does not exist: {checkpoint_path}"
                )
            init_cfg = ConfigDict(type="Pretrained", checkpoint=str(checkpoint_path))

        self.pvt = PyramidVisionTransformer(
            pretrain_img_size=224,
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[4, 2, 2, 2],
            strides=[4, 2, 2, 2],
            paddings=[0, 0, 0, 0],
            sr_ratios=[8, 4, 2, 1],
            out_indices=out_indices,
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            init_cfg=init_cfg,
        )
        self.pvt.init_weights()

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
        outputs = self.pvt(image)
        if len(outputs) != 3:
            raise RuntimeError(f"PVTAdapterEncoder expected 3 outputs, got {len(outputs)}.")
        return outputs
