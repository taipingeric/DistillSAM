from typing import Dict, List

import torch
from torch import nn


class SamImageEncoderIntermediate(nn.Module):
    """Frozen SAM image encoder wrapper that exposes selected block outputs."""

    def __init__(self, sam_image_encoder: nn.Module, adapter_layers: List[int] = None) -> None:
        super().__init__()
        self.image_encoder = sam_image_encoder
        self.adapter_layers = adapter_layers or [4, 8, 12]
        self._validate_adapter_layers()
        self.image_encoder.eval()
        for param in self.image_encoder.parameters():
            param.requires_grad_(False)

    def _validate_adapter_layers(self) -> None:
        depth = len(self.image_encoder.blocks)
        invalid = [layer for layer in self.adapter_layers if layer < 1 or layer > depth]
        if invalid:
            raise ValueError(
                f"adapter_layers must be within [1, {depth}], got invalid layers {invalid}."
            )

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> Dict[str, object]:
        x = self.image_encoder.patch_embed(image)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed

        intermediates = {}
        adapter_layer_set = set(self.adapter_layers)
        for index, block in enumerate(self.image_encoder.blocks, start=1):
            x = block(x)
            if index in adapter_layer_set:
                intermediates[index] = x.detach()

        image_embeddings = self.image_encoder.neck(x.permute(0, 3, 1, 2)).detach()
        return {
            "image_embeddings": image_embeddings,
            "intermediates": intermediates,
        }
