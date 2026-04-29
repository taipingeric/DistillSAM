from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .adapter_blocks import LightCrossAttentionAdapterBlock, count_trainable_parameters
from .pvt_adapter_encoder import PVTAdapterEncoder


class DepthwisePromptHead(nn.Sequential):
    def __init__(self, channels: int = 256) -> None:
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )


class TrainableAdapterPVTCross(nn.Module):
    """PVT-tiny adapter with paper-style two-way cross-attention adapter blocks."""

    def __init__(
        self,
        pvt_pretrained: Optional[str] = None,
        num_prompt_tokens: int = 16,
        adapter_layers: List[int] = None,
        embed_dim: int = 256,
        num_heads: int = 8,
        token_stride: int = 1,
        pvt_trainable: bool = True,
        debug_shapes: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_prompt_tokens = int(num_prompt_tokens)
        self.adapter_layers = adapter_layers or [4, 8, 12]
        self.token_stride = int(token_stride)
        if self.token_stride < 1:
            raise ValueError(f"token_stride must be >= 1, got {token_stride}.")
        self.debug_shapes = bool(debug_shapes)
        self._printed_shapes = False

        self.encoder = PVTAdapterEncoder(
            pretrained=pvt_pretrained,
            trainable=pvt_trainable,
            verbose=self.debug_shapes,
        )
        self.proj_f2 = nn.Conv2d(128, embed_dim, kernel_size=1)
        self.proj_f3 = nn.Conv2d(320, embed_dim, kernel_size=1)
        self.proj_f4 = nn.Conv2d(512, embed_dim, kernel_size=1)

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )

        self.adapter_blocks = nn.ModuleList(
            [
                LightCrossAttentionAdapterBlock(embed_dim=embed_dim, num_heads=num_heads)
                for _ in self.adapter_layers
            ]
        )

        self.dense_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.dense_head = DepthwisePromptHead(embed_dim)
        self.prompt_queries = nn.Parameter(torch.zeros(1, self.num_prompt_tokens, embed_dim))
        nn.init.normal_(self.prompt_queries, mean=0.0, std=0.02)
        self.prompt_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.prompt_norm = nn.LayerNorm(embed_dim)

        if self.debug_shapes:
            self._print_parameter_summary()

    def _print_parameter_summary(self) -> None:
        pvt_params = count_trainable_parameters(self.encoder)
        total_params = count_trainable_parameters(self)
        non_pvt_params = total_params - pvt_params
        print(f"[TrainableAdapterPVTCross] PVT trainable params: {pvt_params}")
        print(f"[TrainableAdapterPVTCross] non-PVT adapter trainable params: {non_pvt_params}")
        for index, block in zip(self.adapter_layers, self.adapter_blocks):
            print(
                f"[TrainableAdapterPVTCross] adapter block layer {index} params: "
                f"{count_trainable_parameters(block)}"
            )
        print(f"[TrainableAdapterPVTCross] total adapter trainable params: {total_params}")
        if total_params > 16_000_000:
            print(
                "[TrainableAdapterPVTCross][WARNING] total trainable params exceed 16M; "
                "consider --token-stride 2 or fewer adapter layers if memory is tight."
            )

    def _flatten_tokens(self, feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(2).transpose(1, 2).contiguous()

    def forward(
        self,
        image: torch.Tensor,
        sam_image_embedding: torch.Tensor,
        sam_intermediates: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        f2, f3, f4 = self.encoder(image)
        target_size = sam_image_embedding.shape[-2:]

        f2_64 = F.interpolate(
            self.proj_f2(f2),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        f3_64 = self.proj_f3(f3)
        if f3_64.shape[-2:] != target_size:
            f3_64 = F.interpolate(
                f3_64,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        f4_64 = F.interpolate(
            self.proj_f4(f4),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        pvt_fused = self.fuse(torch.cat([f2_64, f3_64, f4_64], dim=1))
        token_feature = pvt_fused
        if self.token_stride > 1:
            token_feature = F.avg_pool2d(
                token_feature,
                kernel_size=self.token_stride,
                stride=self.token_stride,
            )
        adapter_tokens = self._flatten_tokens(token_feature)

        updated_sam_tokens = None
        updated_sam_tokens_list = []
        for layer, block in zip(self.adapter_layers, self.adapter_blocks):
            if layer not in sam_intermediates:
                raise KeyError(f"Missing SAM intermediate for adapter layer {layer}.")
            sam_feat = sam_intermediates[layer]
            sam_tokens = sam_feat.reshape(sam_feat.shape[0], -1, sam_feat.shape[-1])
            adapter_tokens, updated_sam_tokens = block(adapter_tokens, sam_tokens)
            updated_sam_tokens_list.append(updated_sam_tokens)
            if self.debug_shapes and not self._printed_shapes:
                print(
                    f"[shape] adapter_tokens after block {layer}: "
                    f"{tuple(adapter_tokens.shape)}"
                )

        if updated_sam_tokens is None:
            raise RuntimeError("No adapter blocks were executed.")

        updated_sam_tokens = torch.stack(updated_sam_tokens_list, dim=0).mean(dim=0)
        b, _, _ = updated_sam_tokens.shape
        updated_sam_feature = updated_sam_tokens.transpose(1, 2).reshape(
            b,
            self.embed_dim,
            target_size[0],
            target_size[1],
        )
        dense_base = sam_image_embedding + updated_sam_feature
        dense_prompt_embeddings = self.dense_scale * self.dense_head(dense_base)

        prompt_queries = self.prompt_queries.expand(image.shape[0], -1, -1)
        sparse_prompt_embeddings, _ = self.prompt_attn(
            query=prompt_queries,
            key=adapter_tokens,
            value=adapter_tokens,
            need_weights=False,
        )
        sparse_prompt_embeddings = self.prompt_norm(prompt_queries + sparse_prompt_embeddings)

        if self.debug_shapes and not self._printed_shapes:
            print(f"[shape] PVT feature2: {tuple(f2.shape)}")
            print(f"[shape] PVT feature3: {tuple(f3.shape)}")
            print(f"[shape] PVT feature4: {tuple(f4.shape)}")
            print(f"[shape] projected f2: {tuple(f2_64.shape)}")
            print(f"[shape] projected f3: {tuple(f3_64.shape)}")
            print(f"[shape] projected f4: {tuple(f4_64.shape)}")
            print(f"[shape] pvt_fused: {tuple(pvt_fused.shape)}")
            print(f"[shape] adapter_tokens: {tuple(adapter_tokens.shape)}")
            print(f"[shape] final updated_sam_tokens: {tuple(updated_sam_tokens.shape)}")
            print(f"[shape] dense_prompt_embeddings: {tuple(dense_prompt_embeddings.shape)}")
            print(f"[shape] sparse_prompt_embeddings: {tuple(sparse_prompt_embeddings.shape)}")
            self._printed_shapes = True

        return sparse_prompt_embeddings, dense_prompt_embeddings
