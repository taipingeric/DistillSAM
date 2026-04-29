import torch
from torch import nn


class LightCrossAttentionAdapterBlock(nn.Module):
    """Two-way cross-attention block between adapter tokens and frozen SAM tokens."""

    def __init__(
        self,
        embed_dim: int = 256,
        sam_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sam_proj = nn.Linear(sam_dim, embed_dim)

        self.c1 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.c2 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.adapter_norm1 = nn.LayerNorm(embed_dim)
        self.sam_norm = nn.LayerNorm(embed_dim)
        self.adapter_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, adapter_tokens: torch.Tensor, sam_tokens_768: torch.Tensor):
        sam_tokens_256 = self.sam_proj(sam_tokens_768)

        adapter_delta, _ = self.c1(
            query=adapter_tokens,
            key=sam_tokens_256,
            value=sam_tokens_256,
            need_weights=False,
        )
        adapter_tokens = self.adapter_norm1(adapter_tokens + adapter_delta)

        sam_delta, _ = self.c2(
            query=sam_tokens_256,
            key=adapter_tokens,
            value=adapter_tokens,
            need_weights=False,
        )
        updated_sam_tokens_256 = self.sam_norm(sam_tokens_256 + sam_delta)

        adapter_tokens = self.adapter_norm2(adapter_tokens + self.ffn(adapter_tokens))
        return adapter_tokens, updated_sam_tokens_256


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)
