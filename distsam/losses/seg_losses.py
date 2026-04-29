from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        if valid_mask is None:
            valid_mask = torch.ones_like(target)
        else:
            valid_mask = valid_mask.float()

        probs = probs * valid_mask
        target = target * valid_mask
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * target).sum(dim=dims)
        denominator = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth + self.eps)
        return 1.0 - dice.mean()


def bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    target = target.float()
    if valid_mask is None:
        valid_mask = torch.ones_like(target)
    else:
        valid_mask = valid_mask.float()

    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    valid_pixels = valid_mask.sum().clamp_min(1.0)
    bce = (bce_map * valid_mask).sum() / valid_pixels
    dice = BinaryDiceLoss()(logits, target, valid_mask)
    loss = bce + dice
    return loss, {"bce": bce.detach(), "dice": dice.detach()}


def semantic_ce_loss(
    semantic_logits: torch.Tensor,
    semantic_mask: torch.Tensor,
    ignore_index: int = 255,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    target = semantic_mask.long()
    if valid_mask is not None:
        valid = valid_mask[:, 0] > 0.5 if valid_mask.ndim == 4 else valid_mask > 0.5
        target = target.clone()
        target[~valid] = ignore_index
    loss = F.cross_entropy(semantic_logits, target, ignore_index=ignore_index)
    return {"loss": loss, "loss_ce": loss.detach()}
