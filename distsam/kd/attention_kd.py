from typing import Iterable, Optional

import torch
import torch.nn.functional as F


def aggregate_mask_token_attention(
    attention: torch.Tensor,
    num_mask_tokens: int,
    output_size: int = 64,
) -> torch.Tensor:
    if attention.ndim != 4:
        raise ValueError(f"attention must be BxHeadsxTokensxHW, got {tuple(attention.shape)}.")
    mask_attention = attention[:, :, 1 : 1 + int(num_mask_tokens), :]
    attention_map = mask_attention.mean(dim=1).mean(dim=1)
    return attention_map.view(attention.shape[0], 1, int(output_size), int(output_size))


def minmax_normalize_attention(attention_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = attention_map.flatten(1)
    min_value = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    max_value = flat.max(dim=1)[0].view(-1, 1, 1, 1)
    return (attention_map - min_value) / (max_value - min_value + float(eps))


def _foreground_mask_from_semantic(
    semantic_mask: torch.Tensor,
    foreground_classes: Optional[Iterable[int]],
    num_classes: int,
    background_id: int,
    ignore_index: int,
) -> torch.Tensor:
    if foreground_classes is None:
        classes = [
            class_id
            for class_id in range(int(num_classes))
            if class_id != int(background_id) and class_id != int(ignore_index)
        ]
    else:
        classes = [int(class_id) for class_id in foreground_classes]
    foreground = torch.zeros_like(semantic_mask, dtype=torch.bool)
    for class_id in classes:
        foreground |= semantic_mask == int(class_id)
    foreground &= semantic_mask != int(ignore_index)
    return foreground


def binarize_teacher_attention(
    teacher_attention_map: torch.Tensor,
    semantic_mask: torch.Tensor,
    foreground_classes: Optional[Iterable[int]],
    num_classes: int,
    background_id: int,
    ignore_index: int,
    mode: str = "topk_gt_area",
    topk_ratio: float = 0.2,
    threshold: float = 0.5,
    min_ratio: float = 0.01,
    max_ratio: float = 0.80,
) -> tuple:
    if mode not in {"topk_gt_area", "topk_ratio", "threshold"}:
        raise ValueError(
            "attention binarize mode must be 'topk_gt_area', 'topk_ratio', or 'threshold', "
            f"got {mode!r}."
        )
    batch_size = teacher_attention_map.shape[0]
    foreground = _foreground_mask_from_semantic(
        semantic_mask=semantic_mask,
        foreground_classes=foreground_classes,
        num_classes=num_classes,
        background_id=background_id,
        ignore_index=ignore_index,
    ).float().unsqueeze(1)
    valid = (semantic_mask != int(ignore_index)).float().unsqueeze(1)
    foreground_64 = F.interpolate(foreground, size=teacher_attention_map.shape[-2:], mode="nearest") > 0.5
    valid_64 = F.interpolate(valid, size=teacher_attention_map.shape[-2:], mode="nearest") > 0.5
    valid_counts = valid_64.flatten(1).sum(dim=1).clamp_min(1.0)
    fg_ratio = (foreground_64.flatten(1).sum(dim=1).float() / valid_counts).clamp(
        min=float(min_ratio),
        max=float(max_ratio),
    )

    if mode == "threshold":
        binary = teacher_attention_map >= float(threshold)
    else:
        ratio = torch.full_like(fg_ratio, float(topk_ratio)) if mode == "topk_ratio" else fg_ratio
        masked_attention = teacher_attention_map.masked_fill(~valid_64, float("-inf"))
        flat = masked_attention.flatten(1)
        binary_flat = torch.zeros_like(flat, dtype=torch.bool)
        for index in range(batch_size):
            k = int(round(float(ratio[index].item()) * float(valid_counts[index].item())))
            k = max(1, min(k, int(valid_counts[index].item())))
            top_indices = torch.topk(flat[index], k=k, dim=0).indices
            binary_flat[index, top_indices] = True
        binary = binary_flat.view_as(teacher_attention_map)
    binary = binary & valid_64
    actual_ratio = binary.flatten(1).float().sum(dim=1) / valid_counts
    return binary.float(), actual_ratio.mean()


def binary_attention_map_kd_loss(
    student_attention_map: torch.Tensor,
    teacher_binary_attention_mask: torch.Tensor,
    loss_type: str = "bce_dice",
    dice_weight: float = 1.0,
    eps: float = 1e-6,
):
    if loss_type not in {"bce", "dice", "bce_dice", "mse_binary"}:
        raise ValueError(
            "attention KD loss_type must be 'bce', 'dice', 'bce_dice', or 'mse_binary', "
            f"got {loss_type!r}."
        )
    target = teacher_binary_attention_mask.detach().float()
    student = student_attention_map.clamp(min=float(eps), max=1.0 - float(eps))
    bce = F.binary_cross_entropy(student, target)
    dims = tuple(range(1, student.ndim))
    intersection = (student * target).sum(dim=dims)
    denominator = student.sum(dim=dims) + target.sum(dim=dims)
    dice = 1.0 - ((2.0 * intersection + float(eps)) / (denominator + float(eps))).mean()
    mse = F.mse_loss(student, target)

    if loss_type == "bce":
        total = bce
    elif loss_type == "dice":
        total = dice
    elif loss_type == "mse_binary":
        total = mse
    else:
        total = bce + float(dice_weight) * dice
    return {
        "loss_attention_kd": total,
        "loss_attention_bce": bce if loss_type in {"bce", "bce_dice"} else student.new_tensor(0.0),
        "loss_attention_dice": dice if loss_type in {"dice", "bce_dice"} else student.new_tensor(0.0),
        "loss_attention_mse": mse if loss_type == "mse_binary" else student.new_tensor(0.0),
        "teacher_binary_fg_ratio": target.flatten(1).mean(dim=1).mean().detach(),
    }
