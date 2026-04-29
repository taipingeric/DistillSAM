from typing import Dict

import torch


def _mean_valid(values):
    valid = [value for value in values if value == value]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))


def compute_multiclass_iou_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> Dict[str, object]:
    pred = pred.detach().cpu().long()
    target = target.detach().cpu().long()
    valid = target != ignore_index

    per_class_iou = []
    per_class_dice = []
    skipped_classes = []
    for class_id in range(num_classes):
        pred_c = (pred == class_id) & valid
        target_c = (target == class_id) & valid
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        denom = pred_c.sum().item() + target_c.sum().item()
        if union == 0 and denom == 0:
            per_class_iou.append(float("nan"))
            per_class_dice.append(float("nan"))
            skipped_classes.append(class_id)
        else:
            per_class_iou.append(float(intersection / union) if union > 0 else 0.0)
            per_class_dice.append(float(2.0 * intersection / denom) if denom > 0 else 0.0)

    return {
        "per_class_iou": per_class_iou,
        "per_class_dice": per_class_dice,
        "mIoU_all": _mean_valid(per_class_iou),
        "mDice_all": _mean_valid(per_class_dice),
        "mIoU_fg": _mean_valid(per_class_iou[1:]),
        "mDice_fg": _mean_valid(per_class_dice[1:]),
        "skipped_classes": skipped_classes,
    }
