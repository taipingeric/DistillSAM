import torch
import torch.nn.functional as F


def sparse_mask_token_kd_loss(
    student_mask_tokens_out: torch.Tensor,
    teacher_mask_tokens_out: torch.Tensor,
    loss_type: str = "mse",
    cos_weight: float = 0.0,
):
    if loss_type != "mse":
        raise ValueError(f"DistillSAM sparse KD only supports loss_type='mse', got {loss_type!r}.")
    teacher = teacher_mask_tokens_out.detach()
    mse = F.mse_loss(student_mask_tokens_out, teacher)
    if cos_weight > 0:
        cosine_loss = 1.0 - F.cosine_similarity(student_mask_tokens_out, teacher, dim=-1).mean()
    else:
        cosine_loss = student_mask_tokens_out.new_tensor(0.0)
    total = mse + float(cos_weight) * cosine_loss
    return {
        "loss_sparse_kd": total,
        "loss_sparse_kd_mse": mse,
        "loss_sparse_kd_cos": cosine_loss,
    }
