import torch
import torch.nn.functional as F


def dense_prompt_kd_loss(
    student_dense_prompt_embeddings: torch.Tensor,
    teacher_dense_prompt_embeddings: torch.Tensor,
    loss_type: str = "mse",
    cos_weight: float = 0.0,
    fg_weight_mask=None,
):
    if loss_type not in {"mse", "normalized_mse", "cosine", "mse_cos"}:
        raise ValueError(
            "DistillSAM dense KD supports loss_type in "
            "{'mse', 'normalized_mse', 'cosine', 'mse_cos'}, "
            f"got {loss_type!r}."
        )
    if fg_weight_mask is not None:
        raise NotImplementedError("fg_weight_mask is reserved for later ablations and is not enabled yet.")

    teacher = teacher_dense_prompt_embeddings.detach()
    mse = F.mse_loss(student_dense_prompt_embeddings, teacher)
    student_n = F.normalize(student_dense_prompt_embeddings, dim=1)
    teacher_n = F.normalize(teacher, dim=1)
    normalized_mse = F.mse_loss(student_n, teacher_n)
    cosine_loss = 1.0 - F.cosine_similarity(
        student_dense_prompt_embeddings,
        teacher,
        dim=1,
    ).mean()

    if loss_type == "mse":
        total = mse
    elif loss_type == "normalized_mse":
        total = normalized_mse
    elif loss_type == "cosine":
        total = cosine_loss
    else:
        total = mse + float(cos_weight) * cosine_loss
    return {
        "loss_dense_kd": total,
        "loss_dense_kd_mse": mse,
        "loss_dense_kd_normalized_mse": normalized_mse,
        "loss_dense_kd_cos": cosine_loss,
    }
