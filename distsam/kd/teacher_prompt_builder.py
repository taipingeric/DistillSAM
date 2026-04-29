import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from PIL import Image

from distsam.prompting import sample_foreground_prompt


REPO_ROOT = Path(__file__).resolve().parents[2]
SEGMENT_ANYTHING_ROOT = REPO_ROOT / "segment-anything"
if str(SEGMENT_ANYTHING_ROOT) not in sys.path:
    sys.path.insert(0, str(SEGMENT_ANYTHING_ROOT))

from segment_anything.utils.transforms import ResizeLongestSide  # noqa: E402


def resolve_teacher_prompt_strategy(config: Dict, dataset_name: str, cli_strategy: Optional[str] = None) -> str:
    if cli_strategy:
        return cli_strategy
    sparse_cfg = config.get("kd", {}).get("sparse_kd", {})
    per_dataset = sparse_cfg.get("per_dataset_teacher_prompt_strategy", {})
    if dataset_name in per_dataset:
        return per_dataset[dataset_name]
    return sparse_cfg.get("teacher_prompt_strategy", "fg_random_pos_16")


def build_teacher_prompts(
    semantic_mask,
    raw_image_shape,
    strategy,
    num_prompt_tokens,
    num_classes,
    foreground_classes,
    background_id,
    ignore_index,
    seed,
):
    del raw_image_shape
    rng = np.random.default_rng(int(seed))
    prompt = sample_foreground_prompt(
        strategy=strategy,
        semantic_mask=semantic_mask,
        num_classes=num_classes,
        foreground_classes=foreground_classes,
        background_id=background_id,
        ignore_index=ignore_index,
        total_budget=num_prompt_tokens,
        rng=rng,
    )
    if prompt["meta"].get("skip_reason"):
        raise RuntimeError(f"Teacher prompt sampling failed: {prompt['meta']['skip_reason']}")
    return prompt


def _to_original_size_tuple(raw_image_shape):
    return int(raw_image_shape[0]), int(raw_image_shape[1])


def encode_teacher_prompts_with_sam(
    sam_prompt_encoder,
    point_coords,
    point_labels,
    box,
    device,
    raw_image_shape,
    image_size: int = 1024,
):
    transform = ResizeLongestSide(image_size)
    original_size = _to_original_size_tuple(raw_image_shape)

    points = None
    if point_coords is not None:
        transformed_coords = transform.apply_coords(point_coords.astype(np.float32), original_size)
        coords_torch = torch.as_tensor(transformed_coords, dtype=torch.float32, device=device)[None, :, :]
        labels_torch = torch.as_tensor(point_labels.astype(np.int64), dtype=torch.int64, device=device)[None, :]
        points = (coords_torch, labels_torch)

    box_torch = None
    if box is not None:
        transformed_box = transform.apply_boxes(np.asarray(box, dtype=np.float32)[None, :], original_size)
        box_torch = torch.as_tensor(transformed_box, dtype=torch.float32, device=device)

    with torch.no_grad():
        sparse_prompt_embeddings, dense_prompt_embeddings = sam_prompt_encoder(
            points=points,
            boxes=box_torch,
            masks=None,
        )
    return sparse_prompt_embeddings, dense_prompt_embeddings


def infer_foreground_classes(mask: np.ndarray, background_id: int, ignore_index: int, num_classes: Optional[int]):
    if num_classes is not None:
        candidates = list(range(int(num_classes)))
    else:
        candidates = [int(label) for label in np.unique(mask)]
    return [
        class_id
        for class_id in candidates
        if class_id != int(background_id)
        and class_id != int(ignore_index)
        and np.any(mask == class_id)
    ]


def build_gt_foreground_mask_input(
    semantic_mask,
    foreground_classes,
    ignore_index: int = 255,
    target_size: int = 256,
    pos_logit: float = 10.0,
    neg_logit: float = -10.0,
):
    semantic_mask = np.asarray(semantic_mask)
    classes = [int(class_id) for class_id in foreground_classes]
    fg_mask = np.isin(semantic_mask, classes)
    fg_mask[semantic_mask == int(ignore_index)] = False
    resized = np.asarray(
        Image.fromarray(fg_mask.astype(np.uint8, copy=True)).resize(
            (int(target_size), int(target_size)),
            resample=Image.NEAREST,
        )
    ).astype(bool, copy=False)
    mask_input = np.full((1, int(target_size), int(target_size)), float(neg_logit), dtype=np.float32)
    mask_input[0, resized] = float(pos_logit)
    return mask_input


def encode_teacher_dense_prompt_with_sam(
    sam_prompt_encoder,
    mask_input,
    device,
):
    if isinstance(mask_input, np.ndarray):
        masks = torch.as_tensor(mask_input, dtype=torch.float32, device=device)
    else:
        masks = mask_input.to(device=device, dtype=torch.float32)
    if masks.ndim == 3:
        masks = masks.unsqueeze(0)
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(f"mask_input must have shape Bx1xHxW or 1xHxW, got {tuple(masks.shape)}.")

    with torch.no_grad():
        _, dense_prompt_embeddings = sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=masks,
        )
    return dense_prompt_embeddings


def build_gt_foreground_mask_inputs_from_batch(
    model,
    dataset,
    batch,
    num_classes: int,
    foreground_classes: Optional[Iterable[int]],
    background_id: int,
    ignore_index: int,
    pos_logit: float = 10.0,
    neg_logit: float = -10.0,
):
    raw_model = model.module if hasattr(model, "module") else model
    mask_inputs = []
    for mask_path in batch["mask_path"]:
        raw_mask = dataset._read_mask(Path(mask_path))
        semantic_mask = dataset.remap_mask(raw_mask)
        classes = (
            list(foreground_classes)
            if foreground_classes is not None
            else infer_foreground_classes(semantic_mask, background_id, ignore_index, num_classes)
        )
        mask_inputs.append(
            build_gt_foreground_mask_input(
                semantic_mask=semantic_mask,
                foreground_classes=classes,
                ignore_index=ignore_index,
                target_size=raw_model.sam.prompt_encoder.mask_input_size[0],
                pos_logit=pos_logit,
                neg_logit=neg_logit,
            )
        )
    return np.stack(mask_inputs, axis=0)


def teacher_dense_prompts_from_batch(
    model,
    dataset,
    batch,
    num_classes: int,
    foreground_classes: Optional[Iterable[int]],
    background_id: int,
    ignore_index: int,
    device,
    pos_logit: float = 10.0,
    neg_logit: float = -10.0,
):
    raw_model = model.module if hasattr(model, "module") else model
    mask_input = build_gt_foreground_mask_inputs_from_batch(
        model=model,
        dataset=dataset,
        batch=batch,
        num_classes=num_classes,
        foreground_classes=foreground_classes,
        background_id=background_id,
        ignore_index=ignore_index,
        pos_logit=pos_logit,
        neg_logit=neg_logit,
    )
    return encode_teacher_dense_prompt_with_sam(
        sam_prompt_encoder=raw_model.sam.prompt_encoder,
        mask_input=mask_input,
        device=device,
    ).detach()


def teacher_decoder_outputs_from_batch(
    model,
    dataset,
    batch,
    image_embeddings,
    image_pe,
    strategy: str,
    num_prompt_tokens: int,
    num_classes: int,
    foreground_classes: Optional[Iterable[int]],
    background_id: int,
    ignore_index: int,
    seed: int,
    device,
    return_tokens: bool = True,
    return_attn: bool = False,
):
    raw_model = model.module if hasattr(model, "module") else model
    if not hasattr(raw_model, "decode_prompts"):
        raise NotImplementedError("DistillSAM KD requires a model with decode_prompts().")

    teacher_tokens = []
    teacher_attn = []
    for index, mask_path in enumerate(batch["mask_path"]):
        raw_mask = dataset._read_mask(Path(mask_path))
        semantic_mask = dataset.remap_mask(raw_mask)
        classes = (
            list(foreground_classes)
            if foreground_classes is not None
            else infer_foreground_classes(semantic_mask, background_id, ignore_index, num_classes)
        )
        prompt = build_teacher_prompts(
            semantic_mask=semantic_mask,
            raw_image_shape=semantic_mask.shape,
            strategy=strategy,
            num_prompt_tokens=num_prompt_tokens,
            num_classes=num_classes,
            foreground_classes=classes,
            background_id=background_id,
            ignore_index=ignore_index,
            seed=int(seed) + int(index),
        )
        prompt_boxes = prompt.get("boxes")
        boxes = list(prompt_boxes) if prompt_boxes is not None else [prompt.get("box")]
        per_prompt_tokens = []
        per_prompt_attn = []
        for box in boxes:
            sparse_prompt_embeddings, dense_prompt_embeddings = encode_teacher_prompts_with_sam(
                sam_prompt_encoder=raw_model.sam.prompt_encoder,
                point_coords=prompt.get("point_coords"),
                point_labels=prompt.get("point_labels"),
                box=box,
                device=device,
                raw_image_shape=semantic_mask.shape,
                image_size=raw_model.sam.image_encoder.img_size,
            )
            with torch.no_grad():
                teacher_output = raw_model.decode_prompts(
                    image_embeddings=image_embeddings[index : index + 1].detach(),
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_prompt_embeddings,
                    dense_prompt_embeddings=dense_prompt_embeddings,
                    multimask_output=True,
                    return_tokens=return_tokens,
                    return_attn=return_attn,
                )
            if return_tokens:
                per_prompt_tokens.append(teacher_output["mask_tokens_out"].detach())
            if return_attn:
                per_prompt_attn.append(teacher_output["attention_maps"]["final_attn_token_to_image"].detach())
        if return_tokens:
            teacher_tokens.append(torch.stack(per_prompt_tokens, dim=0).mean(dim=0))
        if return_attn:
            teacher_attn.append(torch.stack(per_prompt_attn, dim=0).mean(dim=0))
    output = {}
    if return_tokens:
        output["mask_tokens_out"] = torch.cat(teacher_tokens, dim=0)
    if return_attn:
        output["attention_maps"] = {
            "final_attn_token_to_image": torch.cat(teacher_attn, dim=0),
        }
    return output


def teacher_mask_tokens_from_batch(
    model,
    dataset,
    batch,
    image_embeddings,
    image_pe,
    strategy: str,
    num_prompt_tokens: int,
    num_classes: int,
    foreground_classes: Optional[Iterable[int]],
    background_id: int,
    ignore_index: int,
    seed: int,
    device,
):
    return teacher_decoder_outputs_from_batch(
        model=model,
        dataset=dataset,
        batch=batch,
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        strategy=strategy,
        num_prompt_tokens=num_prompt_tokens,
        num_classes=num_classes,
        foreground_classes=foreground_classes,
        background_id=background_id,
        ignore_index=ignore_index,
        seed=seed,
        device=device,
        return_tokens=True,
        return_attn=False,
    )["mask_tokens_out"]
