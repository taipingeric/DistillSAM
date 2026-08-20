"""DistillSAM training entrypoint.

Single GPU:
python distsam/tools/train_distillsam.py \
  --config distsam/configs/EndoVisSub2017.yaml \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --pvt-pretrained checkpoints/pvt/pvt_tiny.pth \
  --adapter-layers 4,8,12 \
  --token-stride 1 \
  --segmentation-mode semantic \
  --use-revised-decoder \
  --epochs 50 \
  --batch-size 1 \
  --lr 1e-4 \
  --device cuda \
  --save-dir outputs/final_train

DDP:
torchrun --nproc_per_node=2 distsam/tools/train_distillsam.py \
  --config distsam/configs/EndoVisSub2017.yaml \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --pvt-pretrained checkpoints/pvt/pvt_tiny.pth \
  --adapter-layers 4,8,12 \
  --token-stride 1 \
  --segmentation-mode semantic \
  --use-revised-decoder \
  --epochs 50 \
  --batch-size 1 \
  --lr 1e-4 \
  --device cuda \
  --save-dir outputs/final_train_ddp

Binary foreground mode uses the original SAM mask decoder:
python distsam/tools/train_distillsam.py \
  --config distsam/configs/EndoVisSub2017.yaml \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --pvt-pretrained checkpoints/pvt/pvt_tiny.pth \
  --adapter-layers 4,8,12 \
  --token-stride 1 \
  --segmentation-mode binary \
  --epochs 50 \
  --batch-size 1 \
  --lr 1e-4 \
  --device cuda \
  --save-dir outputs/final_binary_train

On the main process, logs stream to the console and to
<save-dir>/train.log (flushed on every write, not just at buffer-fill or
process exit) -- no shell `>` redirection needed.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    import yaml
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
except ModuleNotFoundError as exc:
    raise SystemExit(f"Missing dependency: {exc.name}. Please install torch and pyyaml.") from exc


SPARSE_KD_WEIGHT = 0.1
DENSE_KD_WEIGHT = 0.1
ATTENTION_KD_WEIGHT = 0.05


class _Tee:
    """Mirrors writes to multiple streams, flushing on every write."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return False


def setup_logging(log_path):
    """Mirrors stdout/stderr to log_path, flushed on every write.

    print() output sent to a file via shell redirection (`> log.txt`) sits
    in Python's block-buffered stdout until the buffer fills or the process
    exits, so the log file stays empty for most of a long training run
    otherwise. Opening our own line-buffered file handle here and teeing
    every write to it avoids that."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)


REPO_ROOT = Path(__file__).resolve().parents[2]
SEGMENT_ANYTHING_ROOT = REPO_ROOT / "segment-anything"
for path in (REPO_ROOT, SEGMENT_ANYTHING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from distsam.data import AdeSegDataset, default_seg_collate  # noqa: E402
from distsam.kd import (  # noqa: E402
    aggregate_mask_token_attention,
    binarize_teacher_attention,
    binary_attention_map_kd_loss,
    dense_prompt_kd_loss,
    minmax_normalize_attention,
    resolve_teacher_prompt_strategy,
    sparse_mask_token_kd_loss,
    teacher_decoder_outputs_from_batch,
    teacher_dense_prompts_from_batch,
)
from distsam.losses.seg_losses import bce_dice_loss, semantic_ce_loss  # noqa: E402
from distsam.models.sam_semantic_adapter_model import SamSemanticAdapterModel  # noqa: E402
from distsam.utils.distributed import (  # noqa: E402
    all_reduce_scalar,
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_main_process,
    reduce_dict,
    reduce_metric_sums,
    save_on_main,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train final Distillation-SAM model.")
    parser.add_argument("--config", default="distsam/configs/EndoVisSub2017.yaml")
    parser.add_argument("--sam-checkpoint", default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--adapter-type", choices=["pvt_cross"], default=None)
    parser.add_argument("--pvt-pretrained", default=None)
    parser.add_argument("--adapter-layers", default=None)
    parser.add_argument("--token-stride", type=int, default=None)
    parser.add_argument("--num-prompt-tokens", type=int, default=None)
    parser.add_argument("--segmentation-mode", choices=["semantic", "binary", "auto"], default=None)
    parser.add_argument("--use-revised-decoder", action="store_true", default=None)
    parser.add_argument("--teacher-prompt-strategy", default=None)
    parser.add_argument("--sparse-kd-loss-type", default=None)
    parser.add_argument("--sparse-kd-cos-weight", type=float, default=None)
    parser.add_argument("--dense-kd-loss-type", choices=["mse", "normalized_mse", "cosine", "mse_cos"], default=None)
    parser.add_argument("--dense-kd-cos-weight", type=float, default=None)
    parser.add_argument("--dense-kd-pos-logit", type=float, default=None)
    parser.add_argument("--dense-kd-neg-logit", type=float, default=None)
    parser.add_argument("--attention-kd-loss-type", choices=["bce", "dice", "bce_dice", "mse_binary"], default=None)
    parser.add_argument("--attention-dice-weight", type=float, default=None)
    parser.add_argument("--attention-kd-target", default=None)
    parser.add_argument("--attention-kd-token-type", default=None)
    parser.add_argument("--attention-binarize", choices=["topk_gt_area", "topk_ratio", "threshold"], default=None)
    parser.add_argument("--attention-topk-ratio", type=float, default=None)
    parser.add_argument("--attention-threshold", type=float, default=None)
    parser.add_argument("--attention-min-ratio", type=float, default=None)
    parser.add_argument("--attention-max-ratio", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-dir", default="outputs/final_train")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument(
        "--best-metric",
        choices=["mDice_all", "mDice_fg", "mIoU_all", "mIoU_fg"],
        default="mDice_all",
    )
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-shapes", action="store_true")
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=0)
    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument("--sync-bn", action="store_true")
    parser.add_argument("--find-unused-parameters", action="store_true")
    parser.add_argument("--dataloader-start-method", choices=["spawn", "fork", "forkserver"], default="spawn")
    return parser.parse_args()


def resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_device(args):
    if getattr(args, "distributed", False):
        return torch.device("cuda", args.local_rank)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available.")
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def parse_adapter_layers(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def dataloader_worker_kwargs(args):
    if args.num_workers <= 0:
        return {}
    return {
        "persistent_workers": True,
        "multiprocessing_context": args.dataloader_start_method,
    }


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def strip_module_prefix_if_needed(state_dict):
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_options(config_path, args):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    revised_cfg = model_cfg.get("revised_decoder", {})
    sparse_cfg = config.get("kd", {}).get("sparse_kd", {})
    dense_cfg = config.get("kd", {}).get("dense_kd", {})
    attention_cfg = config.get("kd", {}).get("attention_kd", {})
    dataset_name = dataset_cfg.get("name", Path(config_path).stem)
    teacher_strategy = resolve_teacher_prompt_strategy(config, dataset_name, args.teacher_prompt_strategy)
    num_classes = int(model_cfg.get("num_classes", dataset_cfg.get("num_classes", 2)))
    segmentation_mode = args.segmentation_mode or model_cfg.get("segmentation_mode", "semantic")
    if segmentation_mode == "auto":
        segmentation_mode = "binary" if num_classes <= 2 else "semantic"
    if args.use_revised_decoder is not None:
        use_revised_decoder = bool(args.use_revised_decoder)
    elif segmentation_mode == "binary":
        use_revised_decoder = False
    else:
        use_revised_decoder = bool(model_cfg.get("use_revised_decoder", True))
    if segmentation_mode == "semantic" and not use_revised_decoder:
        raise ValueError("semantic mode requires --use-revised-decoder.")
    if segmentation_mode == "binary" and use_revised_decoder:
        raise ValueError("binary mode uses the original SAM mask decoder; omit --use-revised-decoder.")
    return {
        "model": {
            "num_classes": num_classes,
            "adapter_type": args.adapter_type or model_cfg.get("adapter_type", "pvt_cross"),
            "pvt_pretrained": args.pvt_pretrained if args.pvt_pretrained is not None else model_cfg.get("pvt_pretrained", ""),
            "adapter_layers": parse_adapter_layers(args.adapter_layers) or parse_adapter_layers(model_cfg.get("adapter_layers", [4, 8, 12])),
            "token_stride": args.token_stride if args.token_stride is not None else int(model_cfg.get("token_stride", 1)),
            "num_prompt_tokens": args.num_prompt_tokens if args.num_prompt_tokens is not None else int(model_cfg.get("num_prompt_tokens", 16)),
            "segmentation_mode": segmentation_mode,
            "use_revised_decoder": use_revised_decoder,
            "class_head_hidden_dim": int(revised_cfg.get("class_head_hidden_dim", 256)),
            "use_default_mask_tokens": bool(revised_cfg.get("use_default_mask_tokens", True)),
            "extra_mask_tokens": int(revised_cfg.get("extra_mask_tokens", 0)),
            "ignore_index": int(dataset_cfg.get("ignore_index", 255)),
        },
        "kd": {
            "teacher_prompt_strategy": teacher_strategy,
            "sparse_kd_loss_type": args.sparse_kd_loss_type or sparse_cfg.get("loss_type", "mse"),
            "sparse_kd_cos_weight": float(args.sparse_kd_cos_weight if args.sparse_kd_cos_weight is not None else sparse_cfg.get("cos_weight", 0.0)),
            "dense_kd_loss_type": args.dense_kd_loss_type or dense_cfg.get("loss_type", "mse"),
            "dense_kd_cos_weight": float(args.dense_kd_cos_weight if args.dense_kd_cos_weight is not None else dense_cfg.get("cos_weight", 0.0)),
            "dense_kd_pos_logit": float(args.dense_kd_pos_logit if args.dense_kd_pos_logit is not None else dense_cfg.get("pos_logit", 10.0)),
            "dense_kd_neg_logit": float(args.dense_kd_neg_logit if args.dense_kd_neg_logit is not None else dense_cfg.get("neg_logit", -10.0)),
            "attention_kd_loss_type": args.attention_kd_loss_type or attention_cfg.get("loss_type", "bce_dice"),
            "attention_dice_weight": float(args.attention_dice_weight if args.attention_dice_weight is not None else attention_cfg.get("dice_weight", 1.0)),
            "attention_kd_target": args.attention_kd_target or attention_cfg.get("target", "final_attn_token_to_image"),
            "attention_kd_token_type": args.attention_kd_token_type or attention_cfg.get("token_type", "mask_tokens"),
            "attention_binarize": args.attention_binarize or attention_cfg.get("binarize", "topk_gt_area"),
            "attention_topk_ratio": float(args.attention_topk_ratio if args.attention_topk_ratio is not None else attention_cfg.get("topk_ratio", 0.2)),
            "attention_threshold": float(args.attention_threshold if args.attention_threshold is not None else attention_cfg.get("threshold", 0.5)),
            "attention_min_ratio": float(args.attention_min_ratio if args.attention_min_ratio is not None else attention_cfg.get("min_ratio", 0.01)),
            "attention_max_ratio": float(args.attention_max_ratio if args.attention_max_ratio is not None else attention_cfg.get("max_ratio", 0.80)),
        },
        "dataset": {
            "name": dataset_name,
            "background_id": int(dataset_cfg.get("background_id", 0)),
            "ignore_index": int(dataset_cfg.get("ignore_index", 255)),
            "foreground_classes": dataset_cfg.get("foreground_classes"),
        },
        "config": config,
    }


def make_model(args, options, device):
    model_kwargs = {key: value for key, value in options["model"].items() if key != "ignore_index"}
    return SamSemanticAdapterModel(
        sam_checkpoint=resolve_repo_path(args.sam_checkpoint),
        debug_shapes=args.debug_shapes,
        **model_kwargs,
    ).to(device)


def per_class_metric_sums(pred, target, num_classes, ignore_index, device):
    pred = pred.long()
    target = target.long()
    valid = target != ignore_index
    intersection = torch.zeros(num_classes, dtype=torch.float64, device=device)
    union = torch.zeros(num_classes, dtype=torch.float64, device=device)
    pred_count = torch.zeros(num_classes, dtype=torch.float64, device=device)
    target_count = torch.zeros(num_classes, dtype=torch.float64, device=device)
    for class_id in range(num_classes):
        pred_c = (pred == class_id) & valid
        target_c = (target == class_id) & valid
        intersection[class_id] = (pred_c & target_c).sum()
        union[class_id] = (pred_c | target_c).sum()
        pred_count[class_id] = pred_c.sum()
        target_count[class_id] = target_c.sum()
    return {
        "intersection": intersection,
        "union": union,
        "pred_count": pred_count,
        "target_count": target_count,
    }


def metrics_from_sums(sums):
    intersection = sums["intersection"].cpu()
    union = sums["union"].cpu()
    pred_count = sums["pred_count"].cpu()
    target_count = sums["target_count"].cpu()
    per_class_iou = []
    per_class_dice = []
    skipped = []
    for class_id in range(len(intersection)):
        denom = pred_count[class_id] + target_count[class_id]
        if union[class_id] == 0 and denom == 0:
            per_class_iou.append(float("nan"))
            per_class_dice.append(float("nan"))
            skipped.append(class_id)
        else:
            per_class_iou.append(float(intersection[class_id] / union[class_id]) if union[class_id] > 0 else 0.0)
            per_class_dice.append(float(2.0 * intersection[class_id] / denom) if denom > 0 else 0.0)

    def mean_valid(values):
        valid = [value for value in values if value == value]
        return float(sum(valid) / len(valid)) if valid else float("nan")

    return {
        "per_class_iou": per_class_iou,
        "per_class_dice": per_class_dice,
        "mIoU_all": mean_valid(per_class_iou),
        "mDice_all": mean_valid(per_class_dice),
        "mIoU_fg": mean_valid(per_class_iou[1:]),
        "mDice_fg": mean_valid(per_class_dice[1:]),
        "skipped_classes": skipped,
    }


def compute_param_summary(model):
    raw = unwrap_model(model)
    total_params = sum(param.numel() for param in raw.parameters() if param.requires_grad)
    pvt_params = sum(
        param.numel()
        for name, param in raw.named_parameters()
        if param.requires_grad and name.startswith("adapter.encoder.pvt")
    )
    block_params = 0
    num_blocks = 0
    if hasattr(raw.adapter, "adapter_blocks") and len(raw.adapter.adapter_blocks) > 0:
        num_blocks = len(raw.adapter.adapter_blocks)
        block_params = sum(param.numel() for param in raw.adapter.adapter_blocks[0].parameters() if param.requires_grad)
    return {
        "total_params": int(total_params),
        "pvt_trainable_params": int(pvt_params),
        "one_adapter_block_params": int(block_params),
        "num_adapter_blocks": int(num_blocks),
        "params_for_print": int(pvt_params + block_params),
    }


def write_train_config(save_dir, args, options, param_summary):
    if not is_main_process():
        return
    payload = {
        "entrypoint": "distsam/tools/train_distillsam.py",
        "config": str(args.config),
        "dataset_name": options["dataset"]["name"],
        "sam_checkpoint": str(args.sam_checkpoint),
        "pvt_pretrained": options["model"]["pvt_pretrained"],
        "adapter_type": options["model"]["adapter_type"],
        "adapter_layers": options["model"]["adapter_layers"],
        "token_stride": options["model"]["token_stride"],
        "segmentation_mode": options["model"]["segmentation_mode"],
        "use_revised_decoder": options["model"]["use_revised_decoder"],
        "teacher_prompt_strategy": options["kd"]["teacher_prompt_strategy"],
        "sparse_kd_loss_type": options["kd"]["sparse_kd_loss_type"],
        "dense_kd_loss_type": options["kd"]["dense_kd_loss_type"],
        "attention_kd_loss_type": options["kd"]["attention_kd_loss_type"],
        "attention_binarize": options["kd"]["attention_binarize"],
        "attention_dice_weight": options["kd"]["attention_dice_weight"],
        "lr": args.lr,
        "batch_size_per_gpu": args.batch_size,
        "world_size": get_world_size(),
        "global_batch_size": args.batch_size * get_world_size(),
        "epochs": args.epochs,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "best_metric": args.best_metric,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "total_params": param_summary["total_params"],
        "pvt_trainable_params": param_summary["pvt_trainable_params"],
        "one_adapter_block_params": param_summary["one_adapter_block_params"],
        "num_adapter_blocks": param_summary["num_adapter_blocks"],
    }
    with open(save_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_epoch(model, loader, dataset, optimizer, device, options, train, epoch, args):
    model.train(train)
    loss_sum = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)
    metric_sums = None
    kd_cfg = options["kd"]
    model_cfg = options["model"]
    dataset_cfg = options["dataset"]
    segmentation_mode = model_cfg["segmentation_mode"]
    metric_num_classes = model_cfg["num_classes"] if segmentation_mode == "semantic" else 2

    for step, batch in enumerate(loader, start=1):
        image = batch["image"].to(device, non_blocking=True)
        semantic_target = batch["semantic_mask"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        output = model(image, return_kd_features=train, return_attn=train)

        if segmentation_mode == "semantic":
            seg_loss = semantic_ce_loss(
                output["semantic_logits"],
                semantic_target,
                ignore_index=model_cfg["ignore_index"],
                valid_mask=valid_mask,
            )
        else:
            foreground_target = batch["foreground_mask"].to(device, non_blocking=True)
            binary_loss, _ = bce_dice_loss(output["logits"], foreground_target, valid_mask=valid_mask)
            seg_loss = {"loss": binary_loss, "loss_ce": binary_loss.detach()}
        if train:
            teacher_output = teacher_decoder_outputs_from_batch(
                model=model,
                dataset=dataset,
                batch=batch,
                image_embeddings=output["image_embeddings"],
                image_pe=output["image_pe"],
                strategy=kd_cfg["teacher_prompt_strategy"],
                num_prompt_tokens=model_cfg["num_prompt_tokens"],
                num_classes=model_cfg["num_classes"],
                foreground_classes=dataset_cfg.get("foreground_classes"),
                background_id=dataset_cfg["background_id"],
                ignore_index=dataset_cfg["ignore_index"],
                seed=args.seed + epoch * 100000 + step * 997,
                device=device,
                return_tokens=True,
                return_attn=True,
            )
            sparse_loss = sparse_mask_token_kd_loss(
                output["student_mask_tokens_out"],
                teacher_output["mask_tokens_out"],
                loss_type=kd_cfg["sparse_kd_loss_type"],
                cos_weight=kd_cfg["sparse_kd_cos_weight"],
            )
            teacher_dense = teacher_dense_prompts_from_batch(
                model=model,
                dataset=dataset,
                batch=batch,
                num_classes=model_cfg["num_classes"],
                foreground_classes=dataset_cfg.get("foreground_classes"),
                background_id=dataset_cfg["background_id"],
                ignore_index=dataset_cfg["ignore_index"],
                device=device,
                pos_logit=kd_cfg["dense_kd_pos_logit"],
                neg_logit=kd_cfg["dense_kd_neg_logit"],
            )
            dense_loss = dense_prompt_kd_loss(
                output["student_dense_prompt_embeddings"],
                teacher_dense,
                loss_type=kd_cfg["dense_kd_loss_type"],
                cos_weight=kd_cfg["dense_kd_cos_weight"],
            )
            raw_model = unwrap_model(model)
            num_mask_tokens = raw_model.sam.mask_decoder.num_mask_tokens
            student_map = minmax_normalize_attention(
                aggregate_mask_token_attention(
                    output["student_attention_maps"]["final_attn_token_to_image"],
                    num_mask_tokens=num_mask_tokens,
                )
            )
            teacher_map = minmax_normalize_attention(
                aggregate_mask_token_attention(
                    teacher_output["attention_maps"]["final_attn_token_to_image"],
                    num_mask_tokens=num_mask_tokens,
                )
            ).detach()
            teacher_binary, _ = binarize_teacher_attention(
                teacher_attention_map=teacher_map,
                semantic_mask=semantic_target,
                foreground_classes=dataset_cfg.get("foreground_classes"),
                num_classes=model_cfg["num_classes"],
                background_id=dataset_cfg["background_id"],
                ignore_index=dataset_cfg["ignore_index"],
                mode=kd_cfg["attention_binarize"],
                topk_ratio=kd_cfg["attention_topk_ratio"],
                threshold=kd_cfg["attention_threshold"],
                min_ratio=kd_cfg["attention_min_ratio"],
                max_ratio=kd_cfg["attention_max_ratio"],
            )
            attention_loss = binary_attention_map_kd_loss(
                student_attention_map=student_map,
                teacher_binary_attention_mask=teacher_binary,
                loss_type=kd_cfg["attention_kd_loss_type"],
                dice_weight=kd_cfg["attention_dice_weight"],
            )
            kd_sparse = SPARSE_KD_WEIGHT * sparse_loss["loss_sparse_kd"]
            kd_dense = DENSE_KD_WEIGHT * dense_loss["loss_dense_kd"]
            kd_attn = ATTENTION_KD_WEIGHT * attention_loss["loss_attention_kd"]
        else:
            zero = seg_loss["loss"].new_tensor(0.0)
            kd_sparse = zero
            kd_dense = zero
            kd_attn = zero

        loss = seg_loss["loss"] + kd_sparse + kd_dense + kd_attn
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if segmentation_mode == "semantic":
            pred = output["semantic_logits"].argmax(dim=1)
            metric_target = semantic_target
        else:
            pred = (torch.sigmoid(output["logits"][:, 0]) > 0.5).long()
            metric_target = batch["foreground_mask"][:, 0].to(device, non_blocking=True).long()
            metric_target = metric_target.clone()
            metric_target[valid_mask[:, 0] <= 0.5] = model_cfg["ignore_index"]
        local_sums = per_class_metric_sums(pred, metric_target, metric_num_classes, model_cfg["ignore_index"], device)
        metric_sums = local_sums if metric_sums is None else {key: metric_sums[key] + local_sums[key] for key in metric_sums}
        loss_sum += loss.detach()
        count += 1.0

        should_log = train and (step == 1 or step % args.log_interval == 0)
        if should_log:
            reduced = reduce_dict(
                {
                    "loss": loss.detach(),
                    "seg": seg_loss["loss"].detach(),
                    "kd_sparse": kd_sparse.detach(),
                    "kd_dense": kd_dense.detach(),
                    "kd_attn": kd_attn.detach(),
                },
                average=True,
            )
        if should_log and is_main_process():
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[train] epoch={epoch} iter={step}/{len(loader)} lr={lr:.2e} "
                f"loss={reduced['loss'].item():.4f} seg={reduced['seg'].item():.4f} "
                f"kd_sparse={reduced['kd_sparse'].item():.4f} "
                f"kd_dense={reduced['kd_dense'].item():.4f} "
                f"kd_attn={reduced['kd_attn'].item():.4f}"
            )

    if metric_sums is None:
        metric_sums = per_class_metric_sums(
            torch.empty(0, device=device),
            torch.empty(0, device=device),
            metric_num_classes,
            model_cfg["ignore_index"],
            device,
        )
    reduced_loss_sum = all_reduce_scalar(loss_sum, average=False, device=device)
    reduced_count = all_reduce_scalar(count, average=False, device=device)
    return reduced_loss_sum / max(reduced_count, 1.0), metrics_from_sums(reduce_metric_sums(metric_sums))


def make_checkpoint(model, optimizer, epoch, best_score, options, args, param_summary):
    raw_model = unwrap_model(model)
    model_options = dict(options["model"])
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_metric": args.best_metric,
        "best_score": float(best_score),
        "model_options": model_options,
        "meta": {
            "entrypoint": "train_distillsam.py",
            "dataset_name": options["dataset"]["name"],
            "epoch": int(epoch),
            "best_metric": args.best_metric,
            "best_score": float(best_score),
            "teacher_prompt_strategy": options["kd"]["teacher_prompt_strategy"],
            "total_params": param_summary["total_params"],
        },
        "train_config": {
            "dataset_name": options["dataset"]["name"],
            "teacher_prompt_strategy": options["kd"]["teacher_prompt_strategy"],
            "total_params": param_summary["total_params"],
        },
    }
    return payload


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    raw_model = unwrap_model(model)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    raw_model.load_state_dict(strip_module_prefix_if_needed(state), strict=False)
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if isinstance(checkpoint, dict):
        best_value = checkpoint.get("best_score", checkpoint.get("best_metric", -1.0))
        if isinstance(best_value, str):
            best_value = -1.0
        return int(checkpoint.get("epoch", 0)), float(best_value)
    return 0, -1.0


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    init_distributed_mode(args)
    config_path = resolve_repo_path(args.config)
    save_dir = resolve_repo_path(args.save_dir)
    device = resolve_device(args)
    if is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(save_dir / "train.log")

    options = load_options(config_path, args)
    if options["kd"]["attention_kd_target"] != "final_attn_token_to_image":
        raise NotImplementedError("Only attention_kd_target='final_attn_token_to_image' is supported.")
    if options["kd"]["attention_kd_token_type"] != "mask_tokens":
        raise NotImplementedError("Only attention_kd_token_type='mask_tokens' is supported.")

    train_dataset = AdeSegDataset(config_path, split="train")
    val_dataset = AdeSegDataset(config_path, split="val")
    train_sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True, drop_last=False) if args.distributed else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False, drop_last=False) if args.distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=default_seg_collate,
        pin_memory=torch.cuda.is_available(),
        **dataloader_worker_kwargs(args),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=default_seg_collate,
        pin_memory=torch.cuda.is_available(),
        **dataloader_worker_kwargs(args),
    )

    model = make_model(args, options, device)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=args.find_unused_parameters,
        )
    optimizer = torch.optim.AdamW(unwrap_model(model).get_trainable_parameters(), lr=args.lr)
    param_summary = compute_param_summary(model)

    if is_main_process():
        print(f"[Teacher] dataset={options['dataset']['name']} strategy={options['kd']['teacher_prompt_strategy']}")
        print(f"[Params] PVT trainable params: {param_summary['pvt_trainable_params']}")
        print(f"[Params] one adapter block params: {param_summary['one_adapter_block_params']}")
        print(f"[Params] params: {param_summary['params_for_print']}")
        print("[Params] total params saved to train_config.json")
        write_train_config(save_dir, args, options, param_summary)

    start_epoch = 1
    best_score = -1.0
    if args.resume:
        loaded_epoch, best_score = load_checkpoint(resolve_repo_path(args.resume), model, optimizer, device)
        start_epoch = loaded_epoch + 1
        if is_main_process():
            print(f"[resume] start_epoch={start_epoch} best_{args.best_metric}={best_score:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        run_epoch(model, train_loader, train_dataset, optimizer, device, options, True, epoch, args)
        do_eval = epoch % args.eval_interval == 0 or epoch == args.epochs
        val_metrics = None
        if do_eval:
            _, val_metrics = run_epoch(model, val_loader, val_dataset, optimizer, device, options, False, epoch, args)
            if is_main_process():
                current_score = val_metrics[args.best_metric]
                if current_score == current_score and current_score > best_score:
                    best_score = current_score
                    save_on_main(
                        make_checkpoint(model, optimizer, epoch, best_score, options, args, param_summary),
                        save_dir / "best.pth",
                    )
                    print(f"[checkpoint] saved best.pth {args.best_metric}={best_score:.4f}")
                print(
                    f"[val] epoch={epoch} mIoU_fg={val_metrics['mIoU_fg']:.4f} "
                    f"mDice_fg={val_metrics['mDice_fg']:.4f} "
                    f"mIoU_all={val_metrics['mIoU_all']:.4f} "
                    f"mDice_all={val_metrics['mDice_all']:.4f} "
                    f"best_metric={args.best_metric} best={best_score:.4f}"
                )
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            save_on_main(
                make_checkpoint(model, optimizer, epoch, best_score, options, args, param_summary),
                save_dir / f"epoch_{epoch:03d}.pth",
            )
            if is_main_process():
                print(f"[checkpoint] saved epoch_{epoch:03d}.pth")

    cleanup_distributed()


if __name__ == "__main__":
    main()
