"""DistillSAM evaluation entrypoint.

python distsam/tools/eval_distillsam.py \
  --config distsam/configs/EndoVisSub2017.yaml \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --checkpoint outputs/final_train/best.pth \
  --adapter-layers 4,8,12 \
  --token-stride 1 \
  --segmentation-mode semantic \
  --use-revised-decoder \
  --split val \
  --device cuda \
  --out-dir outputs/final_eval

For a binary foreground checkpoint, use --segmentation-mode binary and omit
--use-revised-decoder.
"""

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    import yaml
    from PIL import Image
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:
    raise SystemExit(f"Missing dependency: {exc.name}. Please install torch, numpy, pillow and pyyaml.") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
SEGMENT_ANYTHING_ROOT = REPO_ROOT / "segment-anything"
for path in (REPO_ROOT, SEGMENT_ANYTHING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from distsam.data import AdeSegDataset, default_seg_collate  # noqa: E402
from distsam.metrics import compute_multiclass_iou_dice  # noqa: E402
from distsam.models.sam_semantic_adapter_model import SamSemanticAdapterModel  # noqa: E402
from distsam.tools.train_distillsam import parse_adapter_layers, resolve_repo_path, strip_module_prefix_if_needed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate final Distillation-SAM model.")
    parser.add_argument("--config", default="distsam/configs/EndoVisSub2017.yaml")
    parser.add_argument("--sam-checkpoint", default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter-type", choices=["pvt_cross"], default=None)
    parser.add_argument("--pvt-pretrained", default=None)
    parser.add_argument("--adapter-layers", default=None)
    parser.add_argument("--token-stride", type=int, default=None)
    parser.add_argument("--num-prompt-tokens", type=int, default=None)
    parser.add_argument("--segmentation-mode", choices=["semantic", "binary", "auto"], default=None)
    parser.add_argument("--use-revised-decoder", action="store_true", default=None)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="outputs/final_eval")
    parser.add_argument("--num-vis", type=int, default=8)
    parser.add_argument("--debug-shapes", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available.")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_options(config_path, args, checkpoint):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    revised_cfg = model_cfg.get("revised_decoder", {})
    ckpt_options = checkpoint.get("model_options", {}) if isinstance(checkpoint, dict) else {}
    num_classes = int(ckpt_options.get("num_classes", model_cfg.get("num_classes", dataset_cfg.get("num_classes", 2))))
    segmentation_mode = args.segmentation_mode or ckpt_options.get("segmentation_mode", model_cfg.get("segmentation_mode", "semantic"))
    if segmentation_mode == "auto":
        segmentation_mode = "binary" if num_classes <= 2 else "semantic"
    if args.use_revised_decoder is not None:
        use_revised_decoder = bool(args.use_revised_decoder)
    elif segmentation_mode == "binary":
        use_revised_decoder = False
    else:
        use_revised_decoder = bool(ckpt_options.get("use_revised_decoder", model_cfg.get("use_revised_decoder", True)))
    if segmentation_mode == "semantic" and not use_revised_decoder:
        raise ValueError("semantic mode requires --use-revised-decoder.")
    if segmentation_mode == "binary" and use_revised_decoder:
        raise ValueError("binary mode uses the original SAM mask decoder; omit --use-revised-decoder.")
    return {
        "num_classes": num_classes,
        "adapter_type": args.adapter_type or ckpt_options.get("adapter_type", model_cfg.get("adapter_type", "pvt_cross")),
        "pvt_pretrained": args.pvt_pretrained if args.pvt_pretrained is not None else ckpt_options.get("pvt_pretrained", model_cfg.get("pvt_pretrained", "")),
        "adapter_layers": parse_adapter_layers(args.adapter_layers) or parse_adapter_layers(ckpt_options.get("adapter_layers", model_cfg.get("adapter_layers", [4, 8, 12]))),
        "token_stride": args.token_stride if args.token_stride is not None else int(ckpt_options.get("token_stride", model_cfg.get("token_stride", 1))),
        "num_prompt_tokens": args.num_prompt_tokens if args.num_prompt_tokens is not None else int(ckpt_options.get("num_prompt_tokens", model_cfg.get("num_prompt_tokens", 16))),
        "segmentation_mode": segmentation_mode,
        "use_revised_decoder": use_revised_decoder,
        "class_head_hidden_dim": int(ckpt_options.get("class_head_hidden_dim", revised_cfg.get("class_head_hidden_dim", 256))),
        "use_default_mask_tokens": bool(ckpt_options.get("use_default_mask_tokens", revised_cfg.get("use_default_mask_tokens", True))),
        "extra_mask_tokens": int(ckpt_options.get("extra_mask_tokens", revised_cfg.get("extra_mask_tokens", 0))),
        "ignore_index": int(dataset_cfg.get("ignore_index", 255)),
    }


def colorize(mask, ignore_index=255):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    valid = mask != ignore_index
    for label in np.unique(mask[valid]):
        label = int(label)
        rgb[mask == label] = np.array(
            [(label * 37 + 17) % 255, (label * 67 + 29) % 255, (label * 97 + 53) % 255],
            dtype=np.uint8,
        )
    rgb[mask == ignore_index] = np.array([180, 180, 180], dtype=np.uint8)
    return rgb


def image_canvas(image_path, input_size, image_size=1024):
    image = np.asarray(Image.open(image_path).convert("RGB"))
    h, w = input_size
    resized = np.asarray(Image.fromarray(image).resize((w, h), resample=Image.BILINEAR))
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    canvas[:h, :w] = resized
    return canvas


def save_raw_id_prediction(pred, out_path, inverse_label_map, ignore_index):
    raw = np.full(pred.shape, ignore_index, dtype=np.uint8)
    for train_id, raw_id in inverse_label_map.items():
        raw[pred == int(train_id)] = int(raw_id)
    Image.fromarray(raw).save(out_path)


def save_visualizations(batch, pred, out_dir, start_index, max_items, ignore_index, inverse_label_map):
    saved = 0
    for item_idx, name in enumerate(batch["name"]):
        if start_index + saved >= max_items:
            break
        image = image_canvas(batch["image_path"][item_idx], batch["input_size"][item_idx])
        gt = batch["semantic_mask"][item_idx].cpu().numpy()
        pred_np = pred[item_idx].detach().cpu().numpy()
        pred_vis = colorize(pred_np, ignore_index)
        overlay = image.astype(np.float32)
        valid = gt != ignore_index
        overlay[valid] = 0.55 * overlay[valid] + 0.45 * pred_vis[valid].astype(np.float32)
        canvas = np.concatenate(
            [image, colorize(gt, ignore_index), pred_vis, np.clip(overlay, 0, 255).astype(np.uint8)],
            axis=1,
        )
        out_index = start_index + saved
        Image.fromarray(canvas).save(out_dir / f"{out_index:03d}_{name}_semantic_eval.png")
        if inverse_label_map:
            save_raw_id_prediction(pred_np, out_dir / f"{out_index:03d}_{name}_pred_raw_id.png", inverse_label_map, ignore_index)
        saved += 1
    return saved


def save_binary_visualizations(batch, pred, out_dir, start_index, max_items):
    saved = 0
    for item_idx, name in enumerate(batch["name"]):
        if start_index + saved >= max_items:
            break
        image = image_canvas(batch["image_path"][item_idx], batch["input_size"][item_idx])
        gt = batch["foreground_mask"][item_idx, 0].cpu().numpy()
        pred_np = pred[item_idx].detach().cpu().numpy()
        gt_vis = np.repeat((gt * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        pred_vis = np.repeat((pred_np * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        overlay = image.astype(np.float32)
        overlay[pred_np > 0] = 0.55 * overlay[pred_np > 0] + 0.45 * np.array([255, 80, 80], dtype=np.float32)
        canvas = np.concatenate([image, gt_vis, pred_vis, np.clip(overlay, 0, 255).astype(np.uint8)], axis=1)
        out_index = start_index + saved
        Image.fromarray(canvas).save(out_dir / f"{out_index:03d}_{name}_binary_eval.png")
        saved += 1
    return saved


def load_checkpoint(model, checkpoint):
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(strip_module_prefix_if_needed(state), strict=False)


def print_metrics(metrics):
    print(f"mIoU_all: {metrics['mIoU_all']:.4f}")
    print(f"mDice_all: {metrics['mDice_all']:.4f}")
    print(f"mIoU_fg: {metrics['mIoU_fg']:.4f}")
    print(f"mDice_fg: {metrics['mDice_fg']:.4f}")
    print(f"per-class IoU: {metrics['per_class_iou']}")
    print(f"per-class Dice: {metrics['per_class_dice']}")
    if metrics["skipped_classes"]:
        print(f"Skipped absent classes: {metrics['skipped_classes']}")


def main():
    args = parse_args()
    checkpoint_path = resolve_repo_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config_path = resolve_repo_path(args.config)
    options = load_options(config_path, args, checkpoint)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print(f"checkpoint: {checkpoint_path}")
    print(f"dataset/split: {Path(args.config).stem}/{args.split}")

    dataset = AdeSegDataset(config_path, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=default_seg_collate,
    )
    model_kwargs = {key: value for key, value in options.items() if key != "ignore_index"}
    model = SamSemanticAdapterModel(
        sam_checkpoint=resolve_repo_path(args.sam_checkpoint),
        debug_shapes=args.debug_shapes,
        **model_kwargs,
    ).to(device)
    load_checkpoint(model, checkpoint)
    model.eval()

    preds = []
    targets = []
    saved = 0
    inverse_label_map = dataset.inverse_label_map or {}
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            output = model(image)
            if options["segmentation_mode"] == "semantic":
                pred = output["semantic_logits"].argmax(dim=1).cpu()
                target = batch["semantic_mask"].long()
            else:
                pred = (torch.sigmoid(output["logits"][:, 0]) > 0.5).long().cpu()
                target = batch["foreground_mask"][:, 0].long()
                target = target.clone()
                target[batch["valid_mask"][:, 0] <= 0.5] = options["ignore_index"]
            preds.append(pred)
            targets.append(target)
            if saved < args.num_vis:
                if options["segmentation_mode"] == "semantic":
                    saved += save_visualizations(
                        batch,
                        pred,
                        out_dir,
                        saved,
                        args.num_vis,
                        options["ignore_index"],
                        inverse_label_map,
                    )
                else:
                    saved += save_binary_visualizations(batch, pred, out_dir, saved, args.num_vis)

    metric_num_classes = options["num_classes"] if options["segmentation_mode"] == "semantic" else 2
    metrics = compute_multiclass_iou_dice(
        torch.cat(preds),
        torch.cat(targets),
        metric_num_classes,
        options["ignore_index"],
    )
    print_metrics(metrics)
    print(f"visualization output dir: {out_dir}")


if __name__ == "__main__":
    main()
