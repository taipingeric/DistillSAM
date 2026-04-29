"""Create a dataset yaml for an ADE-style segmentation dataset.

Example:
python distsam/tools/create_ade_config.py \
  --root datasets/NewDataset \
  --name NewDataset \
  --out distsam/configs/NewDataset.yaml \
  --train-images images/training \
  --train-masks annotations/training \
  --val-images images/validation \
  --val-masks annotations/validation \
  --background-id 0 \
  --ignore-index 255 \
  --remap-labels
"""

import argparse
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    import yaml
    from PIL import Image
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}. Please install pillow, numpy and pyyaml, then rerun."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Create ADE-style segmentation dataset config.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-images", default="images/training")
    parser.add_argument("--train-masks", default="annotations/training")
    parser.add_argument("--val-images", default="images/validation")
    parser.add_argument("--val-masks", default="annotations/validation")
    parser.add_argument("--background-id", type=int, default=0)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--remap-labels", action="store_true")
    parser.add_argument("--mask-suffix", default=".png")
    parser.add_argument(
        "--image-suffixes",
        default=".jpg,.jpeg,.png",
        help="Comma-separated image suffix list written into yaml.",
    )
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--num-prompt-tokens", type=int, default=16)
    return parser.parse_args()


def resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def read_label_mask(mask_path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3 and mask.shape[2] == 3:
        same_channels = np.array_equal(mask[:, :, 0], mask[:, :, 1]) and np.array_equal(
            mask[:, :, 0], mask[:, :, 2]
        )
        if same_channels:
            return mask[:, :, 0]
        raise ValueError(
            f"Mask {mask_path} has 3 different channels. This is not a label-id mask; "
            "convert it to a single-channel label-id png before creating the config."
        )
    raise ValueError(
        f"Mask {mask_path} must be HxW label ids or HxWx3 with identical channels, got {mask.shape}."
    )


def scan_mask_dir(mask_dir: Path, mask_suffix: str) -> Counter:
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")
    mask_paths = sorted(path for path in mask_dir.iterdir() if path.is_file() and path.suffix == mask_suffix)
    if not mask_paths:
        raise RuntimeError(f"No mask files with suffix {mask_suffix!r} found in {mask_dir}.")

    counts = Counter()
    for mask_path in mask_paths:
        mask = read_label_mask(mask_path)
        labels, pixel_counts = np.unique(mask, return_counts=True)
        for label, count in zip(labels, pixel_counts):
            counts[int(label)] += int(count)
    return counts


def is_continuous(labels):
    if not labels:
        return False
    return labels == list(range(min(labels), max(labels) + 1)) and min(labels) == 0


def build_maps(raw_labels, background_id, ignore_index, remap_labels):
    valid_labels = [label for label in raw_labels if label != ignore_index]
    if background_id not in valid_labels:
        valid_labels = [background_id] + valid_labels
    valid_labels = sorted(set(valid_labels))

    if remap_labels:
        label_map = {background_id: 0}
        next_id = 1
        for label in valid_labels:
            if label == background_id:
                continue
            label_map[label] = next_id
            next_id += 1
        inverse_label_map = {train_id: raw_id for raw_id, train_id in label_map.items()}
        num_classes = len(label_map)
        return num_classes, label_map, inverse_label_map

    num_classes = max(valid_labels) + 1 if valid_labels else 0
    return num_classes, None, None


def yaml_dump(data, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def print_next_command(out_path: Path, dataset_name: str):
    display_path = out_path
    try:
        display_path = out_path.relative_to(REPO_ROOT)
    except ValueError:
        pass
    print("\nNext step:")
    print("python distsam/tools/check_dataset.py \\")
    print(f"  --config {display_path} \\")
    print("  --split train \\")
    print("  --num-vis 8 \\")
    print(f"  --out-dir outputs/check_dataset/{dataset_name}_train")


def main():
    args = parse_args()
    root = resolve_repo_path(args.root)
    out_path = resolve_repo_path(args.out)
    train_mask_dir = root / args.train_masks
    val_mask_dir = root / args.val_masks
    train_image_dir = root / args.train_images
    val_image_dir = root / args.val_images

    for path, name in [
        (root, "dataset root"),
        (train_image_dir, "train image dir"),
        (val_image_dir, "val image dir"),
        (train_mask_dir, "train mask dir"),
        (val_mask_dir, "val mask dir"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    counts = Counter()
    counts.update(scan_mask_dir(train_mask_dir, args.mask_suffix))
    counts.update(scan_mask_dir(val_mask_dir, args.mask_suffix))
    raw_labels = sorted(counts.keys())
    valid_labels = [label for label in raw_labels if label != args.ignore_index]
    continuous = is_continuous(valid_labels)
    num_classes, label_map, inverse_label_map = build_maps(
        raw_labels,
        args.background_id,
        args.ignore_index,
        args.remap_labels,
    )

    print(f"Dataset root: {root}")
    print(f"Raw label ids: {raw_labels}")
    print("Pixel counts per raw label:")
    for label in raw_labels:
        print(f"  {label}: {counts[label]}")
    print(f"Labels continuous from 0: {continuous}")
    if not args.remap_labels and not continuous:
        print("[WARNING] Raw labels are not continuous. Consider using --remap-labels.")
    print(f"Suggested num_classes: {num_classes}")
    if label_map is not None:
        print(f"Suggested label_map: {label_map}")
        print(f"Suggested inverse_label_map: {inverse_label_map}")

    image_suffixes = [suffix.strip() for suffix in args.image_suffixes.split(",") if suffix.strip()]
    dataset_cfg = {
        "name": args.name,
        "root": args.root,
        "train_images": args.train_images,
        "train_masks": args.train_masks,
        "val_images": args.val_images,
        "val_masks": args.val_masks,
        "image_suffixes": image_suffixes,
        "mask_suffix": args.mask_suffix,
        "image_size": args.image_size,
        "background_id": args.background_id,
        "ignore_index": args.ignore_index,
        "num_classes": int(num_classes),
        "label_map": None,
        "inverse_label_map": None,
        "splits": {
            "train": {
                "image_dir": args.train_images,
                "mask_dir": args.train_masks,
            },
            "val": {
                "image_dir": args.val_images,
                "mask_dir": args.val_masks,
            },
        },
    }
    if label_map is not None:
        dataset_cfg["label_map"] = {int(k): int(v) for k, v in label_map.items()}
        dataset_cfg["inverse_label_map"] = {int(k): int(v) for k, v in inverse_label_map.items()}

    config = {
        "dataset": dataset_cfg,
        "sam": {
            "image_size": args.image_size,
            "pixel_mean": [123.675, 116.28, 103.53],
            "pixel_std": [58.395, 57.12, 57.375],
        },
        "model": {
            "num_prompt_tokens": args.num_prompt_tokens,
        },
    }
    yaml_dump(config, out_path)
    print(f"\nWrote config: {out_path}")
    print_next_command(out_path, args.name)


if __name__ == "__main__":
    main()
