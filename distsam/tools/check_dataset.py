import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    import yaml  # noqa: F401
    from PIL import Image
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:
    missing = exc.name
    raise SystemExit(
        f"Missing dependency: {missing}. Please install the required environment "
        "with torch, pillow, numpy and pyyaml, then rerun this script."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from distsam.data import AdeSegDataset, default_seg_collate  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Check ADE-style dataset with SAM preprocessing.")
    parser.add_argument("--config", default="distsam/configs/EndoVisSub2017.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--num-vis", type=int, default=8)
    parser.add_argument("--out-dir", default="outputs/check_dataset")
    return parser.parse_args()


def colorize_mask(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    mask = mask.astype(np.int64, copy=False)
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    valid = mask != ignore_index
    labels = np.unique(mask[valid])
    for label in labels:
        label_int = int(label)
        color = np.array(
            [
                (label_int * 37 + 17) % 255,
                (label_int * 67 + 29) % 255,
                (label_int * 97 + 53) % 255,
            ],
            dtype=np.uint8,
        )
        rgb[mask == label_int] = color
    rgb[mask == ignore_index] = np.array([180, 180, 180], dtype=np.uint8)
    return rgb


def save_visualization(sample, out_dir: Path, vis_index: int, ignore_index: int) -> None:
    image = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
    semantic_mask = sample["semantic_mask"].cpu().numpy()
    foreground_mask = sample["foreground_mask"][0].cpu().numpy()
    valid_mask = sample["valid_mask"][0].cpu().numpy()

    semantic_vis = colorize_mask(semantic_mask, ignore_index=ignore_index)
    foreground_vis = np.repeat((foreground_mask * 255).astype(np.uint8)[:, :, None], 3, axis=2)

    valid_vis = np.zeros((valid_mask.shape[0], valid_mask.shape[1], 3), dtype=np.uint8)
    valid_vis[valid_mask > 0.5] = np.array([255, 255, 255], dtype=np.uint8)
    valid_vis[valid_mask <= 0.5] = np.array([255, 80, 80], dtype=np.uint8)

    input_h, input_w = sample["input_size"]
    image_canvas = np.zeros_like(semantic_vis)
    resized_image = np.asarray(
        Image.fromarray(image).resize((input_w, input_h), resample=Image.BILINEAR)
    )
    image_canvas[:input_h, :input_w] = resized_image

    panels = [image_canvas, semantic_vis, foreground_vis, valid_vis]
    canvas = np.concatenate(panels, axis=1)
    out_path = out_dir / f"{vis_index:03d}_{sample['name']}_remapped_check.png"
    Image.fromarray(canvas).save(out_path)


def summarize_dataset(dataset: AdeSegDataset, num_vis: int, out_dir: Path) -> None:
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=default_seg_collate,
    )

    print(f"Dataset samples: {len(dataset)}")
    print("First 20 sample names:")
    for sample in dataset.samples[:20]:
        print(f"  {sample['name']}")

    raw_label_pixel_counts = {}
    remapped_label_pixel_counts = {}
    ignore_index = dataset.preprocessor.ignore_index
    first_batch = None
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, batch in enumerate(dataloader):
        if first_batch is None:
            first_batch = batch

        raw_mask = dataset._read_mask(Path(batch["mask_path"][0]))
        remapped_mask = dataset.remap_mask(raw_mask)
        print("Processed semantic_mask unique:", torch.unique(batch["semantic_mask"]))
        print(
            "Processed semantic_mask ignore pixels:",
            (batch["semantic_mask"] == ignore_index).sum().item(),
        )

        raw_labels, raw_counts = np.unique(raw_mask, return_counts=True)
        raw_stats = {
            int(label.item()): int(count.item()) for label, count in zip(raw_labels, raw_counts)
        }
        remapped_labels, remapped_counts = np.unique(remapped_mask, return_counts=True)
        remapped_stats = {
            int(label.item()): int(count.item())
            for label, count in zip(remapped_labels, remapped_counts)
        }
        print(f"Raw mask unique labels [{batch['name'][0]}]: {raw_stats}")
        print(f"Remapped mask unique labels [{batch['name'][0]}]: {remapped_stats}")
        for label, count in raw_stats.items():
            raw_label_pixel_counts[label] = raw_label_pixel_counts.get(label, 0) + count
        for label, count in remapped_stats.items():
            remapped_label_pixel_counts[label] = remapped_label_pixel_counts.get(label, 0) + count

    for vis_index in range(num_vis):
        sample = dataset[vis_index % len(dataset)]
        save_visualization(sample, out_dir, vis_index=vis_index, ignore_index=ignore_index)

    all_raw_label_ids = sorted(raw_label_pixel_counts.keys())
    print(f"All raw label ids: {all_raw_label_ids}")
    print("Raw pixel counts per label:")
    for label in all_raw_label_ids:
        print(f"  {label}: {raw_label_pixel_counts[label]}")

    all_remapped_label_ids = sorted(remapped_label_pixel_counts.keys())
    print(f"All remapped label ids: {all_remapped_label_ids}")
    print("Remapped pixel counts per label:")
    for label in all_remapped_label_ids:
        print(f"  {label}: {remapped_label_pixel_counts[label]}")

    if first_batch is not None:
        print(f"image tensor shape: {tuple(first_batch['image'].shape)}")
        print(f"semantic_mask shape: {tuple(first_batch['semantic_mask'].shape)}")
        print(f"foreground_mask shape: {tuple(first_batch['foreground_mask'].shape)}")
        print(f"valid_mask shape: {tuple(first_batch['valid_mask'].shape)}")

    print(f"Saved visualizations to: {out_dir}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    dataset = AdeSegDataset(config_path, split=args.split)
    summarize_dataset(dataset, num_vis=args.num_vis, out_dir=out_dir)


if __name__ == "__main__":
    main()
