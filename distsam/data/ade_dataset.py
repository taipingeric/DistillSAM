from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from .sam_preprocess import SamSegPreprocessor


class AdeSegDataset(Dataset):
    """ADE-style semantic segmentation dataset with SAM-style preprocessing."""

    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(self, config: Union[str, Path, Dict], split: str = "train") -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.config = self._load_config(config)
        self.dataset_cfg = self.config.get("dataset", self.config)
        self.split = self._normalize_split(split)

        root = self.dataset_cfg.get("root")
        if root is None:
            raise KeyError("Dataset config must define dataset.root.")
        self.root = self._resolve_repo_path(root)

        split_cfg = self._get_split_config(self.split)
        self.image_dir = self._resolve_under_root(split_cfg, "image_dir")
        self.mask_dir = self._resolve_under_root(split_cfg, "mask_dir")

        image_size = int(self.dataset_cfg.get("image_size", 1024))
        self.ignore_index = int(self.dataset_cfg.get("ignore_index", 255))
        self.background_id = int(self.dataset_cfg.get("background_id", 0))
        self.num_classes = self.dataset_cfg.get("num_classes")
        if self.num_classes is not None:
            self.num_classes = int(self.num_classes)
        self.label_map = self._parse_label_map(self.dataset_cfg.get("label_map"))
        self.inverse_label_map = self._parse_label_map(self.dataset_cfg.get("inverse_label_map"))
        self.preprocessor = SamSegPreprocessor(
            image_size=image_size,
            ignore_index=self.ignore_index,
            background_id=self.background_id,
        )

        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(f"No image files found in {self.image_dir}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]

        image = np.asarray(Image.open(image_path).convert("RGB"))
        raw_mask = self._read_mask(mask_path)

        if image.shape[:2] != raw_mask.shape[:2]:
            raise ValueError(
                f"Image and mask size mismatch for {image_path.name}: "
                f"image {image.shape[:2]}, mask {raw_mask.shape[:2]}."
            )

        mask = self.remap_mask(raw_mask)
        processed = self.preprocessor(image, mask)
        metadata = processed["metadata"]

        return {
            "image": processed["image"],
            "semantic_mask": processed["semantic_mask"].long(),
            "foreground_mask": processed["foreground_mask"].float(),
            "valid_mask": processed["valid_mask"].float(),
            "original_size": metadata["original_size"],
            "input_size": metadata["input_size"],
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "name": sample["name"],
        }

    def _load_config(self, config: Union[str, Path, Dict]) -> Dict:
        if isinstance(config, dict):
            return config
        config_path = self._resolve_repo_path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _normalize_split(self, split: str) -> str:
        if split == "training":
            return "train"
        if split == "validation":
            return "val"
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}.")
        return split

    def _get_split_config(self, split: str) -> Dict:
        splits = self.dataset_cfg.get("splits")
        if splits is not None:
            if split not in splits:
                raise KeyError(f"Config dataset.splits has no entry for split {split!r}.")
            return splits[split]

        image_key = f"{split}_image_dir"
        mask_key = f"{split}_mask_dir"
        if image_key in self.dataset_cfg and mask_key in self.dataset_cfg:
            return {
                "image_dir": self.dataset_cfg[image_key],
                "mask_dir": self.dataset_cfg[mask_key],
            }
        raise KeyError(
            "Dataset config must define dataset.splits.{train,val}.{image_dir,mask_dir} "
            "or train_image_dir/train_mask_dir and val_image_dir/val_mask_dir."
        )

    @staticmethod
    def _parse_label_map(label_map):
        if label_map is None:
            return None
        return {int(raw_id): int(train_id) for raw_id, train_id in label_map.items()}

    def remap_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.label_map is None:
            return mask

        remapped = np.full(mask.shape, self.ignore_index, dtype=np.int64)
        for raw_id, train_id in self.label_map.items():
            remapped[mask == raw_id] = train_id
        return remapped

    def _resolve_repo_path(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return self.repo_root / path

    def _resolve_under_root(self, split_cfg: Dict, key: str) -> Path:
        if key not in split_cfg:
            raise KeyError(f"Split config must define {key}.")
        path = Path(split_cfg[key])
        if path.is_absolute():
            resolved = path
        else:
            resolved = self.root / path
        if not resolved.exists():
            raise FileNotFoundError(f"Configured {key} directory does not exist: {resolved}")
        return resolved

    def _build_samples(self) -> List[Dict[str, object]]:
        image_paths = [
            path
            for path in sorted(self.image_dir.iterdir())
            if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS
        ]
        samples = []
        missing_masks = []
        for image_path in image_paths:
            mask_path = self.mask_dir / f"{image_path.stem}.png"
            if not mask_path.exists():
                missing_masks.append((image_path, mask_path))
                continue
            samples.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "name": image_path.stem,
                }
            )

        if missing_masks:
            examples = "\n".join(
                f"  image={image_path} expected_mask={mask_path}"
                for image_path, mask_path in missing_masks[:10]
            )
            raise FileNotFoundError(
                f"Missing {len(missing_masks)} mask png files matching image stems. Examples:\n{examples}"
            )
        return samples

    @staticmethod
    def _read_mask(mask_path: Path) -> np.ndarray:
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
                "convert it to a single-channel label-id png before using this dataset."
            )
        raise ValueError(
            f"Mask {mask_path} must be HxW label ids or HxWx3 with identical channels, got shape {mask.shape}."
        )
