from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class SamSegPreprocessor:
    """SAM-style preprocessing for an RGB image and a semantic segmentation mask."""

    def __init__(
        self,
        image_size: int = 1024,
        pixel_mean=(123.675, 116.28, 103.53),
        pixel_std=(58.395, 57.12, 57.375),
        ignore_index: int = 255,
        background_id: int = 0,
    ) -> None:
        self.image_size = int(image_size)
        self.ignore_index = int(ignore_index)
        self.background_id = int(background_id)
        self.pixel_mean = torch.tensor(pixel_mean, dtype=torch.float32).view(3, 1, 1)
        self.pixel_std = torch.tensor(pixel_std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: np.ndarray, semantic_mask: np.ndarray) -> Dict[str, object]:
        self._validate_image(image)
        self._validate_mask(semantic_mask)

        original_h, original_w = image.shape[:2]
        if semantic_mask.shape[:2] != (original_h, original_w):
            raise ValueError(
                "Image and mask must have the same original size before preprocessing, "
                f"got image {(original_h, original_w)} and mask {semantic_mask.shape[:2]}."
            )

        resized_h, resized_w, scale = self._get_resized_size(original_h, original_w)

        resized_image = np.asarray(
            Image.fromarray(image).resize((resized_w, resized_h), resample=Image.BILINEAR)
        )
        mask_for_resize = self._to_pil_mask_array(semantic_mask)
        resized_mask = np.asarray(
            Image.fromarray(mask_for_resize).resize(
                (resized_w, resized_h), resample=Image.NEAREST
            )
        )

        image_tensor = torch.from_numpy(resized_image.transpose(2, 0, 1).copy()).float()
        image_tensor = (image_tensor - self.pixel_mean) / self.pixel_std

        mask_tensor = torch.from_numpy(resized_mask.astype(np.int64, copy=True)).long()

        pad_h = self.image_size - resized_h
        pad_w = self.image_size - resized_w
        if pad_h < 0 or pad_w < 0:
            raise RuntimeError(
                f"Resized size {(resized_h, resized_w)} exceeds target image_size {self.image_size}."
            )

        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), value=0.0)
        semantic_mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), value=self.ignore_index)

        foreground_mask = (
            (semantic_mask_tensor != self.background_id)
            & (semantic_mask_tensor != self.ignore_index)
        ).float().unsqueeze(0)

        valid_mask = torch.zeros((self.image_size, self.image_size), dtype=torch.float32)
        valid_mask[:resized_h, :resized_w] = 1.0
        valid_mask = valid_mask.unsqueeze(0)

        return {
            "image": image_tensor,
            "semantic_mask": semantic_mask_tensor,
            "foreground_mask": foreground_mask,
            "valid_mask": valid_mask,
            "metadata": {
                "original_size": (original_h, original_w),
                "input_size": (resized_h, resized_w),
                "scale": scale,
                "pad_h": pad_h,
                "pad_w": pad_w,
            },
        }

    def _get_resized_size(self, old_h: int, old_w: int) -> Tuple[int, int, float]:
        scale = float(self.image_size) / float(max(old_h, old_w))
        new_h = int(old_h * scale + 0.5)
        new_w = int(old_w * scale + 0.5)
        return new_h, new_w, scale

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be a numpy array, got {type(image)!r}.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be RGB HWC with 3 channels, got shape {image.shape}.")
        if image.dtype != np.uint8:
            raise ValueError(f"image must be uint8 with values in [0, 255], got {image.dtype}.")

    @staticmethod
    def _validate_mask(semantic_mask: np.ndarray) -> None:
        if not isinstance(semantic_mask, np.ndarray):
            raise TypeError(f"semantic_mask must be a numpy array, got {type(semantic_mask)!r}.")
        if semantic_mask.ndim != 2:
            raise ValueError(f"semantic_mask must be HxW label ids, got shape {semantic_mask.shape}.")

    @staticmethod
    def _to_pil_mask_array(semantic_mask: np.ndarray) -> np.ndarray:
        if semantic_mask.min() < 0:
            raise ValueError("semantic_mask must contain non-negative label ids.")
        max_value = int(semantic_mask.max())
        if max_value <= np.iinfo(np.uint8).max:
            return semantic_mask.astype(np.uint8, copy=True)
        if max_value <= np.iinfo(np.uint16).max:
            return semantic_mask.astype(np.uint16, copy=True)
        return semantic_mask.astype(np.int32, copy=True)
