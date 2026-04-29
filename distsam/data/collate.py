import torch


def default_seg_collate(batch):
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "semantic_mask": torch.stack([item["semantic_mask"] for item in batch], dim=0),
        "foreground_mask": torch.stack([item["foreground_mask"] for item in batch], dim=0),
        "valid_mask": torch.stack([item["valid_mask"] for item in batch], dim=0),
        "original_size": [item["original_size"] for item in batch],
        "input_size": [item["input_size"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "mask_path": [item["mask_path"] for item in batch],
        "name": [item["name"] for item in batch],
    }
