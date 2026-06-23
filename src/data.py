"""CIFAR-10 data loader backed by the HuggingFace datasets cache."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as tv_transforms

DEFAULT_DATASET_ID = "uoft-cs/cifar10"
DEFAULT_SPLIT = "train"
NUM_CLASSES = 10


def build_image_transform(image_size: int) -> tv_transforms.Compose:
    """Resize (if needed) and convert to tensor in `[0, 1]`."""
    return tv_transforms.Compose(
        [
            tv_transforms.Resize(
                image_size,
                interpolation=tv_transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            tv_transforms.CenterCrop(image_size),
            tv_transforms.ToTensor(),
        ]
    )


def collate_image_batch(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    """Stack images + labels; normalize images to `[-1, 1]`."""
    images = torch.stack([item["image"] for item in batch])
    labels = torch.as_tensor([int(item["label"]) for item in batch], dtype=torch.long)
    data = images.reshape(images.shape[0], -1) * 2.0 - 1.0
    out: dict[str, Tensor] = {"data": data, "class_id": labels}
    if batch and "sample_id" in batch[0]:
        out["sample_id"] = torch.as_tensor(
            [int(item["sample_id"]) for item in batch],
            dtype=torch.long,
        )
    return out


def build_cifar10_dataloader(
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    split: str = DEFAULT_SPLIT,
    batch_size: int,
    num_workers: int = 4,
    image_size: int = 32,
    pin_memory: bool = False,
    rank: int = 0,
    world_size: int = 1,
    include_sample_id: bool = False,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Build the CIFAR-10 dataloader used by the experiment."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split)
    if include_sample_id:
        dataset = dataset.map(
            lambda example, idx: {**example, "sample_id": idx},
            with_indices=True,
        )
    transform = build_image_transform(image_size)
    # HF CIFAR-10 uses "img" for the image column.
    image_col = "img" if "img" in dataset.column_names else "image"
    label_col = "label" if "label" in dataset.column_names else "fine_label"

    def _apply(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        out: dict[str, list[Any]] = {
            "image": [transform(img.convert("RGB")) for img in batch[image_col]],
            "label": list(batch[label_col]),
        }
        if include_sample_id:
            out["sample_id"] = list(batch["sample_id"])
        return out

    dataset.set_transform(_apply)

    sampler: DistributedSampler | None = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_image_batch,
        persistent_workers=num_workers > 0,
    )
