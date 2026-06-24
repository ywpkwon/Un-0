"""Public ImageNet-64 data loader over preprocessed on-disk PNG shards.

⚠️  This is a REFERENCE loader, not a fixed format: point it at your own
preprocessed data and customize for your storage layout. It assumes the
on-disk ImageFolder tree produced by ``scripts/imagenet_preprocessing.py``::

    <root>/<class_id:05d>/<index:08d>.png      # 64x64 RGB, lossless PNG

Class directories are named by the integer class id (00000..00999), and the
loader surfaces that id directly (not torchvision's sorted-folder index), so
labels line up 1-to-1 with training and FID. It returns the model's
``{"data": flat [-1, 1] tensor, "class_id": long}`` batch contract and shows
the knobs a real run needs (workers, prefetch, pinned memory, DDP sharding).
"""

from __future__ import annotations

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import pil_to_tensor

IMAGE_SIZE = 64
NUM_CLASSES = 1000


def decode_image(image: Image.Image) -> Tensor:
    """Decode a 64x64 RGB PIL image to a flat `[-1, 1]` tensor of shape (12288,)."""
    pixels = pil_to_tensor(image.convert("RGB")).to(torch.float32) / 255.0
    return pixels.reshape(-1) * 2.0 - 1.0


def collate_image_batch(
    batch: list[tuple[Tensor, int]],
) -> dict[str, Tensor]:
    """Collate (flat tensor, label) pairs into the model's `{data, class_id}`."""
    data = torch.stack([d for d, _ in batch])
    class_id = torch.as_tensor([c for _, c in batch], dtype=torch.long)
    return {"data": data, "class_id": class_id}


class _ImageNet64Folder(ImageFolder):
    """ImageFolder yielding flat `[-1, 1]` tensors keyed by true class id.

    Returns the integer class id parsed from the folder name, not
    torchvision's sorted-folder index.
    """

    def __init__(self, root: str) -> None:
        super().__init__(root, loader=lambda p: Image.open(p).convert("RGB"))
        # class_to_idx maps "00007" -> sorted_index; invert to recover ids.
        self._idx_to_class_id = {idx: int(name) for name, idx in self.class_to_idx.items()}

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, folder_idx = super().__getitem__(index)
        return decode_image(image), self._idx_to_class_id[folder_idx]


def build_imagenet64_dataloader(
    *,
    root: str,
    batch_size: int,
    num_workers: int = 16,
    prefetch_factor: int = 4,
    pin_memory: bool = False,
    shuffle: bool = True,
    drop_last: bool = True,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Build the ImageNet-64 dataloader over a preprocessed ImageFolder tree.

    Reference implementation only: it reads PNGs from local disk to keep the
    release self-contained. Swap in your own storage backend (object store,
    webdataset, a streaming loader, ...) as long as it yields the same
    ``{"data", "class_id"}`` batch contract.

    ``root`` is the split directory (e.g. ``<data>/train``). Under DDP
    (``world_size > 1``) a ``DistributedSampler`` shards across ranks.
    """
    dataset = _ImageNet64Folder(root)
    sampler: DistributedSampler | None = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_image_batch,
        persistent_workers=num_workers > 0,
    )
