"""Frechet Inception Distance for class-conditional generation.

One ``compute_fid`` covers both releases: CIFAR-10 scores against clean-fid's
named ``cifar10``/train reference statistics; ImageNet-64 scores against custom
statistics built once from a directory of real validation images. Generation is
class-balanced by default, or conditioned on caller-supplied labels (e.g. the
real val labels, 1-to-1) via ``gen_class_ids``.

Both paths use clean-FID. The ImageNet-64 number is a training-time proxy, not
the ADM-evaluator FID against ``VIRTUAL_imagenet64_labeled.npz`` that ADM/EDM/DiT
report; that evaluator is not included here (see ``scripts/imagenet_preprocessing.py``).

Generation runs on one process by default. Pass ``rank``/``world_size`` (with an
initialized process group) to shard generation: each rank generates its slice of
``gen_class_ids`` into a shared ``image_dir`` (rank-unique filenames), then rank 0
scores the combined directory. Sample generation is the in-training FID bottleneck
on large runs, so distributing it is worthwhile; single-process stays the default
for simple callers (CIFAR eval).

Library module; callers own checkpoint loading, seeding, and I/O.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import torch
from torch import Tensor, nn
import torch.distributed as dist
from torchvision.utils import save_image

CIFAR_IMAGE_SIZE = 32
# clean-fid caches custom reference statistics under this name; the suffix marks
# it as the ImageNet-64 validation reference so it is not confused with the
# built-in named datasets (cifar10, etc.).
_IMAGENET64_VAL_STATS_NAME = "un0_imagenet64_val"


def _class_balanced_ids(
    num_samples: int,
    num_classes: int,
    device: torch.device,
) -> Tensor:
    """Return `num_samples` shuffled class ids with balanced class counts."""
    per_class = num_samples // num_classes
    ids = torch.arange(num_classes, device=device).repeat_interleave(per_class)
    remainder = num_samples - ids.numel()
    if remainder:
        ids = torch.cat([ids, torch.randint(num_classes, (remainder,), device=device)])
    return ids[torch.randperm(ids.numel(), device=device)]


def _shard_for_rank(labels: Tensor, *, rank: int, world_size: int) -> Tensor:
    """Return this rank's contiguous slice of `labels` for sharded generation.

    Ranks `0..remainder-1` get one extra label, so concatenating every rank's
    shard in rank order reproduces `labels` exactly — the scored set is then
    identical to a single-process dump.
    """
    n = labels.numel()
    base = n // world_size
    remainder = n % world_size
    start = rank * base + min(rank, remainder)
    count = base + (1 if rank < remainder else 0)
    return labels[start : start + count]


def _dump_samples(
    model: nn.Module,
    *,
    class_ids: Tensor,
    batch_size: int,
    device: torch.device,
    image_dir: Path,
    image_size: int,
    prefix: str = "gen_",
) -> None:
    """Generate one sample per id in ``class_ids`` and save each as a PNG.

    ``prefix`` namespaces filenames so multiple ranks can write disjoint samples
    into one shared directory for sharded generation.
    """
    class_ids_all = class_ids.to(device)
    idx = 0
    with torch.no_grad():
        for start in range(0, class_ids_all.numel(), batch_size):
            batch_ids = class_ids_all[start : start + batch_size]
            gen_flat = model.sample(batch_ids)
            nchw = gen_flat.reshape(-1, 3, image_size, image_size)
            normalized = ((nchw + 1.0) * 0.5).clamp(0.0, 1.0)
            for img in normalized:
                save_image(img, image_dir / f"{prefix}{idx:06d}.png")
                idx += 1


def _score_against_reference(
    gen_dir: Path,
    *,
    real_image_dir: str | Path | None,
    num_real_samples: int | None,
    image_size: int,
) -> float:
    """Clean-FID score a directory of generated PNGs against the reference.

    CIFAR (``real_image_dir`` None) scores against the named ``cifar10`` stats;
    ImageNet builds custom stats from ``real_image_dir`` once, then scores.

    Callers must have disabled the TorchScript GPU fuser on Blackwell+ (via
    ``common.disable_torchscript_gpu_fuser_on_blackwell``) before reaching here;
    otherwise clean-fid's TorchScript Inception crashes in NVRTC on sm_103.
    """
    from cleanfid import fid as cleanfid

    if real_image_dir is None:
        return float(
            cleanfid.compute_fid(
                str(gen_dir),
                dataset_name="cifar10",
                dataset_res=image_size,
                dataset_split="train",
                mode="clean",
            )
        )
    if not cleanfid.test_stats_exists(_IMAGENET64_VAL_STATS_NAME, "clean"):
        cleanfid.make_custom_stats(
            _IMAGENET64_VAL_STATS_NAME,
            str(real_image_dir),
            num=num_real_samples,
            mode="clean",
        )
    return float(
        cleanfid.compute_fid(
            str(gen_dir),
            dataset_name=_IMAGENET64_VAL_STATS_NAME,
            dataset_split="custom",
            mode="clean",
        )
    )


def compute_fid(
    model: nn.Module,
    *,
    num_samples: int,
    num_classes: int,
    batch_size: int,
    device: torch.device,
    image_size: int = CIFAR_IMAGE_SIZE,
    real_image_dir: str | Path | None = None,
    num_real_samples: int | None = None,
    gen_class_ids: Tensor | None = None,
    image_dir: str | Path | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> float:
    """Compute clean-FID for class-conditional samples.

    Reference statistics:
      - ``real_image_dir`` is None (CIFAR): score against the named ``cifar10``
        train statistics at ``image_size`` (downloaded/cached by clean-fid).
      - ``real_image_dir`` is set (ImageNet): build custom statistics from that
        directory once (``num_real_samples`` images), then score against them.

    Generation labels:
      - ``gen_class_ids`` is None: ``num_samples`` class-balanced ids.
      - ``gen_class_ids`` is set: used verbatim (e.g. real val labels, 1-to-1).

    Distributed generation (``world_size > 1``): each rank generates its shard of
    the labels into a shared ``image_dir`` with a rank-unique filename prefix,
    ``torch.distributed.barrier`` synchronizes, then rank 0 scores the combined
    directory and returns the FID (other ranks return ``nan``). A shared
    ``image_dir`` is required in this mode (and the process group must be
    initialized). When ``world_size == 1`` (the default) an unset ``image_dir``
    uses a tempdir cleaned up on return.
    """
    distributed = world_size > 1
    if distributed and image_dir is None:
        raise ValueError("image_dir (shared) is required when world_size > 1.")

    all_ids = (
        _class_balanced_ids(num_samples, num_classes, device)
        if gen_class_ids is None
        else gen_class_ids
    )
    my_ids = _shard_for_rank(all_ids, rank=rank, world_size=world_size) if distributed else all_ids

    def _run(path: Path) -> float:
        if my_ids.numel() > 0:
            _dump_samples(
                model,
                class_ids=my_ids,
                batch_size=batch_size,
                device=device,
                image_dir=path,
                image_size=image_size,
                prefix=f"gen_r{rank}_" if distributed else "gen_",
            )
        if distributed:
            dist.barrier()
            if rank != 0:
                return float("nan")
        return _score_against_reference(
            path,
            real_image_dir=real_image_dir,
            num_real_samples=num_real_samples,
            image_size=image_size,
        )

    if distributed:
        shared = Path(image_dir)
        if rank == 0:
            shared.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        return _run(shared)
    if image_dir is not None:
        explicit = Path(image_dir)
        explicit.mkdir(parents=True, exist_ok=True)
        return _run(explicit)
    with tempfile.TemporaryDirectory() as td:
        return _run(Path(td))
