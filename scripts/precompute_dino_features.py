"""Build a DINO view bank for every CIFAR-10 train row (use with --precomputed-dino-features)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from data import build_cifar10_dataloader  # noqa: E402
from losses import DINOFeatureExtractor, extract_feature_views  # noqa: E402

FEATURE_BATCH_SIZE = 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cifar10_train_dino_views.pt"),
        help="Output .pt path (views tensor + metadata).",
    )
    parser.add_argument("--dataset", default="uoft-cs/cifar10")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this script.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    loader = build_cifar10_dataloader(
        dataset_id=args.dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        image_size=int(args.image_size),
        pin_memory=True,
        include_sample_id=True,
        shuffle=False,
        drop_last=False,
    )
    n_total = len(loader.dataset)

    dino = DINOFeatureExtractor().to(device)
    dino = torch.compile(dino)
    dino.eval()

    bank: torch.Tensor | None = None
    num_views = 0
    feat_dim = 0

    for batch in tqdm(loader, desc="dino precompute"):
        x = batch["data"].to(device=device, non_blocking=True)
        sid = batch["sample_id"].cpu()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            views = extract_feature_views(
                dino,
                x,
                batch_size=FEATURE_BATCH_SIZE,
                image_size=int(args.image_size),
            )
        block = (
            torch.stack(views, dim=1)
            .detach()
            .to(
                dtype=torch.bfloat16,
            )
            .cpu()
        )
        if bank is None:
            num_views = int(block.shape[1])
            feat_dim = int(block.shape[2])
            bank = torch.empty(
                n_total,
                num_views,
                feat_dim,
                dtype=torch.bfloat16,
            )
        bank[sid] = block.to(dtype=torch.bfloat16)

    if bank is None:
        raise RuntimeError("Empty dataloader; no rows written to bank.")

    torch.save(
        {
            "views": bank.contiguous(),
            "num_samples": int(bank.shape[0]),
            "num_views": num_views,
            "feat_dim": feat_dim,
            "dataset_id": args.dataset,
            "image_size": int(args.image_size),
        },
        args.output,
    )
    print(f"Wrote {args.output} shape={tuple(bank.shape)} dtype={bank.dtype}")


if __name__ == "__main__":
    main()
