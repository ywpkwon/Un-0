"""Training entry point for class-conditional ImageNet generation (64x64).

A separate script from ``train_cifar10.py`` on purpose: that path stays pure
and public, and this one carries the ImageNet specifics (1000 classes, 10-step
euler integration, per-step class subsampling, DINO without antialias, FID
against custom validation statistics). Generic training helpers are imported
from ``common.py`` rather than duplicated; some loop scaffolding is
intentionally duplicated for clarity and robustness.

Public data path: ``build_imagenet64_dataloader`` reads a preprocessed
ImageFolder PNG tree (see ``scripts/imagenet_preprocessing.py``); there is no
streaming / object-store dependency. Proxy FID generation is sharded across
ranks under DDP (rank 0 scores the combined samples) and runs single-process
otherwise.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from typing import Any

import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from common import (
    autocast_context,
    make_scheduler,
    resolve_device,
    save_sample_grid,
    seed_everything,
)
from decoupled_adamw import DecoupledAdamW
from imagenet_data import NUM_CLASSES, build_imagenet64_dataloader
from losses import DINOFeatureExtractor, PerClassQueue, conditional_drift_loss
from model import build_imagenet64_model, prepare_class_ids_for_generation

IMAGE_SIZE = 64
WEIGHT_DECAY = 0.0
BETA1 = 0.9
BETA2 = 0.95
WARMUP_FRACTION = 0.15
GRAD_CLIP_NORM = 2.0
FEATURE_BATCH_SIZE = 256
GAMMA = 0.2
NUM_CLASSES_PER_STEP = 64
LOG_EVERY = 10
SAVE_EVERY = 300
SAMPLE_EVERY = 10
FID_NUM_GEN_SAMPLES = 10000
FID_NUM_REAL_SAMPLES = 50000
FID_BATCH_SIZE = 512
# Fixed classes for the sample grid (a readable 10-class slice of the 1000).
SAMPLE_CLASSES = (0, 1, 207, 281, 291, 388, 417, 933, 971, 980)
NUM_SAMPLES_PER_CLASS = 10


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-root",
        required=True,
        help=(
            "Preprocessed ImageNet-64 train tree: <data-root>/<class:05d>/*.png. "
            "Build it with scripts/imagenet_preprocessing.py; customize the "
            "loader for your own storage layout."
        ),
    )
    parser.add_argument(
        "--val-root",
        default=None,
        help=(
            "Preprocessed val tree (<val-root>/<class:05d>/*.png). Used as the "
            "FID real-image reference AND the source of the 1-to-1 generation "
            "labels. Required when --fid-every-epochs > 0."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3600)
    parser.add_argument(
        "--lr-schedule-epochs",
        type=int,
        default=None,
        help=(
            "Epoch horizon the LR warmup+decay is shaped for (defaults to "
            "--epochs). Set to the full reference length when running a short "
            "prefix so per-epoch LR matches the full run."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--precision", choices=("fp32", "tf32", "bf16", "fp16"), default="bf16"
    )
    parser.add_argument("--dino-weight", type=float, default=1.0)
    parser.add_argument("--pixel-weight", type=float, default=0.1)
    parser.add_argument("--queue-size", type=int, default=128)
    parser.add_argument("--num-pos", type=int, default=64)
    parser.add_argument(
        "--queue-storage-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints/imagenet64")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--fid-every-epochs",
        type=int,
        default=0,
        help="If >0, compute and log the proxy FID every N epochs.",
    )
    return parser


def _sample_grid_class_ids(device: torch.device) -> Tensor:
    """Build the fixed eval grid: NUM_SAMPLES_PER_CLASS of each SAMPLE_CLASSES."""
    classes = torch.tensor(SAMPLE_CLASSES, device=device)
    return classes.repeat_interleave(NUM_SAMPLES_PER_CLASS)


def _read_val_labels(val_root: str, *, num_labels: int, seed: int = 0) -> Tensor:
    """Return `num_labels` real validation labels to condition train-time FID.

    The in-training proxy FID generates one sample per returned label, so the
    generated set has the same class marginal as the real validation set
    (`compute_fid`'s `gen_class_ids`). Labels are read from the preprocessed val
    tree's folder names (no streaming dependency) and deterministically shuffled,
    so any prefix is ~class-balanced (val is 50 images/class). Reading the labels
    is cheap: the loader is stubbed (`loader=lambda _p: None`) so no image is
    decoded — only `ImageFolder`'s directory scan runs.
    """
    folder = ImageFolder(val_root, loader=lambda _p: None)
    idx_to_class_id = {idx: int(name) for name, idx in folder.class_to_idx.items()}
    all_labels = torch.tensor(
        [idx_to_class_id[t] for t in folder.targets], dtype=torch.long
    )
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(all_labels.numel(), generator=generator)
    n = min(num_labels, all_labels.numel())
    return all_labels[order][:n]


def train(args: argparse.Namespace) -> None:
    """Run ImageNet-64 training from parsed command line arguments."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        rank = 0
        local_rank = 0
        device = resolve_device("auto")
    is_main = rank == 0

    seed_everything(int(args.seed) + rank)
    if args.precision == "tf32":
        torch.set_float32_matmul_precision("high")

    if args.fid_every_epochs > 0 and args.val_root is None:
        raise ValueError("--val-root is required when --fid-every-epochs > 0.")

    loader = build_imagenet64_dataloader(
        root=args.data_root,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        rank=rank,
        world_size=world_size,
    )

    raw_model = build_imagenet64_model().to(device)
    dino = DINOFeatureExtractor(antialias=False).to(device)
    dino = torch.compile(dino, dynamic=False)

    start_epoch = 0
    global_step = 0
    resume_state: dict[str, Any] | None = None
    if args.resume is not None:
        resume_state = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(resume_state["model"])
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        global_step = int(resume_state.get("global_step", 0))

    model: nn.Module = (
        DistributedDataParallel(raw_model, device_ids=[local_rank])
        if distributed
        else raw_model
    )

    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )
    steps_per_epoch = len(loader)
    schedule_epochs = int(args.lr_schedule_epochs or args.epochs)
    total_steps = schedule_epochs * steps_per_epoch
    scheduler = make_scheduler(
        optimizer, total_steps=total_steps, warmup_fraction=WARMUP_FRACTION,
    )

    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])

    checkpoint_dir = Path(args.checkpoint_dir)
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "weight_decay": WEIGHT_DECAY}

    wandb_run = None
    wandb = None
    if is_main and args.wandb_project is not None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Either drop --wandb-project or run "
                "`uv sync --group logging`."
            ) from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            group=args.wandb_group,
            config=config,
            dir=str(checkpoint_dir),
        )
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("fid", step_metric="epoch")

    def get_train_state(epoch_value: int) -> dict[str, Any]:
        return {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch_value,
            "global_step": global_step,
            "config": config,
        }

    grid_class_ids = _sample_grid_class_ids(device)

    fid_all_class_ids: Tensor | None = None
    if args.fid_every_epochs > 0:
        fid_all_class_ids = _read_val_labels(
            args.val_root, num_labels=FID_NUM_GEN_SAMPLES, seed=int(args.seed),
        )

    queue_size = int(args.queue_size)
    num_pos = int(args.num_pos)
    use_queue = queue_size > 0
    queue_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.queue_storage_dtype]
    queue = (
        PerClassQueue(
            num_classes=NUM_CLASSES,
            queue_size=queue_size,
            data_dim=3 * IMAGE_SIZE * IMAGE_SIZE,
            device=device,
            dtype=queue_dtype,
        )
        if use_queue
        else None
    )

    for epoch in range(start_epoch, int(args.epochs)):
        model.train()
        dino.train()
        if distributed and loader.sampler is not None:
            loader.sampler.set_epoch(epoch)
        progress_iter = enumerate(loader)
        progress = (
            tqdm(progress_iter, total=steps_per_epoch, desc=f"epoch {epoch + 1}/{args.epochs}")
            if is_main
            else progress_iter
        )
        for _step, batch in progress:
            x_real = batch["data"].to(device=device, non_blocking=True)
            class_id_real = batch["class_id"].to(device=device, non_blocking=True)
            class_id_gen = prepare_class_ids_for_generation(
                num_samples=x_real.shape[0],
                num_classes_per_step=NUM_CLASSES_PER_STEP,
                num_total_classes=NUM_CLASSES,
                device=device,
            )

            x_real_pos: Tensor | None = None
            class_id_pos: Tensor | None = None
            queue_ready = True
            if use_queue:
                queue.push(x_real.detach(), class_id_real)
                gen_classes = torch.unique(class_id_gen)
                ready_gen = gen_classes[queue.ready_mask(num_pos)[gen_classes]]
                ready_local = ready_gen.numel() > 0
                if distributed:
                    ready_t = torch.tensor(
                        int(ready_local), device=device, dtype=torch.int32
                    )
                    dist.all_reduce(ready_t, op=dist.ReduceOp.MIN)
                    queue_ready = bool(ready_t)
                else:
                    queue_ready = ready_local
                if queue_ready:
                    x_real_pos, class_id_pos, _ = queue.draw(ready_gen, num_pos=num_pos)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                x_gen = model(class_id_gen)
                if queue_ready:
                    loss, metrics = conditional_drift_loss(
                        x_real,
                        x_gen,
                        class_id_real,
                        class_id_gen,
                        dino=dino,
                        dino_weight=float(args.dino_weight),
                        pixel_weight=float(args.pixel_weight),
                        gamma=GAMMA,
                        feature_batch_size=FEATURE_BATCH_SIZE,
                        image_size=IMAGE_SIZE,
                        x_real_pos=x_real_pos,
                        class_id_pos=class_id_pos,
                        compile_drift=True,
                    )
                else:
                    loss = (x_gen * 0.0).sum()
                    metrics = {"loss/total": loss.detach()}
                metrics["queue_ready"] = torch.as_tensor(
                    float(queue_ready), device=device, dtype=x_gen.dtype,
                )

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if is_main and global_step % LOG_EVERY == 0:
                log_metrics = {
                    name: float(value.detach().cpu()) for name, value in metrics.items()
                }
                log_metrics["lr"] = float(scheduler.get_last_lr()[0])
                log_metrics["grad_norm"] = float(grad_norm.detach().cpu())
                progress.set_postfix(log_metrics)
                if wandb_run is not None:
                    wandb_run.log(log_metrics, step=global_step)

        epoch_num = epoch + 1
        fid_firing = (
            args.fid_every_epochs > 0 and epoch_num % args.fid_every_epochs == 0
        )
        should_sample = epoch_num % SAMPLE_EVERY == 0 or fid_firing
        if is_main and should_sample:
            samples = raw_model.sample(grid_class_ids)
            sample_path = checkpoint_dir / "samples" / f"epoch_{epoch_num:04d}.png"
            save_sample_grid(samples, sample_path, image_size=IMAGE_SIZE)
            if wandb_run is not None:
                wandb_run.log(
                    {"samples": wandb.Image(str(sample_path))}, step=global_step
                )
        # Proxy FID. Under DDP, generation is sharded across ranks into one
        # shared dir (rank-unique filenames); rank 0 scores the combined samples
        # while the others get nan. All ranks must call compute_fid so the
        # barrier inside it stays collective. Single-process uses a tempdir.
        if fid_firing:
            from metrics import compute_fid

            # A shared on-disk dir is needed to gather ranks' samples; it is
            # removed after scoring so 10k PNGs/event don't accumulate over the
            # run. Single-process falls back to compute_fid's tempdir.
            fid_dir = (
                checkpoint_dir / "fid_samples" / f"epoch_{epoch_num:04d}"
                if distributed
                else None
            )
            raw_model.eval()
            fid_value = compute_fid(
                raw_model,
                num_samples=FID_NUM_GEN_SAMPLES,
                num_classes=NUM_CLASSES,
                batch_size=FID_BATCH_SIZE,
                device=device,
                image_size=IMAGE_SIZE,
                real_image_dir=args.val_root,
                num_real_samples=FID_NUM_REAL_SAMPLES,
                gen_class_ids=fid_all_class_ids.to(device),
                image_dir=fid_dir,
                rank=rank,
                world_size=world_size,
            )
            raw_model.train()
            if is_main:
                if fid_dir is not None:
                    shutil.rmtree(fid_dir, ignore_errors=True)
                print(f"FID @ epoch {epoch_num}: {fid_value:.4f}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({"fid": fid_value, "epoch": epoch_num}, step=global_step)
        if is_main and epoch_num % SAVE_EVERY == 0:
            state = get_train_state(epoch)
            torch.save(state, checkpoint_dir / f"epoch_{epoch_num:04d}.pt")
            torch.save(state, checkpoint_dir / "latest.pt")
        if distributed:
            dist.barrier()

    if is_main:
        torch.save(get_train_state(int(args.epochs) - 1), checkpoint_dir / "final.pt")
    if is_main and wandb_run is not None:
        wandb_run.finish()
    if distributed:
        dist.destroy_process_group()


def main() -> None:
    """Parse arguments and run training."""
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
