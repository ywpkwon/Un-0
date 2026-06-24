"""Plain PyTorch training entry point for class-conditional CIFAR-10 generation."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
from typing import Any

import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from un0.common import (
    autocast_context,
    disable_cudnn_sdp_on_blackwell,
    disable_torchscript_gpu_fuser_on_blackwell,
    make_scheduler,
    resolve_device,
    save_sample_grid,
    seed_everything,
)
from un0.data import NUM_CLASSES, build_cifar10_dataloader
from un0.decoupled_adamw import DecoupledAdamW
from un0.losses import (
    DINOFeatureExtractor,
    PerClassQueue,
    conditional_drift_loss,
    gather_precomputed_dino_views,
)
from un0.model import build_cifar10_model

IMAGE_SIZE = 32
WEIGHT_DECAY = 1e-3
BETA1 = 0.9
BETA2 = 0.95
WARMUP_FRACTION = 0.1
GRAD_CLIP_NORM = 1.0
FEATURE_BATCH_SIZE = 64
SAVE_EVERY = 100
SAMPLE_EVERY = 100
EARLY_SAMPLE_EPOCHS = frozenset({1, 5, 25, 50})
NUM_SAMPLES_PER_CLASS = 10
LOG_EVERY = 10
GAMMA = 0.2
FID_NUM_SAMPLES = 50000
FID_BATCH_SIZE = 256


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="uoft-cs/cifar10")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.0013894955)
    parser.add_argument("--precision", choices=("fp32", "tf32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--dino-weight", type=float, default=1.0)
    parser.add_argument("--pixel-weight", type=float, default=0.004)
    parser.add_argument(
        "--parameterization",
        choices=("standard", "mup"),
        default="standard",
        help="Coupling-matrix parameterization.",
    )
    parser.add_argument(
        "--relativization",
        choices=("absolute", "mean_relative", "ref_oscillator", "pairwise"),
        default="mean_relative",
        help="Phase relativization before sin/cos readout encoding.",
    )
    parser.add_argument(
        "--encoding",
        choices=("raw", "sin", "sin_cos"),
        default="sin_cos",
        help=(
            "Readout encoding of the phases fed to the decoder. 'sin_cos' is "
            "the release default; 'raw' passes phases straight to the decoder "
            "(no readout transform)."
        ),
    )
    parser.add_argument(
        "--n-oscillators",
        type=int,
        default=4096,
        help="Number of main Kuramoto oscillators.",
    )
    parser.add_argument(
        "--n-conditional-oscillators",
        type=int,
        default=8,
        help="Number of class-conditioning driver oscillators.",
    )
    parser.add_argument(
        "--class-dropout-prob",
        type=float,
        default=0.1,
        help="Probability of dropping the class drive during training.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help=(
            "Integration steps in the generator rollout. 0 decodes the random "
            "initial phases directly (decoder-only, no dynamics)."
        ),
    )
    parser.add_argument(
        "--solver",
        choices=("euler", "rk4"),
        default="euler",
        help="ODE solver for the dynamics rollout.",
    )
    parser.add_argument(
        "--freeze-dynamics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Freeze the Kuramoto dynamics at initialization (reservoir mode); "
            "only the decoder trains."
        ),
    )
    parser.add_argument(
        "--decoder-in-channels",
        type=int,
        default=None,
        help=(
            "Initial decoder channel count. Defaults to "
            "2 * n_oscillators / 16 for the 4x4 decoder stem."
        ),
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=2048,
        help="Per-class positive queue capacity. 0 disables the queue.",
    )
    parser.add_argument(
        "--num-pos",
        type=int,
        default=64,
        help="Positives drawn from each queue per step (queue mode only).",
    )
    parser.add_argument(
        "--queue-storage-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help=(
            "Queue buffer storage dtype. bfloat16 halves VRAM at larger "
            "scales (e.g. ImageNet-64 with 1000 classes); fp32 is bit-exact "
            "and cheap at CIFAR scale."
        ),
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints/cifar10")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--precomputed-dino-features",
        default=None,
        help=(
            "Path to a .pt bank from scripts/precompute_dino_features.py. "
            "Skips live DINO on real images; generator views still go live."
        ),
    )
    parser.add_argument(
        "--fid-every-epochs",
        type=int,
        default=0,
        help=(
            "If >0, compute and log FID to W&B every N epochs (rank 0 only). "
            "Other ranks wait at the end-of-epoch barrier."
        ),
    )
    parser.add_argument(
        "--fid-num-samples",
        type=int,
        default=50000,
        help="Class-balanced sample count for periodic FID.",
    )
    parser.add_argument(
        "--fid-batch-size",
        type=int,
        default=256,
        help="Batch size for periodic FID sample generation.",
    )
    return parser


def build_optimizer(
    *,
    model: nn.Module,
    lr: float,
) -> tuple[torch.optim.Optimizer, list[str]]:
    """Build the DecoupledAdamW optimizer and its LR-log group name."""
    opt = DecoupledAdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )
    return opt, ["lr"]


def _eval_class_ids(device: torch.device) -> Tensor:
    """Class labels for the fixed eval grid: NUM_SAMPLES_PER_CLASS of each class."""
    return torch.arange(NUM_CLASSES, device=device).repeat_interleave(NUM_SAMPLES_PER_CLASS)


def train(args: argparse.Namespace) -> None:
    """Run training from parsed command line arguments."""
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

    disable_cudnn_sdp_on_blackwell()
    disable_torchscript_gpu_fuser_on_blackwell()

    seed_everything(int(args.seed) + rank)
    if args.precision == "tf32":
        torch.set_float32_matmul_precision("high")

    use_precomputed = args.precomputed_dino_features is not None

    if args.fid_every_epochs > 0 and importlib.util.find_spec("cleanfid") is None:
        raise ImportError(
            "clean-fid is required for --fid-every-epochs > 0. Install it "
            "with `uv sync --group eval`."
        )

    loader = build_cifar10_dataloader(
        dataset_id=args.dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        image_size=IMAGE_SIZE,
        pin_memory=device.type == "cuda",
        rank=rank,
        world_size=world_size,
        include_sample_id=use_precomputed,
    )

    raw_model = build_cifar10_model(
        n_oscillators=int(args.n_oscillators),
        n_conditional_oscillators=int(args.n_conditional_oscillators),
        class_dropout_prob=float(args.class_dropout_prob),
        num_steps=int(args.num_steps),
        decoder_in_channels=args.decoder_in_channels,
        parameterization=str(args.parameterization),
        relativization=str(args.relativization),
        encoding=str(args.encoding),
        solver=str(args.solver),
    ).to(device)
    if args.freeze_dynamics:
        for param in raw_model.dynamics.parameters():
            param.requires_grad_(False)  # noqa: FBT003
    config = vars(args).copy()
    dino = DINOFeatureExtractor().to(device)
    # Compile DINO's forward. Input shape is fixed by IMAGE_SIZE and
    # feature_batch_size, so Inductor compiles once and reuses. Backbone
    # params are frozen, so no recompilation on train/eval toggles.
    # dynamic=False: a smaller final chunk (batch not a multiple of
    # feature_batch_size) would otherwise trigger a dynamic-shape recompile,
    # and torch 2.11's meta kernel for _upsample_bicubic2d_aa_backward asserts
    # on a symbolic output size. Static per-shape compiles sidestep that.
    dino = torch.compile(dino, dynamic=False)

    dino_real_bank: Tensor | None = None
    if use_precomputed:
        bank_path = Path(args.precomputed_dino_features)
        if not bank_path.is_file():
            raise FileNotFoundError(f"Precomputed bank not found: {bank_path}")
        payload = torch.load(bank_path, map_location=device, weights_only=False)
        bank_image_size = int(payload.get("image_size", IMAGE_SIZE))
        if bank_image_size != IMAGE_SIZE:
            raise ValueError(
                f"Bank image_size {bank_image_size} != training {IMAGE_SIZE}.",
            )
        dino_real_bank = payload["views"].to(device=device)
        n_bank = int(dino_real_bank.shape[0])
        n_ds = len(loader.dataset)
        if n_bank != n_ds:
            raise ValueError(
                f"Bank length {n_bank} != dataset length {n_ds} for {bank_path}.",
            )

    start_epoch = 0
    global_step = 0
    resume_state: dict[str, Any] | None = None
    if args.resume is not None:
        # weights_only=False: checkpoints hold optimizer/scheduler state, not just tensors.
        resume_state = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(resume_state["model"])
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        global_step = int(resume_state.get("global_step", 0))

    model: nn.Module = (
        DistributedDataParallel(raw_model, device_ids=[local_rank]) if distributed else raw_model
    )

    optimizer, lr_group_names = build_optimizer(
        model=model,
        lr=float(args.lr),
    )
    steps_per_epoch = len(loader)
    total_steps = int(args.epochs) * steps_per_epoch
    scheduler = make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_fraction=WARMUP_FRACTION,
    )
    scaler = torch.amp.GradScaler("cuda") if args.precision == "fp16" else None

    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])

    checkpoint_dir = Path(args.checkpoint_dir)
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Persist the optimizer hyperparams that aren't argparse args so they show
    # up in the saved config / W&B run config.
    config["weight_decay"] = WEIGHT_DECAY
    config["beta1"] = BETA1
    config["beta2"] = BETA2

    wandb_run = None
    wandb = None
    if is_main and args.wandb_project is not None:
        try:
            import wandb  # lazy: optional dep, installed via `uv sync --group logging`
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
        # FID is logged at end-of-epoch cadence; declaring epoch as its
        # step_metric makes the W&B chart x-axis read in epochs rather than
        # in (sparse, hard-to-eyeball) global step counts.
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("fid", step_metric="epoch")

    def get_train_state(epoch_value: int) -> dict[str, Any]:
        # Save unwrapped state_dict so single-GPU inference can load DDP-trained checkpoints.
        return {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch_value,
            "global_step": global_step,
            "config": config,
        }

    eval_class_ids = _eval_class_ids(device)

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
            track_sample_ids=use_precomputed,
        )
        if use_queue
        else None
    )

    # Shared FID labels: built once and broadcast so every rank shards an
    # identical class-balanced set (per-rank seeding would otherwise diverge).
    fid_all_class_ids: Tensor | None = None
    if args.fid_every_epochs > 0:
        from metrics import _class_balanced_ids

        fid_all_class_ids = _class_balanced_ids(int(args.fid_num_samples), NUM_CLASSES, device)
        if distributed:
            dist.broadcast(fid_all_class_ids, src=0)

    for epoch in range(start_epoch, int(args.epochs)):
        if distributed:
            loader.sampler.set_epoch(epoch)
        model.train()
        dino.train()
        progress_iter = enumerate(loader)
        progress = (
            tqdm(
                progress_iter,
                total=steps_per_epoch,
                desc=f"epoch {epoch + 1}/{args.epochs}",
            )
            if is_main
            else progress_iter
        )
        for _step, batch in progress:
            x_real = batch["data"].to(device=device, non_blocking=True)
            class_id_real = batch["class_id"].to(
                device=device,
                non_blocking=True,
            )
            sample_id_real = (
                batch["sample_id"].to(device=device, non_blocking=True) if use_precomputed else None
            )
            # Gens use the same class labels as the real batch so per-class
            # positives/negatives are aligned in the conditional drift loss.
            class_id_gen = class_id_real

            x_real_pos: Tensor | None = None
            class_id_pos: Tensor | None = None
            sample_id_pos: Tensor | None = None
            queue_ready = True
            if use_queue:
                queue.push(x_real.detach(), class_id_real, sample_id_real)
                gen_classes = torch.unique(class_id_gen)
                ready_local = bool(queue.ready_mask(num_pos)[gen_classes].all())
                # All ranks agree on readiness so no rank starts training
                # against queue draws while another is still warming up.
                if distributed:
                    ready_t = torch.tensor(int(ready_local), device=device, dtype=torch.int32)
                    dist.all_reduce(ready_t, op=dist.ReduceOp.MIN)
                    queue_ready = bool(ready_t)
                else:
                    queue_ready = ready_local
                if queue_ready:
                    x_real_pos, class_id_pos, sample_id_pos = queue.draw(
                        gen_classes,
                        num_pos=num_pos,
                    )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                x_gen = model(class_id_gen)
                if queue_ready:
                    precomputed_pos: list[Tensor] | None = None
                    precomputed_gamma: list[Tensor] | None = None
                    if use_precomputed and float(args.dino_weight) != 0.0:
                        # Pos views come from the queue's sample_ids when in
                        # queue mode, otherwise from the current batch.
                        pos_sid = sample_id_pos if sample_id_pos is not None else sample_id_real
                        precomputed_pos = gather_precomputed_dino_views(
                            dino_real_bank,
                            pos_sid,
                        )
                        if use_queue:
                            # γ-mix views always come from the current batch.
                            precomputed_gamma = gather_precomputed_dino_views(
                                dino_real_bank,
                                sample_id_real,
                            )
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
                        precomputed_pos_views=precomputed_pos,
                        precomputed_gamma_views=precomputed_gamma,
                    )
                else:
                    # Queue warmup: graph-connected zero keeps DDP all-reduce firing.
                    loss = (x_gen * 0.0).sum()
                    metrics = {"loss/total": loss.detach()}
                metrics["queue_ready"] = torch.as_tensor(
                    float(queue_ready),
                    device=device,
                    dtype=x_gen.dtype,
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
            scheduler.step()
            global_step += 1

            if is_main and global_step % LOG_EVERY == 0:
                log_metrics = {name: float(value.detach().cpu()) for name, value in metrics.items()}
                for name, value in zip(
                    lr_group_names,
                    scheduler.get_last_lr(),
                    strict=True,
                ):
                    log_metrics[name] = float(value)
                log_metrics["grad_norm"] = float(grad_norm.detach().cpu())
                progress.set_postfix(log_metrics)
                if wandb_run is not None:
                    wandb_run.log(log_metrics, step=global_step)

        epoch_num = epoch + 1
        fid_firing = args.fid_every_epochs > 0 and epoch_num % args.fid_every_epochs == 0
        should_sample = (
            epoch_num in EARLY_SAMPLE_EPOCHS or epoch_num % SAMPLE_EVERY == 0 or fid_firing
        )
        if is_main and should_sample:
            samples = raw_model.sample(eval_class_ids)
            sample_path = checkpoint_dir / "samples" / f"epoch_{epoch_num:04d}.png"
            save_sample_grid(samples, sample_path, image_size=IMAGE_SIZE)
            if wandb_run is not None:
                wandb_run.log(
                    {"samples": wandb.Image(str(sample_path))},
                    step=global_step,
                )
        # Under DDP, generation is sharded across ranks into one shared dir
        # (rank-unique filenames); rank 0 scores the combined samples while the
        # others get nan. All ranks must call compute_fid so the barrier inside
        # it stays collective. Single-process uses compute_fid's tempdir.
        if fid_firing:
            from metrics import compute_fid

            fid_dir = (
                checkpoint_dir / "fid_samples" / f"epoch_{epoch_num:04d}" if distributed else None
            )
            raw_model.eval()
            fid_value = compute_fid(
                raw_model,
                num_samples=int(args.fid_num_samples),
                num_classes=NUM_CLASSES,
                batch_size=int(args.fid_batch_size),
                device=device,
                gen_class_ids=fid_all_class_ids,
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
                    wandb_run.log(
                        {"fid": fid_value, "epoch": epoch_num},
                        step=global_step,
                    )
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
    """CLI entry point."""
    parser = build_parser()
    train(parser.parse_args())


if __name__ == "__main__":
    main()
