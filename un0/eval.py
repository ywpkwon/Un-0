"""Evaluate a trained checkpoint: load model, generate samples, report FID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from un0.common import (
    disable_torchscript_gpu_fuser_on_blackwell,
    resolve_device,
    seed_everything,
)
from un0.data import NUM_CLASSES
from un0.metrics import compute_fid
from un0.model import build_cifar10_model

DEFAULT_NUM_SAMPLES = 50000
DEFAULT_BATCH_SIZE = 256


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory to keep the generated samples. Defaults to a tempdir.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON path. Writes {fid, checkpoint, num_samples, seed}.",
    )
    return parser


def evaluate(args: argparse.Namespace) -> float:
    """Load checkpoint, compute FID, return the scalar."""
    seed_everything(int(args.seed))
    device = resolve_device("auto")
    disable_torchscript_gpu_fuser_on_blackwell()

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state.get("config") or {}
    # Checkpoints without these config keys were trained with "mup" +
    # "ref_oscillator"; fall back to those so they load correctly.
    model = build_cifar10_model(
        n_oscillators=int(config.get("n_oscillators", 4096)),
        n_conditional_oscillators=int(config.get("n_conditional_oscillators", 8)),
        class_dropout_prob=float(config.get("class_dropout_prob", 0.1)),
        num_steps=int(config.get("num_steps", 25)),
        decoder_in_channels=(
            None
            if config.get("decoder_in_channels") is None
            else int(config["decoder_in_channels"])
        ),
        parameterization=str(config.get("parameterization", "mup")),
        relativization=str(config.get("relativization", "ref_oscillator")),
        encoding=str(config.get("encoding", "sin_cos")),
        solver=str(config.get("solver", "rk4")),
    ).to(device)
    model.load_state_dict(state["model"])
    if args.num_steps is not None:
        model.num_steps = int(args.num_steps)
    args.num_steps = int(model.num_steps)
    model.eval()

    return compute_fid(
        model,
        num_samples=int(args.num_samples),
        num_classes=NUM_CLASSES,
        batch_size=int(args.batch_size),
        device=device,
        image_dir=args.image_dir,
    )


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    fid = evaluate(args)
    print(f"FID: {fid:.4f}")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "fid": fid,
                    "checkpoint": str(args.checkpoint),
                    "num_samples": int(args.num_samples),
                    "num_steps": args.num_steps,
                    "seed": int(args.seed),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
