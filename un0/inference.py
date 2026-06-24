"""Generate class-conditional image samples from a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from un0.common import resolve_device, save_sample_grid, seed_everything
from un0.data import NUM_CLASSES
from un0.model import (
    PRETRAINED_NAMES,
    ConditionalImplicitKuramotoGenerator,
    build_cifar10_model,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Path to a local .pt checkpoint.")
    source.add_argument(
        "--pretrained",
        choices=PRETRAINED_NAMES,
        help="Load released weights from Hugging Face by name.",
    )
    parser.add_argument("--output", default="samples/grid.png")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=list(range(NUM_CLASSES)),
        help="Class ids to sample from (default: all 10).",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=10,
        help="How many images to generate per class.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def generate(args: argparse.Namespace) -> None:
    """Generate and save samples."""
    seed_everything(int(args.seed))
    device = resolve_device("auto")
    if args.pretrained is not None:
        model = ConditionalImplicitKuramotoGenerator.from_pretrained(args.pretrained, device=device)
    else:
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

    classes = torch.tensor(args.classes, device=device, dtype=torch.long)
    class_ids = classes.repeat_interleave(int(args.samples_per_class))
    samples = model.sample(class_ids)
    save_sample_grid(samples, Path(args.output), nrow=int(args.samples_per_class))


def main() -> None:
    parser = build_parser()
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
