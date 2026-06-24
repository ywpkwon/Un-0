"""Shared helpers for the training, evaluation, and inference entry points.

Plain functions with no model- or dataset-specific assumptions, kept here so the
entry-point scripts do not import from one another (an evaluation tool importing
from a training script reads backwards). `save_sample_grid` defaults to the
CIFAR-10 layout (`image_size=32`, `nrow=10`); the ImageNet path passes its own
`image_size` and uses a 10-class grid slice.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
import random
from typing import Literal

import torch
from torch import Tensor
from torchvision.utils import save_image

Precision = Literal["fp32", "tf32", "bf16", "fp16"]


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch RNGs."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    """Resolve `auto` to CUDA when available."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# Blackwell (sm_100) and newer; the bundled cuDNN 9.x has no valid SDPA plan.
_BLACKWELL_SM_MAJOR = 10


def disable_cudnn_sdp_on_blackwell() -> None:
    """Force flash SDPA on Blackwell+ GPUs, where cuDNN attention is broken.

    On Blackwell (sm_100+, e.g. B300 reports sm_103), the cuDNN 9.x bundled with
    torch 2.11+cu128 has no valid SDPA execution plan, so the compiled DINO
    attention crashes with "No valid execution plans built". Disabling the cuDNN
    SDPA backend dispatches to flash instead. Gated on compute capability so
    pre-Blackwell GPUs (H200, A100) keep cuDNN attention.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= _BLACKWELL_SM_MAJOR:
        torch.backends.cuda.enable_cudnn_sdp(False)  # noqa: FBT003


def disable_torchscript_gpu_fuser_on_blackwell() -> None:
    """Disable the TorchScript GPU fusers on Blackwell+, where NVRTC rejects sm_103.

    clean-fid's FID runs a TorchScript InceptionV3, whose GPU fusers JIT-compile
    fused CUDA kernels through NVRTC. On Blackwell (sm_100+, e.g. B300 reports
    sm_103) NVRTC rejects the device arch with "invalid value for
    --gpu-architecture (compute_103)", crashing FID scoring in both eval.py and
    the in-training FID (train_cifar10.py / train_imagenet.py). Disabling the
    fusers makes Inception run eager.

    TorchScript has shipped three GPU fuser backends across torch versions; we
    turn off all of them so the workaround holds regardless of which is active:
      - the legacy fuser (``_jit_override_can_fuse_on_gpu``);
      - the TensorExpr/NNC fuser (the default in torch 2.x), reached via the
        profiling executor that builds the shape-specialized graphs it fuses --
        hence also disabling the profiling executor/mode; and
      - nvfuser (``_jit_set_nvfuser_enabled``), removed from core torch in 2.x,
        so that symbol may not exist -- the try/except tolerates its absence.

    These flags govern TorchScript only; torch.compile (Inductor) is a separate
    path and is unaffected, so this is safe to set once at process start. Gated on
    compute capability so pre-Blackwell GPUs (H200, B200, A100), where the fusers
    work, keep them.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= _BLACKWELL_SM_MAJOR:
        torch._C._jit_override_can_fuse_on_gpu(False)  # noqa: SLF001, FBT003
        torch._C._jit_set_profiling_executor(False)  # noqa: SLF001, FBT003
        torch._C._jit_set_profiling_mode(False)  # noqa: SLF001, FBT003
        try:
            torch._C._jit_set_texpr_fuser_enabled(False)  # noqa: SLF001, FBT003
            torch._C._jit_set_nvfuser_enabled(False)  # noqa: SLF001, FBT003
        except (AttributeError, RuntimeError):
            pass


def save_sample_grid(
    samples: Tensor,
    path: str | Path,
    *,
    image_size: int = 32,
    nrow: int = 10,
) -> None:
    """Save flattened `[-1, 1]` generated samples as an image grid."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = samples.detach().cpu().reshape(samples.shape[0], 3, image_size, image_size)
    save_image(((images + 1.0) * 0.5).clamp(0.0, 1.0), output_path, nrow=nrow)


def linear_warmup_decay_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_fraction: float,
) -> float:
    """Return linear warmup then linear decay multiplier."""
    if total_steps <= 0:
        return 1.0
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    remaining = max(1, total_steps - warmup_steps)
    progress = float(step - warmup_steps) / float(remaining)
    return max(0.0, 1.0 - progress)


def autocast_context(device: torch.device, precision: Precision) -> AbstractContextManager:
    """Return the autocast context for the requested precision."""
    enabled = precision in ("bf16", "fp16")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.amp.autocast(device.type, enabled=enabled, dtype=dtype)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create the linear warmup then linear decay scheduler."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: linear_warmup_decay_multiplier(
            step,
            total_steps=total_steps,
            warmup_fraction=warmup_fraction,
        ),
    )
