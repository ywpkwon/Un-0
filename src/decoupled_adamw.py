"""AdamW with weight decay decoupled from the learning rate."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import math

import torch
from torch.optim import AdamW


class DecoupledAdamW(AdamW):
    """AdamW where per-step weight decay is `(lr/initial_lr) * weight_decay`.

    `torch.optim.AdamW` applies a per-step decay of `lr * weight_decay`, so the
    effective decay scales with the LR schedule's *absolute* magnitude.
    `DecoupledAdamW` divides by `initial_lr`, so the effective decay tracks the
    *relative* schedule shape (1.0 at peak LR, ramping with warmup/decay) and
    does not change if you rescale all LRs by a constant.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-5,
    ) -> None:
        """Initialize the optimizer and snapshot per-group `initial_lr`."""
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=False,
        )
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore state, keeping each group's construction-time `initial_lr`.

        A checkpoint written by a plain `AdamW` carries no `initial_lr`, and
        loading one would otherwise leave the decay factor to be seeded from
        the decayed schedule lr instead of the peak. The `__init__` snapshot is
        preserved for any group the checkpoint does not supply one for.
        """
        initial_lrs = [group["initial_lr"] for group in self.param_groups]
        super().load_state_dict(state_dict)
        for group, initial_lr in zip(
            self.param_groups,
            initial_lrs,
            strict=True,
        ):
            group.setdefault("initial_lr", initial_lr)

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Callable[[], float] | None = None,
    ) -> float | None:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            # __init__ and load_state_dict both seed initial_lr; this only
            # fires for a group added later via add_param_group, whose lr is
            # still its peak at that point.
            if "initial_lr" not in group:
                group["initial_lr"] = lr
            initial_lr = group["initial_lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "DecoupledAdamW does not support sparse gradients",
                    )

                state = self.state[p]
                if "step" not in state:
                    state["step"] = torch.zeros(
                        (),
                        dtype=torch.float,
                        device=p.device,
                    )
                    state["exp_avg"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )
                state["step"] += 1
                step_count = float(state["step"].item())

                if weight_decay != 0:
                    decay_factor = (lr / initial_lr) if initial_lr else 1.0
                    p.mul_(1.0 - decay_factor * weight_decay)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step_count
                bias_correction2 = 1.0 - beta2**step_count
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
