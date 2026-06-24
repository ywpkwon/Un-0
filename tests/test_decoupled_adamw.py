from __future__ import annotations

import torch

from un0.decoupled_adamw import DecoupledAdamW


def test_initial_lr_snapshot_per_group() -> None:
    p = torch.zeros(4, requires_grad=True)
    opt = DecoupledAdamW([p], lr=2.38e-3, weight_decay=1e-3)
    assert opt.param_groups[0]["initial_lr"] == 2.38e-3


def test_decay_factor_scales_with_relative_lr() -> None:
    # At lr == initial_lr, per-step decay equals weight_decay (no scaling).
    # At lr == 0.5 * initial_lr, decay should be 0.5 * weight_decay.
    initial_lr = 1e-3
    weight_decay = 1e-3

    p_full = torch.ones(4, requires_grad=True)
    p_full.grad = torch.zeros_like(p_full)
    opt_full = DecoupledAdamW([p_full], lr=initial_lr, weight_decay=weight_decay)
    opt_full.step()

    p_half = torch.ones(4, requires_grad=True)
    p_half.grad = torch.zeros_like(p_half)
    opt_half = DecoupledAdamW([p_half], lr=initial_lr, weight_decay=weight_decay)
    # Mid-schedule: lr drops to half, but initial_lr stays the same.
    opt_half.param_groups[0]["lr"] = 0.5 * initial_lr
    opt_half.step()

    expected_full = 1.0 - 1.0 * weight_decay  # decay_factor = lr/initial_lr = 1
    expected_half = 1.0 - 0.5 * weight_decay  # decay_factor = lr/initial_lr = 0.5
    torch.testing.assert_close(p_full.detach(), torch.full_like(p_full, expected_full))
    torch.testing.assert_close(p_half.detach(), torch.full_like(p_half, expected_half))


def test_step_count_advances_and_state_persists() -> None:
    p = torch.zeros(4, requires_grad=True)
    opt = DecoupledAdamW([p], lr=1e-3, weight_decay=1e-3)
    p.grad = torch.ones_like(p)
    opt.step()
    opt.step()
    assert int(opt.state[p]["step"].item()) == 2
    assert "exp_avg" in opt.state[p]
    assert "exp_avg_sq" in opt.state[p]


def test_load_state_dict_without_initial_lr_keeps_construction_value() -> None:
    # A checkpoint from plain AdamW carries no `initial_lr`; load_state_dict
    # must keep the construction-time peak so the decay factor stays correct.
    p1 = torch.zeros(4, requires_grad=True)
    opt1 = DecoupledAdamW([p1], lr=1e-3, weight_decay=1e-3)
    p1.grad = torch.ones_like(p1)
    opt1.step()
    state = opt1.state_dict()
    for group in state["param_groups"]:
        group.pop("initial_lr", None)

    p2 = torch.zeros(4, requires_grad=True)
    opt2 = DecoupledAdamW([p2], lr=5e-4, weight_decay=1e-3)
    opt2.load_state_dict(state)
    p2.grad = torch.ones_like(p2)
    opt2.step()
    # lr comes from the checkpoint, but initial_lr is preserved from __init__,
    # so the decoupled decay factor tracks the true peak, not the loaded lr.
    assert opt2.param_groups[0]["lr"] == 1e-3
    assert opt2.param_groups[0]["initial_lr"] == 5e-4


def test_load_state_dict_native_checkpoint_restores_initial_lr() -> None:
    # A checkpoint from DecoupledAdamW carries initial_lr; it is restored as-is.
    p1 = torch.zeros(4, requires_grad=True)
    opt1 = DecoupledAdamW([p1], lr=5e-4, weight_decay=1e-3)
    opt1.param_groups[0]["lr"] = 2.5e-4  # mid-schedule decay
    p1.grad = torch.ones_like(p1)
    opt1.step()
    state = opt1.state_dict()

    p2 = torch.zeros(4, requires_grad=True)
    opt2 = DecoupledAdamW([p2], lr=1e-3, weight_decay=1e-3)
    opt2.load_state_dict(state)
    assert opt2.param_groups[0]["initial_lr"] == 5e-4
    assert opt2.param_groups[0]["lr"] == 2.5e-4
