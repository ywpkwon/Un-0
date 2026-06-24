from __future__ import annotations

import hashlib

import pytest
import torch

from un0.model import (
    ConditionalImplicitKuramotoGenerator,
    ConditionalKuramotoDynamics,
    ReadoutTransform,
    ResizeConvDecoder,
    build_cifar10_model,
    build_imagenet64_model,
    prepare_class_ids_for_generation,
)


def _fingerprint(state_dict) -> str:
    parts = []
    for name in sorted(state_dict):
        p = state_dict[name].double()
        parts.append(f"{name}|{tuple(p.shape)}|{float(p.sum()):.6f}|{float((p * p).sum()):.6f}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def test_readout_transform_sin_cos_ref_shape() -> None:
    """The default readout doubles oscillator count with sin/cos."""
    phases = torch.randn(3, 4)
    readout = ReadoutTransform(encoding="sin_cos", relativization="ref_oscillator")

    features = readout(phases)

    assert features.shape == (3, 8)


def test_conditional_generator_produces_flat_images_and_gradients() -> None:
    """A tiny conditional generator should produce flat images and support backprop."""
    dynamics = ConditionalKuramotoDynamics(
        n_oscillators=4,
        n_conditional_oscillators=2,
        num_classes=3,
    )
    readout = ReadoutTransform(encoding="sin_cos", relativization="ref_oscillator")
    decoder = ResizeConvDecoder(
        feature_dim=8,
        output_dim=16,
        in_channels=2,
        in_height=2,
        in_width=2,
        out_channels=1,
        num_upsamples=1,
    )
    model = ConditionalImplicitKuramotoGenerator(
        dynamics=dynamics,
        readout=readout,
        decoder=decoder,
        num_steps=2,
    )

    class_id = torch.tensor([0, 1, 2, 0, 1])
    samples = model(class_id)
    loss = samples.square().mean()
    loss.backward()

    assert samples.shape == (5, 16)
    assert dynamics.K_drive.grad is not None
    assert dynamics.omega.grad is not None


def test_class_dropout_zeroes_drive_for_dropped_samples() -> None:
    """With class_dropout_prob=1.0 and train mode, drive is zeroed for all samples."""
    dynamics = ConditionalKuramotoDynamics(
        n_oscillators=4,
        n_conditional_oscillators=2,
        num_classes=3,
    )
    readout = ReadoutTransform(encoding="sin_cos", relativization="ref_oscillator")
    decoder = ResizeConvDecoder(
        feature_dim=8,
        output_dim=16,
        in_channels=2,
        in_height=2,
        in_width=2,
        out_channels=1,
        num_upsamples=1,
    )
    model = ConditionalImplicitKuramotoGenerator(
        dynamics=dynamics,
        readout=readout,
        decoder=decoder,
        class_dropout_prob=1.0,
        num_steps=2,
    )
    model.train()

    class_id = torch.tensor([0, 1, 2])
    drive = model._class_drive(class_id)

    assert torch.allclose(drive, torch.zeros_like(drive))


def test_build_cifar10_model_accepts_custom_dimensions() -> None:
    """The CIFAR model builder should support larger conditional models."""
    model = build_cifar10_model(
        n_oscillators=8,
        n_conditional_oscillators=3,
        class_dropout_prob=0.25,
        num_steps=4,
    )

    assert model.dynamics.n == 8
    assert model.dynamics.n_cond == 3
    assert model.class_dropout_prob == 0.25
    assert model.num_steps == 4
    assert model.decoder.feature_dim == 16
    assert model.decoder.in_channels == 1


def test_build_cifar10_model_defaults_to_release_model() -> None:
    """Default builder arguments should preserve the release model."""
    model = build_cifar10_model()

    assert model.dynamics.n == 4096
    assert model.dynamics.n_cond == 8
    assert model.class_dropout_prob == 0.1
    assert model.num_steps == 25
    assert model.decoder.feature_dim == 8192
    assert model.decoder.in_channels == 512


def test_standard_and_mup_have_matching_init_forward_output() -> None:
    """Standard and MuP parameterizations are forward-equivalent at init.

    Standard stores K at ``init / sqrt(n)`` and forwards with scale 1.0;
    MuP stores K at ``init`` and forwards with scale ``1/sqrt(n)``. With
    matched RNG the velocity should be identical up to float noise.
    """
    n = 16
    n_cond = 4
    num_classes = 3
    batch = 2

    torch.manual_seed(0)
    standard = ConditionalKuramotoDynamics(
        n_oscillators=n,
        n_conditional_oscillators=n_cond,
        num_classes=num_classes,
        parameterization="standard",
    )
    torch.manual_seed(0)
    mup = ConditionalKuramotoDynamics(
        n_oscillators=n,
        n_conditional_oscillators=n_cond,
        num_classes=num_classes,
        parameterization="mup",
    )

    state = torch.randn(batch, n + n_cond)
    drive_idx = torch.tensor([0, 1])
    drive_std = standard.K_drive[drive_idx]
    drive_mup = mup.K_drive[drive_idx]
    out_std = standard(state, torch.tensor(0.0), drive_std)
    out_mup = mup(state, torch.tensor(0.0), drive_mup)

    torch.testing.assert_close(out_std, out_mup, rtol=1e-5, atol=1e-5)
    assert standard.K.std().item() < mup.K.std().item()


def _tiny_generator(*, num_steps: int, solver: str = "rk4"):
    dynamics = ConditionalKuramotoDynamics(
        n_oscillators=4,
        n_conditional_oscillators=2,
        num_classes=3,
    )
    readout = ReadoutTransform(encoding="sin_cos", relativization="ref_oscillator")
    decoder = ResizeConvDecoder(
        feature_dim=8,
        output_dim=16,
        in_channels=2,
        in_height=2,
        in_width=2,
        out_channels=1,
        num_upsamples=1,
    )
    model = ConditionalImplicitKuramotoGenerator(
        dynamics=dynamics,
        readout=readout,
        decoder=decoder,
        num_steps=num_steps,
        solver=solver,
    )
    return model, dynamics, decoder


def test_decoder_only_skips_dynamics() -> None:
    """num_steps=0 decodes the random initial phases without touching dynamics."""
    model, dynamics, decoder = _tiny_generator(num_steps=0)

    samples = model(torch.tensor([0, 1, 2]))
    samples.square().mean().backward()

    assert samples.shape == (3, 16)
    assert dynamics.K.grad is None
    assert dynamics.omega.grad is None
    assert next(decoder.parameters()).grad is not None


def test_euler_solver_runs_and_trains_dynamics() -> None:
    """A single Euler step flows gradients back into the dynamics."""
    model, dynamics, _ = _tiny_generator(num_steps=1, solver="euler")

    samples = model(torch.tensor([0, 1, 2]))
    samples.square().mean().backward()

    assert samples.shape == (3, 16)
    assert dynamics.omega.grad is not None


def test_frozen_dynamics_reservoir_gets_no_gradient() -> None:
    """Freezing the dynamics leaves only the decoder trainable (reservoir mode)."""
    model, dynamics, decoder = _tiny_generator(num_steps=2)
    for param in dynamics.parameters():
        param.requires_grad_(False)

    samples = model(torch.tensor([0, 1, 2]))
    samples.square().mean().backward()

    assert all(param.grad is None for param in dynamics.parameters())
    assert next(decoder.parameters()).grad is not None


def test_generator_defaults_to_rk4_integration() -> None:
    """The generator preserves rk4 as the default solver."""
    dynamics = ConditionalKuramotoDynamics(
        n_oscillators=4,
        n_conditional_oscillators=2,
        num_classes=3,
    )
    readout = ReadoutTransform(encoding="sin_cos", relativization="ref_oscillator")
    decoder = ResizeConvDecoder(
        feature_dim=8,
        output_dim=16,
        in_channels=2,
        in_height=2,
        in_width=2,
        out_channels=1,
        num_upsamples=1,
    )
    model = ConditionalImplicitKuramotoGenerator(
        dynamics=dynamics,
        readout=readout,
        decoder=decoder,
        num_steps=2,
    )
    assert model.solver == "rk4"


def test_generator_euler_integration_runs_and_backprops() -> None:
    """Euler integration (the ImageNet-64 setting) produces images and gradients."""
    dynamics = ConditionalKuramotoDynamics(
        n_oscillators=4,
        n_conditional_oscillators=1,
        num_classes=3,
    )
    readout = ReadoutTransform(encoding="sin_cos", relativization="ref_oscillator")
    decoder = ResizeConvDecoder(
        feature_dim=8,
        output_dim=16,
        in_channels=2,
        in_height=2,
        in_width=2,
        out_channels=1,
        num_upsamples=1,
    )
    model = ConditionalImplicitKuramotoGenerator(
        dynamics=dynamics,
        readout=readout,
        decoder=decoder,
        num_steps=10,
        solver="euler",
    )
    class_id = torch.tensor([0, 1, 2])
    samples = model(class_id)
    samples.square().mean().backward()
    assert samples.shape == (3, 16)
    assert dynamics.K_drive.grad is not None


def test_build_imagenet64_model_defaults() -> None:
    """ImageNet-64 builder: 1000 classes, 1 cond oscillator, 10-step euler, mup,
    64x64 decoder."""
    model = build_imagenet64_model(n_oscillators=16, n_conditional_oscillators=1)
    assert model.dynamics.num_classes == 1000
    assert model.dynamics.n_cond == 1
    assert model.dynamics.parameterization == "mup"
    assert model.num_steps == 10
    assert model.solver == "euler"
    assert model.class_dropout_prob == 0.1
    assert model.decoder.output_dim == 3 * 64 * 64
    assert model.decoder.in_height == 4
    assert model.decoder.in_width == 4


def test_build_imagenet64_model_uses_unit_output_gain() -> None:
    """ImageNet-64 decoder uses init_output_gain=1.0 (reference), not 0.5."""
    cifar = build_cifar10_model(
        n_oscillators=16,
        n_conditional_oscillators=1,
        num_steps=2,
    )
    imagenet = build_imagenet64_model(
        n_oscillators=16,
        n_conditional_oscillators=1,
    )
    ratio = (imagenet.decoder.to_output.weight.std() / cifar.decoder.to_output.weight.std()).item()
    assert ratio == pytest.approx(2.0, rel=0.15)


def test_prepare_class_ids_for_generation_picks_exactly_num_classes_per_step() -> None:
    """Generation labels cover exactly `num_classes_per_step` distinct classes."""
    ids = prepare_class_ids_for_generation(
        num_samples=2048,
        num_classes_per_step=64,
        num_total_classes=1000,
    )
    assert ids.shape == (2048,)
    assert torch.unique(ids).numel() == 64
    assert int(ids.min()) >= 0
    assert int(ids.max()) < 1000


def test_prepare_class_ids_for_generation_balances_with_round_robin_remainder() -> None:
    """Each class gets base count; the first `remainder` classes get one extra."""
    ids = prepare_class_ids_for_generation(
        num_samples=10,
        num_classes_per_step=4,
        num_total_classes=50,
    )
    assert ids.shape == (10,)
    counts = sorted(torch.bincount(ids).tolist(), reverse=True)
    assert counts[:4] == [3, 3, 2, 2]


def test_prepare_class_ids_for_generation_is_deterministic_with_generator() -> None:
    """A seeded generator yields reproducible class selection."""
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    ids1 = prepare_class_ids_for_generation(
        num_samples=128,
        num_classes_per_step=8,
        num_total_classes=1000,
        generator=g1,
    )
    ids2 = prepare_class_ids_for_generation(
        num_samples=128,
        num_classes_per_step=8,
        num_total_classes=1000,
        generator=g2,
    )
    assert torch.equal(ids1, ids2)


def test_build_imagenet64_model_param_fingerprint_matches_reference(monkeypatch) -> None:
    """Seeded init reproduces the imagenet-repro builder bit-for-bit (params)."""
    monkeypatch.setattr(torch, "compile", lambda m, *a, **k: m)
    torch.manual_seed(0)
    model = build_imagenet64_model(n_oscillators=16, n_conditional_oscillators=1)
    assert _fingerprint(model.state_dict()) == (
        "53a0ba23f3f53d190acef1636abf10612ae45d30a2b7c45a3e0a0830d397703b"
    )


def test_build_cifar10_model_param_fingerprint_unchanged(monkeypatch) -> None:
    """CIFAR builder init is unchanged from origin/main (no-regression guard)."""
    monkeypatch.setattr(torch, "compile", lambda m, *a, **k: m)
    torch.manual_seed(0)
    model = build_cifar10_model(
        n_oscillators=16,
        n_conditional_oscillators=2,
        num_steps=2,
    )
    assert _fingerprint(model.state_dict()) == (
        "532639f04f2480b1bea9316682f9b6dc8220a27173a1864fc0b9bd3b079b9bf5"
    )


def test_sample_images_returns_chw_in_unit_range() -> None:
    """sample_images reshapes flat samples to (B, 3, H, W) in [0, 1]."""
    model = build_cifar10_model(n_oscillators=8, n_conditional_oscillators=2)
    class_ids = torch.tensor([0, 1, 2])

    flat = model.sample(class_ids)
    images = model.sample_images(class_ids)

    assert flat.shape == (3, 3 * 32 * 32)
    assert images.shape == (3, 3, 32, 32)
    assert float(images.min()) >= 0.0
    assert float(images.max()) <= 1.0

    # sample_images must be exactly the [-1, 1] -> [0, 1] mapping of sample()
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    flat_seeded = model.sample(class_ids, generator=g1)
    images_seeded = model.sample_images(class_ids, generator=g2)
    expected = ((flat_seeded.reshape(-1, 3, 32, 32) + 1.0) * 0.5).clamp(0.0, 1.0)
    torch.testing.assert_close(images_seeded, expected)
