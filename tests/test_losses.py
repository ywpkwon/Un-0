from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from losses import (
    DINOFeatureExtractor,
    conditional_drift_loss,
    conditional_drift_loss_for_views,
    gather_precomputed_dino_views,
)


def _seeded_pair(
    n: int,
    d: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen_rng = torch.Generator().manual_seed(seed)
    x_real = torch.randn(n, d, generator=gen_rng)
    x_gen = torch.randn(n, d, generator=gen_rng, requires_grad=True)
    return x_real, x_gen


def test_conditional_drift_backprops_to_generated_samples() -> None:
    """The drift target is detached, but generated samples receive gradients."""
    x_real = torch.randn(6, 4)
    x_gen = torch.randn(6, 4, requires_grad=True)
    class_id_real = torch.tensor([0, 0, 1, 1, 2, 2])
    class_id_gen = torch.tensor([0, 0, 1, 1, 2, 2])

    loss = conditional_drift_loss_for_views(
        [(x_real, x_gen)],
        class_id_real,
        class_id_gen,
        temperatures=(0.1, 0.2),
        gamma=0.2,
    )
    loss.backward()

    assert loss.ndim == 0
    assert x_gen.grad is not None
    assert torch.isfinite(x_gen.grad).all()


def test_conditional_drift_handles_class_missing_from_batch() -> None:
    """Classes absent from either real or gen side should be skipped silently."""
    x_real = torch.randn(4, 3)
    x_gen = torch.randn(4, 3, requires_grad=True)
    # class 2 appears only in gen batch; should be skipped (no real positives).
    class_id_real = torch.tensor([0, 0, 1, 1])
    class_id_gen = torch.tensor([0, 1, 2, 0])

    loss = conditional_drift_loss_for_views(
        [(x_real, x_gen)],
        class_id_real,
        class_id_gen,
        temperatures=(0.2,),
        gamma=0.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert x_gen.grad is not None


def test_conditional_drift_loss_pixel_only() -> None:
    """CIFAR-10 composite loss runs pixel-only when dino_weight=0."""
    n = 8
    x_real = torch.randn(n, 12)
    x_gen = torch.randn(n, 12, requires_grad=True)
    class_id_real = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    class_id_gen = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    loss, metrics = conditional_drift_loss(
        x_real,
        x_gen,
        class_id_real,
        class_id_gen,
        dino=None,
        dino_weight=0.0,
        pixel_weight=0.1,
        gamma=0.2,
    )
    loss.backward()

    assert "loss/pixel_raw" in metrics
    assert "loss/total" in metrics
    assert x_gen.grad is not None


def test_queue_mode_matches_legacy_when_positives_equal_batch() -> None:
    """Queue mode with x_real_pos == x_real should equal the legacy call."""
    x_real, x_gen = _seeded_pair(6, 4, seed=7)
    class_id = torch.tensor([0, 0, 1, 1, 2, 2])

    torch.manual_seed(0)
    legacy = conditional_drift_loss_for_views(
        [(x_real, x_gen)],
        class_id,
        class_id,
        temperatures=(0.1, 0.2),
        gamma=0.2,
    )
    torch.manual_seed(0)
    queue_mode = conditional_drift_loss_for_views(
        [(x_real, x_gen)],
        class_id,
        class_id,
        gamma_views=[x_real],
        class_id_gamma=class_id,
        temperatures=(0.1, 0.2),
        gamma=0.2,
    )
    assert torch.allclose(legacy, queue_mode)


def test_queue_mode_with_separate_positives_backprops() -> None:
    """Queue-mode loss with disjoint positives still propagates to x_gen."""
    pos_rng = torch.Generator().manual_seed(11)
    x_real_batch = torch.randn(6, 4, generator=pos_rng)
    x_real_pos = torch.randn(9, 4, generator=pos_rng)
    x_gen = torch.randn(6, 4, requires_grad=True)
    class_id_real = torch.tensor([0, 0, 1, 1, 2, 2])
    class_id_pos = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    class_id_gen = class_id_real

    loss, metrics = conditional_drift_loss(
        x_real_batch,
        x_gen,
        class_id_real,
        class_id_gen,
        dino=None,
        dino_weight=0.0,
        pixel_weight=1.0,
        gamma=0.2,
        x_real_pos=x_real_pos,
        class_id_pos=class_id_pos,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert "loss/pixel_raw" in metrics
    assert x_gen.grad is not None
    assert torch.isfinite(x_gen.grad).all()


def test_gather_precomputed_dino_views_indexes_bank() -> None:
    """Bank rows are gathered by ``sample_id`` into per-view tensors."""
    bank = torch.randn(10, 3, 8)
    sample_ids = torch.tensor([1, 3, 5])
    views = gather_precomputed_dino_views(bank, sample_ids)
    assert len(views) == 3
    assert views[0].shape == (3, 8)
    assert torch.allclose(views[0], bank[sample_ids, 0])
    assert torch.allclose(views[2], bank[sample_ids, 2])


def test_cifar10_precomputed_skips_live_dino_on_reals() -> None:
    """Precomputed positives: ``extract_feature_views`` runs on gens only."""
    num_views = 3
    feat_dim = 8
    n = 4
    bank = torch.randn(10, num_views, feat_dim)
    sample_ids = torch.tensor([0, 1, 2, 3])
    x_real = torch.randn(n, 12)
    x_gen = torch.randn(n, 12, requires_grad=True)
    class_id = torch.tensor([0, 0, 1, 1])
    precomputed = gather_precomputed_dino_views(bank, sample_ids)
    extract_calls: list[int] = []

    def fake_extract(
        _extractor: object,
        x_flat: torch.Tensor,
        **_: object,
    ) -> list[torch.Tensor]:
        extract_calls.append(int(x_flat.shape[0]))
        base = x_flat[:, :feat_dim]
        return [base for _ in range(num_views)]

    with patch("losses.extract_feature_views", side_effect=fake_extract):
        loss, metrics = conditional_drift_loss(
            x_real,
            x_gen,
            class_id,
            class_id,
            dino=MagicMock(),
            dino_weight=1.0,
            pixel_weight=0.0,
            precomputed_pos_views=precomputed,
        )
    loss.backward()
    assert extract_calls == [n]
    assert "loss/dino_raw" in metrics
    assert x_gen.grad is not None


def test_precomputed_with_queue_runs_dino_only_on_gens() -> None:
    """Queue+precomputed: pos and γ-mix views come from bank; only gens go live."""
    num_views = 3
    feat_dim = 8
    n = 4
    bank = torch.randn(10, num_views, feat_dim)
    x_real = torch.randn(n, 12)
    x_gen = torch.randn(n, 12, requires_grad=True)
    class_id_real = torch.tensor([0, 0, 1, 1])
    class_id_pos = torch.tensor([0, 0, 1, 1])
    precomputed_pos = gather_precomputed_dino_views(
        bank,
        torch.tensor([0, 1, 2, 3]),
    )
    precomputed_gamma = gather_precomputed_dino_views(
        bank,
        torch.tensor([4, 5, 6, 7]),
    )
    extract_calls: list[int] = []

    def fake_extract(
        _extractor: object,
        x_flat: torch.Tensor,
        **_: object,
    ) -> list[torch.Tensor]:
        extract_calls.append(int(x_flat.shape[0]))
        base = x_flat[:, :feat_dim]
        return [base for _ in range(num_views)]

    with patch("losses.extract_feature_views", side_effect=fake_extract):
        loss, _ = conditional_drift_loss(
            x_real,
            x_gen,
            class_id_real,
            class_id_real,
            dino=MagicMock(),
            dino_weight=1.0,
            pixel_weight=0.0,
            x_real_pos=x_real,
            class_id_pos=class_id_pos,
            precomputed_pos_views=precomputed_pos,
            precomputed_gamma_views=precomputed_gamma,
        )
    loss.backward()
    # Only gens go through DINO; pos and γ-views were both gathered from bank.
    assert extract_calls == [n]
    assert x_gen.grad is not None


def test_precomputed_with_queue_requires_gamma_views() -> None:
    """Queue + precomputed_pos_views without precomputed_gamma_views is an error."""
    x_real = torch.randn(4, 12)
    x_gen = torch.randn(4, 12)
    class_id = torch.tensor([0, 0, 1, 1])
    precomputed = [torch.randn(4, 8) for _ in range(2)]
    with pytest.raises(ValueError, match="precomputed_gamma_views"):
        conditional_drift_loss(
            x_real,
            x_gen,
            class_id,
            class_id,
            dino=MagicMock(),
            dino_weight=1.0,
            x_real_pos=x_real,
            class_id_pos=class_id,
            precomputed_pos_views=precomputed,
        )


def test_drift_returns_graph_connected_zero_when_no_class_has_positives() -> None:
    """When every gen class is absent from positives, loss is zero but stays graph-connected."""
    x_real = torch.randn(4, 3)
    x_gen = torch.randn(4, 3, requires_grad=True)
    class_id_pos = torch.tensor([0, 0, 1, 1])
    class_id_gen = torch.tensor([2, 2, 3, 3])  # disjoint from positives

    loss = conditional_drift_loss_for_views(
        [(x_real, x_gen)],
        class_id_pos,
        class_id_gen,
        temperatures=(0.1,),
        gamma=0.0,
    )
    loss.backward()

    assert float(loss.detach()) == 0.0
    assert x_gen.grad is not None
    assert torch.equal(x_gen.grad, torch.zeros_like(x_gen))


def test_dino_defaults_to_antialias_true() -> None:
    """CIFAR-faithful default: DINO bicubic resize uses antialias=True."""
    with patch("losses.torch.hub.load", return_value=MagicMock()):
        extractor = DINOFeatureExtractor()
    extractor.backbone.get_intermediate_layers = MagicMock(return_value=[])

    captured: dict[str, object] = {}
    real_interpolate = torch.nn.functional.interpolate

    def _spy(*args: object, **kwargs: object) -> torch.Tensor:
        captured.update(kwargs)
        return real_interpolate(*args, **kwargs)

    with patch("losses.F.interpolate", side_effect=_spy):
        extractor(torch.zeros(1, 3 * 32 * 32), image_size=32)
    assert captured["antialias"] is True


def test_dino_antialias_false_when_constructed_for_imagenet() -> None:
    """ImageNet path: DINOFeatureExtractor(antialias=False) passes False through
    (the AA backward kernel is unregistered under torch.compile on CUDA)."""
    with patch("losses.torch.hub.load", return_value=MagicMock()):
        extractor = DINOFeatureExtractor(antialias=False)
    extractor.backbone.get_intermediate_layers = MagicMock(return_value=[])

    captured: dict[str, object] = {}
    real_interpolate = torch.nn.functional.interpolate

    def _spy(*args: object, **kwargs: object) -> torch.Tensor:
        captured.update(kwargs)
        return real_interpolate(*args, **kwargs)

    with patch("losses.F.interpolate", side_effect=_spy):
        extractor(torch.zeros(1, 3 * 64 * 64), image_size=64)
    assert captured["antialias"] is False


def test_compiled_drift_matches_eager() -> None:
    """Compiling the drift core must not change the loss (numerical equivalence)."""

    def _loss(*, compile_drift: bool) -> torch.Tensor:
        torch.manual_seed(0)
        x_real = torch.randn(12, 8)
        x_gen = torch.randn(12, 8, requires_grad=True)
        cls = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
        return conditional_drift_loss_for_views(
            [(x_real, x_gen)],
            cls,
            cls,
            temperatures=(0.05, 0.2),
            gamma=0.2,
            compile_drift=compile_drift,
        )

    eager = _loss(compile_drift=False)
    compiled = _loss(compile_drift=True)
    torch.testing.assert_close(eager, compiled, rtol=1e-4, atol=1e-5)


def test_conditional_drift_loss_golden_value_unchanged() -> None:
    """No-regression guard: conditional_drift_loss reproduces the origin/main value
    on fixed inputs with a deterministic mock DINO."""
    torch.manual_seed(0)
    x_real = torch.rand(12, 3 * 32 * 32)
    x_gen = torch.rand(12, 3 * 32 * 32, requires_grad=True)
    cls = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    proj = torch.randn(3 * 32 * 32, 16)
    dino = MagicMock(side_effect=lambda x, image_size=32: x @ proj)

    total, _ = conditional_drift_loss(
        x_real,
        x_gen,
        cls,
        cls,
        dino=dino,
        dino_weight=0.911,
        pixel_weight=0.114,
        gamma=0.2,
        image_size=32,
    )
    # rel tolerance, not abs=1e-9: CPU reduction order varies run-to-run, so the
    # value wobbles at ~1e-7. A real regression moves it by percent-level.
    assert float(total) == pytest.approx(0.28404372930526733, rel=1e-4)
