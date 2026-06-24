from __future__ import annotations

from collections.abc import Callable, Sequence
import functools
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

DRIFT_TEMPERATURES = (0.02, 0.05, 0.2)
EPS = 1e-12
_QUEUE_INPUT_NDIM = 2


@functools.cache
def _compiled_drift_target() -> Callable[..., Tensor]:
    """Compile the fixed-shape drift core once; cached for the process lifetime."""
    return torch.compile(_drift_target_for_class)


def _get_drift_target_fn(*, compile_drift: bool) -> Callable[..., Tensor]:
    """Return the drift-target core, eager or compiled-and-cached.

    Only the fixed-shape tensor core is compiled; the Python class loop in
    `conditional_drift_loss_for_views` stays eager.
    """
    return _compiled_drift_target() if compile_drift else _drift_target_for_class


def _pairwise_l2(x: Tensor, y: Tensor) -> Tensor:
    """Pairwise Euclidean distances over the last dim; supports bf16 on CUDA."""
    xx = (x * x).sum(dim=-1, keepdim=True)
    yy = (y * y).sum(dim=-1, keepdim=True).transpose(-2, -1)
    cross = x @ y.transpose(-1, -2)
    dist_sq = (xx + yy - 2.0 * cross).clamp_min(0)
    return dist_sq.sqrt()


def _drift_target_for_class(
    gen_c_d: Tensor,
    pos_c: Tensor,
    neg_c: Tensor,
    *,
    n_gen_c: int,
    temperatures: Tensor,
) -> Tensor:
    """Batched drift target across all views and temperatures for one class."""
    eps = gen_c_d.new_tensor(EPS)
    dist_pos = _pairwise_l2(gen_c_d, pos_c)
    dist_neg = _pairwise_l2(gen_c_d, neg_c)

    self_eye = torch.eye(n_gen_c, device=gen_c_d.device, dtype=gen_c_d.dtype)
    self_mask = torch.zeros_like(dist_neg)
    self_mask[:, :, :n_gen_c] = self_eye

    total_sum = (
        dist_pos.sum(dim=(-2, -1))
        + dist_neg.sum(dim=(-2, -1))
        - (dist_neg[:, :, :n_gen_c] * self_eye).sum(dim=(-2, -1))
    )
    n_pos_c = pos_c.shape[1]
    n_neg = neg_c.shape[1]
    total_count = float(n_gen_c * n_pos_c + n_gen_c * n_neg - n_gen_c)
    scale = (total_sum / total_count).clamp_min(eps).detach()

    scale_view = scale.view(-1, 1, 1)
    dist_pos = dist_pos / scale_view
    dist_neg = dist_neg / scale_view + self_mask * 1e6

    logits = torch.cat([-dist_pos, -dist_neg], dim=-1)
    logits_t = (logits.unsqueeze(0) / temperatures.view(-1, 1, 1, 1)).to(dtype=gen_c_d.dtype)
    a_row = torch.softmax(logits_t, dim=-1)
    a_col = torch.softmax(logits_t, dim=-2)
    assignment = torch.sqrt((a_row * a_col).clamp_min(eps))

    a_pos = assignment[..., :n_pos_c]
    a_neg = assignment[..., n_pos_c:]
    w_pos = a_pos * a_neg.sum(dim=-1, keepdim=True)
    w_neg = a_neg * a_pos.sum(dim=-1, keepdim=True)

    drift = w_pos @ pos_c.unsqueeze(0) - w_neg @ neg_c.unsqueeze(0)
    drift_scale = drift.norm(dim=-1).mean(dim=-1).clamp_min(eps).detach()
    drift = drift / drift_scale.view(*drift_scale.shape, 1, 1)

    return gen_c_d.unsqueeze(0) + drift


def conditional_drift_loss_for_views(
    view_pairs: list[tuple[Tensor, Tensor]],
    class_id_pos: Tensor,
    class_id_gen: Tensor,
    *,
    gamma_views: list[Tensor] | None = None,
    class_id_gamma: Tensor | None = None,
    temperatures: Sequence[float] = DRIFT_TEMPERATURES,
    gamma: float = 0.0,
    compile_drift: bool = False,
) -> Tensor:
    """Class-conditional drift loss over one or more feature views.

    For each class present in the gen batch, positives come from
    ``view_pairs[k][0]`` (real side of view ``k``, filtered by
    ``class_id_pos == c``). Negatives are same-class gens (self-masked) plus
    a γ-fraction of other-class reals drawn from ``gamma_views`` (or
    ``view_pairs`` reals when ``gamma_views`` is ``None``), with labels
    ``class_id_gamma`` (or ``class_id_pos`` when ``None``). The legacy path
    (positives and γ-source are the same tensor) is the default.

    Returns a graph-connected zero when no class in ``class_id_gen`` has
    positives available, so DDP all-reduce still fires during queue warmup.
    """
    if not view_pairs:
        raise ValueError("view_pairs must be non-empty.")
    if not 0.0 <= gamma < 1.0:
        raise ValueError(f"gamma must be in [0, 1), got {gamma}.")
    if gamma_views is not None and len(gamma_views) != len(view_pairs):
        raise ValueError(
            f"gamma_views length {len(gamma_views)} must match view_pairs "
            f"length {len(view_pairs)}.",
        )
    if class_id_gamma is None:
        class_id_gamma = class_id_pos

    pos_stack = torch.stack([v for (v, _) in view_pairs], dim=0)
    gen_stack = torch.stack([g for (_, g) in view_pairs], dim=0)
    gamma_stack = torch.stack(gamma_views, dim=0) if gamma_views is not None else pos_stack

    total = view_pairs[0][1].new_zeros(())
    valid_classes = 0
    drift_target_fn = _get_drift_target_fn(compile_drift=compile_drift)

    for c in torch.unique(class_id_gen).tolist():
        gen_mask = class_id_gen == c
        pos_mask = class_id_pos == c
        other_mask = class_id_gamma != c
        n_gen_c = int(gen_mask.sum())
        n_pos = int(pos_mask.sum())
        if n_gen_c == 0 or n_pos == 0:
            continue

        other_perm: Tensor | None = None
        if gamma > 0.0:
            n_other_max = int(other_mask.sum())
            # From q̃(·|c) = (1−γ)q_θ(·|c) + γ p_data(·|∅): with n_gen_c same-class
            # gen negatives carrying weight (1−γ), balance with n_gen_c·γ/(1−γ)
            # other-class reals carrying weight γ.
            n_other = min(round(n_gen_c * gamma / (1.0 - gamma)), n_other_max)
            if n_other > 0:
                other_idx = torch.nonzero(other_mask, as_tuple=True)[0]
                other_perm = other_idx[
                    torch.randperm(len(other_idx), device=other_idx.device)[:n_other]
                ]

        gen_c_idx = torch.nonzero(gen_mask, as_tuple=True)[0]
        pos_c_idx = torch.nonzero(pos_mask, as_tuple=True)[0]
        x_gen_view_c_stack = gen_stack[:, gen_c_idx, :]
        compute_dtype = x_gen_view_c_stack.dtype
        temperatures_t = torch.tensor(
            tuple(temperatures),
            device=gen_stack.device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            pos_c = pos_stack[:, pos_c_idx, :].detach().to(dtype=compute_dtype)
            gen_c_d = gen_stack[:, gen_c_idx, :].detach().to(dtype=compute_dtype)
            if other_perm is not None:
                other = gamma_stack[:, other_perm, :].detach().to(dtype=compute_dtype)
                neg_c = torch.cat([gen_c_d, other], dim=1)
            else:
                neg_c = gen_c_d
            target = drift_target_fn(
                gen_c_d,
                pos_c,
                neg_c,
                n_gen_c=n_gen_c,
                temperatures=temperatures_t,
            )

        target_cast = target.to(dtype=compute_dtype)
        diff = x_gen_view_c_stack.unsqueeze(0) - target_cast
        diff_mse = diff.square().mean(dim=(-2, -1))
        total = total + diff_mse.sum(dim=0).mean(dim=0)
        valid_classes += 1

    if valid_classes == 0:
        # Graph-connected zero so DDP still sees gradients on all params.
        return (view_pairs[0][1] * 0.0).sum()
    return total / valid_classes


class DINOFeatureExtractor(nn.Module):
    """Fixed DINOv2-S/14 feature extractor used by the drift loss."""

    def __init__(self, *, antialias: bool = True) -> None:
        """Load and freeze the DINOv2 backbone."""
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.antialias = antialias

    def train(self, mode: bool = True) -> DINOFeatureExtractor:
        """Keep the frozen backbone in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        x_flat: Tensor,
        *,
        image_size: int = 32,
    ) -> list[Tensor]:
        """Return 68 feature views: 4 layers times (4x4 patches + CLS)."""
        # The backbone is fp32 (frozen torch.hub weights). Inputs from the
        # positive queue stored as bf16 would otherwise hit a Conv2d dtype
        # mismatch when --precision=fp32 disables autocast.
        x_flat = x_flat.to(dtype=self.imagenet_mean.dtype)
        images = x_flat.reshape(x_flat.shape[0], 3, image_size, image_size)
        images = (images + 1.0) / 2.0
        images = (images - self.imagenet_mean.to(images.dtype)) / (
            self.imagenet_std.to(images.dtype)
        )
        interp_dtype = images.dtype
        images_for_interp = images.float() if interp_dtype != torch.float32 else images
        images = F.interpolate(
            images_for_interp,
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
            antialias=self.antialias,
        ).to(dtype=interp_dtype)

        features = self.backbone.get_intermediate_layers(
            images,
            n=4,
            return_class_token=True,
        )
        output: list[Tensor] = []
        for patch_tokens, cls_token in features:
            pooled_patches = _pool_dino_patches(patch_tokens)
            output.extend(pooled_patches[:, :, i, j] for i in range(4) for j in range(4))
            output.append(F.normalize(cls_token, p=2, dim=1))
        return output


def _pool_dino_patches(patch_tokens: Tensor) -> Tensor:
    """L2-normalize DINO patches and pool them to a 4x4 grid."""
    patches = F.normalize(patch_tokens, p=2, dim=2)
    grid_size = math.isqrt(int(patches.shape[1]))
    if grid_size * grid_size != int(patches.shape[1]):
        raise ValueError(f"Expected square DINO patch grid, got {patches.shape[1]}.")
    patches = patches.reshape(
        patches.shape[0],
        grid_size,
        grid_size,
        patches.shape[2],
    ).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(patches, output_size=(4, 4))


def extract_feature_views(
    extractor: nn.Module,
    x_flat: Tensor,
    *,
    batch_size: int = 64,
    image_size: int = 32,
) -> list[Tensor]:
    """Run a feature extractor in chunks and concatenate per-view outputs."""
    outputs_by_view: list[list[Tensor]] | None = None
    for chunk in x_flat.split(batch_size):
        chunk_outputs = extractor(chunk, image_size=image_size)
        views = [chunk_outputs] if isinstance(chunk_outputs, Tensor) else chunk_outputs
        if outputs_by_view is None:
            outputs_by_view = [[] for _ in views]
        if len(outputs_by_view) != len(views):
            raise ValueError("Feature extractor returned inconsistent view counts.")
        for index, view in enumerate(views):
            outputs_by_view[index].append(view)
    if outputs_by_view is None:
        raise ValueError("Cannot extract features from an empty tensor.")
    return [torch.cat(chunks, dim=0) for chunks in outputs_by_view]


def gather_precomputed_dino_views(bank: Tensor, sample_ids: Tensor) -> list[Tensor]:
    """Index a fixed bank of per-image DINO views.

    ``bank`` has shape ``(N, num_views, D)``; ``sample_ids`` is ``(B,)`` int64
    on the same device as ``bank``. Returns ``num_views`` tensors of shape
    ``(B, D)`` in the same order as ``extract_feature_views``.
    """
    block = bank[sample_ids]
    return [block[:, i, :] for i in range(int(block.shape[1]))]


def conditional_drift_loss(
    x_real: Tensor,
    x_gen: Tensor,
    class_id_real: Tensor,
    class_id_gen: Tensor,
    *,
    dino: DINOFeatureExtractor | None,
    dino_weight: float = 0.911,
    pixel_weight: float = 0.114,
    gamma: float = 0.2,
    feature_batch_size: int = 64,
    image_size: int = 32,
    x_real_pos: Tensor | None = None,
    class_id_pos: Tensor | None = None,
    precomputed_pos_views: list[Tensor] | None = None,
    precomputed_gamma_views: list[Tensor] | None = None,
    compile_drift: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the class-conditional DINO + pixel drift objective.

    Shared by the CIFAR-10 and ImageNet-64 training scripts; ``image_size`` and
    the loss weights are the only experiment-specific inputs.

    When ``x_real_pos`` is given (queue mode), positives for each class come
    from ``x_real_pos`` / ``class_id_pos`` while γ-mix other-class negatives
    still come from ``x_real`` / ``class_id_real`` (the current batch). When
    omitted, the batch reals serve both roles (legacy behavior).

    When ``precomputed_pos_views`` is set (same layout as ``extract_feature_views``
    for the positive batch), DINO is not run on positives. In queue mode,
    ``precomputed_gamma_views`` must be supplied for the γ-mix views from the
    current batch — both lookups index the same global bank by sample id.
    Generator views are always computed live.
    """
    using_queue = x_real_pos is not None
    if using_queue and class_id_pos is None:
        raise ValueError("class_id_pos is required when x_real_pos is given.")
    x_pos = x_real_pos if using_queue else x_real
    class_id_for_pos = class_id_pos if using_queue else class_id_real

    total = x_gen.new_zeros(())
    metrics: dict[str, Tensor] = {}

    if pixel_weight != 0.0:
        pixel_raw = conditional_drift_loss_for_views(
            [(x_pos, x_gen)],
            class_id_for_pos,
            class_id_gen,
            gamma_views=[x_real] if using_queue else None,
            class_id_gamma=class_id_real if using_queue else None,
            gamma=gamma,
            compile_drift=compile_drift,
        )
        total = total + float(pixel_weight) * pixel_raw
        metrics["loss/pixel_raw"] = pixel_raw.detach()

    if dino_weight != 0.0:
        if dino is None:
            raise ValueError("dino is required when dino_weight is non-zero.")
        with torch.no_grad():
            if precomputed_pos_views is not None:
                pos_views = precomputed_pos_views
                if using_queue:
                    if precomputed_gamma_views is None:
                        raise ValueError(
                            "precomputed_gamma_views is required when "
                            "queue mode (x_real_pos) is combined with "
                            "precomputed_pos_views.",
                        )
                    gamma_feat_views = precomputed_gamma_views
                else:
                    gamma_feat_views = None
            else:
                pos_views = extract_feature_views(
                    dino,
                    x_pos.detach(),
                    batch_size=feature_batch_size,
                    image_size=image_size,
                )
                gamma_feat_views = None
                if using_queue:
                    gamma_feat_views = extract_feature_views(
                        dino,
                        x_real.detach(),
                        batch_size=feature_batch_size,
                        image_size=image_size,
                    )
        gen_views = extract_feature_views(
            dino,
            x_gen,
            batch_size=feature_batch_size,
            image_size=image_size,
        )
        dino_raw = conditional_drift_loss_for_views(
            list(zip(pos_views, gen_views, strict=True)),
            class_id_for_pos,
            class_id_gen,
            gamma_views=gamma_feat_views,
            class_id_gamma=class_id_real if using_queue else None,
            gamma=gamma,
            compile_drift=compile_drift,
        )
        total = total + float(dino_weight) * dino_raw
        metrics["loss/dino_raw"] = dino_raw.detach()
        metrics["loss/dino_views"] = torch.as_tensor(
            len(gen_views), device=x_gen.device, dtype=x_gen.dtype
        )

    metrics["loss/total"] = total.detach()
    metrics["loss/num_samples"] = torch.as_tensor(
        x_gen.shape[0], device=x_gen.device, dtype=x_gen.dtype
    )
    return total, metrics


class PerClassQueue:
    """Per-class FIFO ring buffer that feeds positives to ``conditional_drift_loss``.

    Backed by a single ``(num_classes, queue_size, data_dim)`` storage tensor
    with device-resident write pointers and sample counts so the hot path
    avoids device-to-host synchronization.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        queue_size: int,
        data_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        track_sample_ids: bool = False,
    ) -> None:
        """Pre-allocate the buffer and per-class bookkeeping.

        Set `track_sample_ids` to also store each row's dataset index, needed
        only to look positives up in a precomputed DINO feature bank.
        """
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
        if queue_size < 1:
            raise ValueError(f"queue_size must be >= 1, got {queue_size}.")
        if data_dim < 1:
            raise ValueError(f"data_dim must be >= 1, got {data_dim}.")

        self.num_classes = int(num_classes)
        self.queue_size = int(queue_size)
        self.data_dim = int(data_dim)
        self.device = torch.device(device)
        self.dtype = dtype
        self.track_sample_ids = bool(track_sample_ids)

        self._buffer = torch.zeros(
            self.num_classes,
            self.queue_size,
            self.data_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self._buffer_ids = (
            torch.zeros(
                self.num_classes,
                self.queue_size,
                device=self.device,
                dtype=torch.long,
            )
            if self.track_sample_ids
            else None
        )
        self._write_ptr = torch.zeros(
            self.num_classes,
            device=self.device,
            dtype=torch.long,
        )
        self._counts = torch.zeros(
            self.num_classes,
            device=self.device,
            dtype=torch.long,
        )

    def push(
        self,
        x: Tensor,
        class_ids: Tensor,
        sample_ids: Tensor | None = None,
    ) -> None:
        """Push a batch into per-class FIFO slots, overwriting the oldest first.

        ``sample_ids`` is a parallel ``(B,)`` int64 tensor of dataset indices,
        required only when the queue was built with ``track_sample_ids``.
        """
        if x.ndim != _QUEUE_INPUT_NDIM or x.shape[1] != self.data_dim:
            raise ValueError(
                f"Expected x shape (B, {self.data_dim}), got {tuple(x.shape)}.",
            )
        if class_ids.ndim != 1 or class_ids.shape[0] != x.shape[0]:
            raise ValueError(
                f"class_ids must have shape ({x.shape[0]},), got {tuple(class_ids.shape)}.",
            )
        if self.track_sample_ids:
            if sample_ids is None:
                raise ValueError("sample_ids is required when track_sample_ids.")
            if sample_ids.ndim != 1 or sample_ids.shape[0] != x.shape[0]:
                raise ValueError(
                    f"sample_ids must have shape ({x.shape[0]},), got {tuple(sample_ids.shape)}.",
                )
        batch = x.shape[0]
        if batch == 0:
            return

        x_dev = x.detach().to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        )
        cids = class_ids.detach().to(
            device=self.device,
            dtype=torch.long,
            non_blocking=True,
        )

        one_hot = F.one_hot(cids, num_classes=self.num_classes).to(torch.long)
        # intra-batch rank of each sample within its class (0-indexed).
        intra_rank = one_hot.cumsum(dim=0)[torch.arange(batch, device=self.device), cids] - 1
        slot = (self._write_ptr[cids] + intra_rank) % self.queue_size
        self._buffer[cids, slot] = x_dev
        if self.track_sample_ids:
            self._buffer_ids[cids, slot] = sample_ids.detach().to(
                device=self.device,
                dtype=torch.long,
                non_blocking=True,
            )

        per_class = one_hot.sum(dim=0)
        self._write_ptr = (self._write_ptr + per_class) % self.queue_size
        self._counts = torch.clamp(self._counts + per_class, max=self.queue_size)

    def draw(
        self,
        class_ids: Tensor,
        num_pos: int,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Draw ``num_pos`` samples per class without replacement.

        Callers must ensure each requested class has at least ``num_pos``
        samples queued (check via ``ready_mask``). Returns
        ``(samples, labels, sample_ids)`` with rows grouped by class in the
        order of ``class_ids``; ``sample_ids`` is ``None`` unless the queue
        tracks them.
        """
        if class_ids.ndim != 1:
            raise ValueError(
                f"class_ids must be 1-D, got shape {tuple(class_ids.shape)}.",
            )
        if num_pos < 1:
            raise ValueError(f"num_pos must be >= 1, got {num_pos}.")

        k = int(class_ids.shape[0])
        if k == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return (
                torch.empty(0, self.data_dim, device=self.device, dtype=self.dtype),
                empty,
                empty if self.track_sample_ids else None,
            )

        cids = class_ids.detach().to(
            device=self.device,
            dtype=torch.long,
            non_blocking=True,
        )
        counts = self._counts.index_select(0, cids)
        # Uniform without replacement: rand scores, mask invalid slots to +inf, topk smallest.
        scores = torch.rand(
            k,
            self.queue_size,
            device=self.device,
            dtype=torch.float32,
        )
        positions = torch.arange(self.queue_size, device=self.device)
        invalid = positions.unsqueeze(0) >= counts.unsqueeze(1)
        scores = scores.masked_fill(invalid, float("inf"))
        _, top_idx = torch.topk(scores, num_pos, dim=1, largest=False)

        expanded_cids = cids.unsqueeze(1).expand(k, num_pos)
        samples = self._buffer[expanded_cids, top_idx].reshape(
            k * num_pos,
            self.data_dim,
        )
        labels = expanded_cids.reshape(k * num_pos)
        sample_ids = None
        if self.track_sample_ids:
            sample_ids = self._buffer_ids[expanded_cids, top_idx].reshape(
                k * num_pos,
            )
        return samples, labels, sample_ids

    def ready_mask(self, num_pos: int) -> Tensor:
        """Device-side bool mask of classes with at least ``num_pos`` samples."""
        return self._counts >= int(num_pos)
