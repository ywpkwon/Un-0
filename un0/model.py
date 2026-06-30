"""Class-conditional Kuramoto implicit image generator (CIFAR-10)."""

from __future__ import annotations

from collections.abc import Callable
import inspect
import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchdiffeq import odeint

Encoding = Literal["raw", "sin", "sin_cos"]
Relativization = Literal["absolute", "mean_relative", "ref_oscillator", "pairwise"]
Parameterization = Literal["standard", "mup"]
Solver = Literal["euler", "rk4"]
DynamicsKind = Literal["kuramoto", "lohe_fixed"]


def _kuramoto_velocity(theta: Tensor, omega: Tensor, coupling: Tensor) -> Tensor:
    """Kuramoto phase velocity: dθ/dt = ω + cos(θ) · (sin(θ) @ Kᵀ) − sin(θ) · (cos(θ) @ Kᵀ)."""
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    weighted_sin = sin_theta @ coupling.transpose(-1, -2)
    weighted_cos = cos_theta @ coupling.transpose(-1, -2)
    return omega + cos_theta * weighted_sin - sin_theta * weighted_cos


class ConditionalKuramotoDynamics(nn.Module):
    """Class-conditional Kuramoto dynamics.

    Two coupled Kuramoto blocks with a class-dependent one-way drive:

      * Main block: `n` oscillators with coupling `K` (shape `(n, n)`)
        and natural frequencies `omega` (shape `(1, n)`).
      * Conditioning block: `n_cond` oscillators with coupling `K_cond`
        (shape `(n_cond, n_cond)`) and frequencies `omega_cond`.
      * Class drive: `K_drive` of shape `(num_classes, n, n_cond)`
        couples the cond block into the main block with per-class weights.
        The main block does NOT feed back into the cond block.

    For a batch element with class label `c` and joint phase state
    `theta = [theta_main; theta_cond]`, the velocity is::

        dtheta_main/dt = omega
            + cos(theta_main) * (sin(theta_main) @ K.T)
            - sin(theta_main) * (cos(theta_main) @ K.T)
            + cos(theta_main) * (K_drive[c] @ sin(theta_cond))
            - sin(theta_main) * (K_drive[c] @ cos(theta_cond))

        dtheta_cond/dt = omega_cond
            + cos(theta_cond) * (sin(theta_cond) @ K_cond.T)
            - sin(theta_cond) * (cos(theta_cond) @ K_cond.T)

    This is the standard Kuramoto term `sum_j K_ij sin(theta_j - theta_i)`
    expanded via the angle-difference identity so it becomes two matmuls per
    block instead of an outer product per batch element.

    Diagonals of `K` and `K_cond` are zeroed on every forward so
    oscillators don't self-couple. Class dropout: with probability
    `class_dropout_prob`, `K_drive[c]` is zeroed for a batch element
    during training, so the model also learns unconditional generation
    (classifier-free-guidance style). Readout takes the first `n` phases.
    """

    def __init__(
        self,
        *,
        n_oscillators: int,
        n_conditional_oscillators: int,
        num_classes: int,
        init_k_scale: float = 1.0,
        init_freq_scale: float = 1.0,
        parameterization: Parameterization = "standard",
    ) -> None:
        """Initialize conditional Kuramoto dynamics."""
        super().__init__()
        if n_oscillators < 2 or n_conditional_oscillators < 1 or num_classes < 1:
            raise ValueError(
                "Need n_oscillators >= 2, n_conditional_oscillators >= 1, num_classes >= 1."
            )
        if parameterization not in ("standard", "mup"):
            raise ValueError(
                f"parameterization must be 'standard' or 'mup', got {parameterization!r}."
            )

        self.n = int(n_oscillators)
        self.n_cond = int(n_conditional_oscillators)
        self.num_classes = int(num_classes)
        self.parameterization = parameterization

        if parameterization == "mup":
            self._K_scale = self.n**-0.5
            self._K_cond_scale = self.n_cond**-0.5
            self._K_drive_scale = self.n_cond**-0.5
            k_init_scale = 1.0
            k_cond_init_scale = 1.0
            k_drive_init_scale = 1.0
        else:
            self._K_scale = 1.0
            self._K_cond_scale = 1.0
            self._K_drive_scale = 1.0
            k_init_scale = self.n**-0.5
            k_cond_init_scale = self.n_cond**-0.5
            k_drive_init_scale = self.n_cond**-0.5

        self.omega = nn.Parameter(init_freq_scale * torch.randn(1, self.n))
        K_init = init_k_scale * k_init_scale * torch.randn(self.n, self.n)
        K_init.fill_diagonal_(0.0)
        self.K = nn.Parameter(K_init)

        self.omega_cond = nn.Parameter(init_freq_scale * torch.randn(1, self.n_cond))
        K_cond_init = init_k_scale * k_cond_init_scale * torch.randn(self.n_cond, self.n_cond)
        K_cond_init.fill_diagonal_(0.0)
        self.K_cond = nn.Parameter(K_cond_init)

        self.K_drive = nn.Parameter(
            init_k_scale * k_drive_init_scale * torch.randn(self.num_classes, self.n, self.n_cond)
        )

    @property
    def state_dim(self) -> int:
        """Total state dimension (main + cond)."""
        return self.n + self.n_cond

    def forward(self, state: Tensor, _time: Tensor, drive: Tensor) -> Tensor:
        """Compute dstate/dt for concatenated (main, cond) phases.

        Args:
            state: Concatenated phases shaped `(batch, n + n_cond)`.
            _time: Unused (required by torchdiffeq signature).
            drive: Per-sample drive matrix shaped `(batch, n, n_cond)`
                (typically `K_drive[class_id]` with optional zeroing for
                class dropout).
        """
        theta_main = state[:, : self.n]
        theta_cond = state[:, self.n :]

        # Subtract the learned diagonal so oscillators don't self-couple.
        # Gradient matches masked_fill: zero on diagonal, full off-diagonal.
        K = (self.K - torch.diag_embed(self.K.diagonal())) * self._K_scale
        K_cond = (self.K_cond - torch.diag_embed(self.K_cond.diagonal())) * self._K_cond_scale

        main_vel = _kuramoto_velocity(theta_main, self.omega, K)
        cond_vel = _kuramoto_velocity(theta_cond, self.omega_cond, K_cond)

        sin_c = torch.sin(theta_cond)
        cos_c = torch.cos(theta_cond)
        sin_m = torch.sin(theta_main)
        cos_m = torch.cos(theta_main)
        drive = drive * self._K_drive_scale
        drive_sin = torch.einsum("bnm,bm->bn", drive, sin_c)
        drive_cos = torch.einsum("bnm,bm->bn", drive, cos_c)
        main_vel = main_vel + cos_m * drive_sin - sin_m * drive_cos

        return torch.cat([main_vel, cond_vel], dim=1)


class ConditionalFixedAnchorLoheDynamics(nn.Module):
    """Class-conditional fixed-anchor Lohe dynamics with analytic equilibrium.

    Each of the `n` free oscillators lives on ``S^(d-1)`` and is pulled toward a
    learned bank of fixed anchors. For class ``c`` and oscillator ``i``:

        h[c, i] = sum_a softplus(K_drive[c, i, a]) * anchor[a]
        x*[c, i] = h[c, i] / ||h[c, i]||

    This is the omega=0, fixed-anchor Lohe setting from fixed-query oscillator
    attention. The training path uses ``x*`` directly; ``forward`` implements
    the matching ODE velocity for finite-time experiments.
    """

    def __init__(
        self,
        *,
        n_oscillators: int,
        n_anchors: int,
        oscillator_dim: int,
        num_classes: int,
        latent_dim: int = 0,
        latent_class_embed_dim: int = 64,
        latent_pos_embed_dim: int = 32,
        latent_hidden_dim: int = 512,
        latent_delta_scale: float = 0.3,
        init_k_scale: float = 1.0,
        parameterization: Parameterization = "standard",
        eps: float = 1e-8,
    ) -> None:
        """Initialize fixed-anchor Lohe dynamics."""
        super().__init__()
        if n_oscillators < 1 or n_anchors < 1 or oscillator_dim < 2 or num_classes < 1:
            raise ValueError(
                "Need n_oscillators >= 1, n_anchors >= 1, oscillator_dim >= 2, "
                "num_classes >= 1."
            )
        if parameterization not in ("standard", "mup"):
            raise ValueError(
                f"parameterization must be 'standard' or 'mup', got {parameterization!r}."
            )
        self.n = int(n_oscillators)
        self.n_cond = int(n_anchors)
        self.oscillator_dim = int(oscillator_dim)
        self.num_classes = int(num_classes)
        self.parameterization = parameterization
        self.eps = float(eps)
        self.uses_closed_form = True
        self.latent_dim = int(latent_dim)
        self.latent_class_embed_dim = int(latent_class_embed_dim)
        self.latent_pos_embed_dim = int(latent_pos_embed_dim)
        self.latent_hidden_dim = int(latent_hidden_dim)
        self.latent_delta_scale = float(latent_delta_scale)
        if self.latent_dim < 0:
            raise ValueError(f"latent_dim must be >= 0, got {latent_dim}.")
        if self.latent_dim > 0 and (
            self.latent_class_embed_dim < 1
            or self.latent_pos_embed_dim < 1
            or self.latent_hidden_dim < 1
        ):
            raise ValueError(
                "Latent-conditioned Lohe requires positive class, position, and hidden dims."
            )

        if parameterization == "mup":
            self._K_drive_scale = self.n_cond**-0.5
            k_drive_init_scale = 1.0
        else:
            self._K_drive_scale = 1.0
            k_drive_init_scale = self.n_cond**-0.5

        self.anchor = nn.Parameter(torch.randn(self.n_cond, self.oscillator_dim))
        self.K_drive = nn.Parameter(
            init_k_scale * k_drive_init_scale * torch.randn(self.num_classes, self.n, self.n_cond)
        )
        if self.latent_dim > 0:
            self.class_embed = nn.Embedding(self.num_classes, self.latent_class_embed_dim)
            self.null_class_embed = nn.Parameter(torch.zeros(self.latent_class_embed_dim))
            self.position_embed = nn.Parameter(
                0.02 * torch.randn(self.n, self.latent_pos_embed_dim)
            )
            latent_in_dim = (
                self.latent_dim + self.latent_class_embed_dim + self.latent_pos_embed_dim
            )
            self.latent_drive_mlp = nn.Sequential(
                nn.Linear(latent_in_dim, self.latent_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.latent_hidden_dim, self.latent_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.latent_hidden_dim, self.n_cond),
            )
            final = self.latent_drive_mlp[-1]
            if isinstance(final, nn.Linear):
                nn.init.normal_(final.weight, mean=0.0, std=0.02)
                nn.init.zeros_(final.bias)
        else:
            self.class_embed = None
            self.null_class_embed = None
            self.position_embed = None
            self.latent_drive_mlp = None

    @property
    def state_dim(self) -> int:
        """Flattened free-oscillator state dimension."""
        return self.n * self.oscillator_dim

    def _anchors(self) -> Tensor:
        return F.normalize(self.anchor, dim=-1, eps=self.eps)

    def _positive_drive(self, drive: Tensor) -> Tensor:
        return F.softplus(drive) * self._K_drive_scale

    def conditioned_drive(
        self,
        class_id: Tensor,
        *,
        class_keep: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Return class drive, optionally perturbed by per-token latent variables.

        Shapes for the latent-conditioned path:
          z_tokens: [B, n, latent_dim]
          class:    [B, n, latent_class_embed_dim]
          pos:      [B, n, latent_pos_embed_dim]
          delta:    [B, n, n_anchors]
        """
        base = self.K_drive[class_id]
        if class_keep is not None:
            keep = class_keep.to(dtype=base.dtype).view(-1, 1, 1)
            base = base * keep
        if self.latent_dim == 0:
            return base

        batch = int(class_id.shape[0])
        device = base.device
        dtype = base.dtype
        z_tokens = torch.randn(
            batch,
            self.n,
            self.latent_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        class_tokens = self.class_embed(class_id).to(dtype=dtype)
        if class_keep is not None:
            null_tokens = self.null_class_embed.to(dtype=dtype).expand_as(class_tokens)
            class_tokens = torch.where(class_keep.view(-1, 1), class_tokens, null_tokens)
        class_tokens = class_tokens[:, None, :].expand(-1, self.n, -1)
        pos_tokens = self.position_embed.to(dtype=dtype)[None, :, :].expand(batch, -1, -1)
        latent_in = torch.cat([z_tokens, class_tokens, pos_tokens], dim=-1)
        latent_delta = self.latent_drive_mlp(latent_in)
        return base + self.latent_delta_scale * latent_delta

    def fixed_point(self, drive: Tensor) -> Tensor:
        """Return the stable Lohe equilibrium shaped ``(batch, n, d)``."""
        weights = self._positive_drive(drive)
        h = torch.einsum("bna,ad->bnd", weights, self._anchors())
        return F.normalize(h, dim=-1, eps=self.eps)

    def sample_initial_state(
        self,
        num_samples: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample free oscillator positions uniformly enough for ODE smoke tests."""
        state = torch.randn(
            num_samples,
            self.n,
            self.oscillator_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        return F.normalize(state, dim=-1, eps=self.eps).reshape(num_samples, self.state_dim)

    def state_to_features(self, state: Tensor) -> Tensor:
        """Flatten vector-valued oscillator states for the decoder."""
        return state.reshape(state.shape[0], self.n * self.oscillator_dim)

    def forward(self, state: Tensor, _time: Tensor, drive: Tensor) -> Tensor:
        """Compute Lohe velocity ``dx/dt = (I - xx^T) h`` for fixed anchors."""
        x = state.reshape(state.shape[0], self.n, self.oscillator_dim)
        weights = self._positive_drive(drive)
        h = torch.einsum("bna,ad->bnd", weights, self._anchors())
        projected = h - x * (x * h).sum(dim=-1, keepdim=True)
        return projected.reshape(state.shape[0], self.state_dim)


class ReadoutTransform(nn.Module):
    """Map raw oscillator phases to decoder features."""

    def __init__(
        self,
        *,
        encoding: Encoding = "sin_cos",
        relativization: Relativization = "ref_oscillator",
    ) -> None:
        """Initialize the transform."""
        super().__init__()
        if encoding not in ("raw", "sin", "sin_cos"):
            raise ValueError(f"Invalid encoding: {encoding!r}.")
        if relativization not in (
            "absolute",
            "mean_relative",
            "ref_oscillator",
            "pairwise",
        ):
            raise ValueError(f"Invalid relativization: {relativization!r}.")
        self.encoding = encoding
        self.relativization = relativization

    def forward(self, phases: Tensor) -> Tensor:
        """Transform raw phases shaped `(batch, n)` to features."""
        if self.relativization == "mean_relative":
            phases = phases - phases.mean(dim=-1, keepdim=True)
        elif self.relativization == "ref_oscillator":
            phases = phases - phases[:, :1]
        elif self.relativization == "pairwise":
            diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)
            phases = diff.reshape(phases.shape[0], -1)

        if self.encoding == "sin":
            return torch.sin(phases)
        if self.encoding == "sin_cos":
            return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        return phases


class IdentityReadout(nn.Module):
    """Pass already-vector-valued oscillator features through unchanged."""

    def forward(self, features: Tensor) -> Tensor:
        """Return input features unchanged."""
        return features


class LoheSpatialTokenReadout(nn.Module):
    """Reshape Lohe oscillator tokens into a channel-first spatial feature map."""

    def __init__(
        self,
        *,
        n_oscillators: int,
        oscillator_dim: int,
        height: int,
        width: int,
    ) -> None:
        """Initialize the token-to-spatial readout."""
        super().__init__()
        if n_oscillators != height * width:
            raise ValueError(
                "n_oscillators must match height * width for spatial Lohe readout. "
                f"Got {n_oscillators} and {height} * {width}."
            )
        self.n_oscillators = int(n_oscillators)
        self.oscillator_dim = int(oscillator_dim)
        self.height = int(height)
        self.width = int(width)

    def forward(self, features: Tensor) -> Tensor:
        """Return features shaped `(batch, oscillator_dim, height, width)`."""
        tokens = features.reshape(
            features.shape[0],
            self.height,
            self.width,
            self.oscillator_dim,
        )
        return tokens.permute(0, 3, 1, 2).contiguous()


class ResizeConvBlock(nn.Module):
    """Nearest-neighbor upsample followed by two convolutions."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the block."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the upsampling block."""
        return self.net(x)


class ResizeConvDecoder(nn.Module):
    """Decode flat or spatial features to flattened images via resize convolutions."""

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        *,
        in_channels: int,
        in_height: int,
        in_width: int,
        out_channels: int = 3,
        num_upsamples: int = 3,
        final_activation: Literal["tanh", "none"] = "tanh",
        init_output_gain: float = 0.5,
    ) -> None:
        """Initialize the decoder."""
        super().__init__()
        if feature_dim != in_channels * in_height * in_width:
            raise ValueError(
                "feature_dim must match in_channels * in_height * in_width. "
                f"Got {feature_dim} and "
                f"{in_channels} * {in_height} * {in_width}."
            )
        height = in_height * (2**num_upsamples)
        width = in_width * (2**num_upsamples)
        expected_output_dim = out_channels * height * width
        if output_dim != expected_output_dim:
            raise ValueError(f"output_dim must be {expected_output_dim}, got {output_dim}.")
        if final_activation not in ("tanh", "none"):
            raise ValueError(f"Invalid final_activation: {final_activation!r}.")

        self.feature_dim = int(feature_dim)
        self.output_dim = int(output_dim)
        self.in_channels = int(in_channels)
        self.in_height = int(in_height)
        self.in_width = int(in_width)
        self.out_channels = int(out_channels)
        self.final_activation = final_activation

        blocks: list[nn.Module] = []
        current_channels = self.in_channels
        for _ in range(num_upsamples):
            next_channels = max(current_channels // 2, 32)
            blocks.append(ResizeConvBlock(current_channels, next_channels))
            current_channels = next_channels

        self.cascade = nn.Sequential(*blocks)
        self.to_output = nn.Conv2d(current_channels, self.out_channels, kernel_size=3, padding=1)
        self._init_weights(init_output_gain=init_output_gain)

    def _init_weights(self, *, init_output_gain: float) -> None:
        """Initialize convolution weights."""
        for module in self.cascade.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=0.2,
                    nonlinearity="leaky_relu",
                )
                nn.init.zeros_(module.bias)
        nn.init.xavier_normal_(
            self.to_output.weight,
            gain=nn.init.calculate_gain("tanh") * float(init_output_gain),
        )
        nn.init.zeros_(self.to_output.bias)

    def forward(self, features: Tensor) -> Tensor:
        """Decode features shaped `(batch, feature_dim)` or `(batch, channels, height, width)`."""
        if features.ndim == 4:
            expected = (self.in_channels, self.in_height, self.in_width)
            actual = tuple(features.shape[1:])
            if actual != expected:
                raise ValueError(f"Expected spatial features with shape (*, {expected}), got {actual}.")
            x = features
        else:
            x = features.reshape(features.shape[0], self.in_channels, self.in_height, self.in_width)
        x = self.to_output(self.cascade(x))
        if self.final_activation == "tanh":
            x = torch.tanh(x)
        return x.reshape(features.shape[0], self.output_dim)


class ConditionalImplicitKuramotoGenerator(nn.Module):
    """End-to-end class-conditional Kuramoto image generator."""

    def __init__(
        self,
        *,
        dynamics: nn.Module,
        readout: ReadoutTransform,
        decoder: nn.Module,
        class_dropout_prob: float = 0.0,
        integration_time: float = 1.0,
        num_steps: int = 25,
        solver: Solver = "rk4",
    ) -> None:
        """Initialize the generator."""
        super().__init__()
        if num_steps < 0:
            raise ValueError(f"num_steps must be >= 0, got {num_steps}.")
        if not 0.0 <= class_dropout_prob <= 1.0:
            raise ValueError(f"class_dropout_prob must be in [0, 1], got {class_dropout_prob}.")
        self.dynamics = dynamics
        self.readout = readout
        self.decoder = decoder
        self.class_dropout_prob = float(class_dropout_prob)
        self.integration_time = float(integration_time)
        self.num_steps = int(num_steps)
        self.solver = solver

    def _time_grid(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.linspace(
            0.0,
            self.integration_time,
            self.num_steps + 1,
            device=device,
            dtype=dtype,
        )

    def _sample_initial_state(
        self,
        num_samples: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample initial phases for main + driver uniformly from [-pi, pi)."""
        sampler = getattr(self.dynamics, "sample_initial_state", None)
        if sampler is not None:
            return sampler(num_samples, device=device, dtype=dtype, generator=generator)
        dim = self.dynamics.state_dim
        return (
            torch.rand(num_samples, dim, device=device, dtype=dtype, generator=generator)
            * (2.0 * torch.pi)
            - torch.pi
        )

    def _class_drive(self, class_id: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        """Gather per-sample class drive, optionally zeroed by class dropout."""
        class_keep: Tensor | None = None
        if self.training and self.class_dropout_prob > 0.0:
            class_keep = (
                torch.rand(
                    class_id.shape[0],
                    device=class_id.device,
                    generator=generator,
                )
                >= self.class_dropout_prob
            )
        conditioned_drive = getattr(self.dynamics, "conditioned_drive", None)
        if conditioned_drive is not None:
            return conditioned_drive(class_id, class_keep=class_keep, generator=generator)
        drive = self.dynamics.K_drive[class_id]
        if class_keep is not None:
            drive = drive * class_keep.to(dtype=drive.dtype).view(-1, 1, 1)
        return drive

    def forward(
        self,
        class_id: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate decoded samples for the given class labels.

        Args:
            class_id: Class labels shaped `(batch,)`, values in `[0, num_classes)`.
            generator: Optional RNG for reproducible sampling.
        """
        param = next(self.parameters())
        batch = int(class_id.shape[0])
        if self.num_steps == 0:
            initial_state = self._sample_initial_state(
                batch,
                device=param.device,
                dtype=param.dtype,
                generator=generator,
            )
            final_state = initial_state
        else:
            drive = self._class_drive(class_id, generator=generator)
            if bool(getattr(self.dynamics, "uses_closed_form", False)):
                final_state = self.dynamics.fixed_point(drive).reshape(batch, -1)
            else:
                initial_state = self._sample_initial_state(
                    batch,
                    device=param.device,
                    dtype=param.dtype,
                    generator=generator,
                )
                time_grid = self._time_grid(device=param.device, dtype=param.dtype)
                states = odeint(
                    lambda t, state: self.dynamics(state, t, drive),
                    initial_state,
                    time_grid,
                    method=self.solver,
                    options={"step_size": self.integration_time / self.num_steps},
                )
                final_state = states[-1]
        state_to_features = getattr(self.dynamics, "state_to_features", None)
        if state_to_features is None:
            features = self.readout(final_state[:, : self.dynamics.n])
        else:
            features = self.readout(state_to_features(final_state))
        return self.decoder(features)

    @torch.no_grad()
    def sample(
        self,
        class_id: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate samples without gradients."""
        was_training = self.training
        self.eval()
        try:
            return self.forward(class_id, generator=generator)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def sample_images(
        self,
        class_id: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate samples as image tensors ``(B, 3, H, W)`` in ``[0, 1]``.

        Convenience wrapper over :meth:`sample`, which returns the canonical
        flat ``(B, 3*H*W)`` tensor in ``[-1, 1]`` used for training and FID.
        Assumes square RGB output (3 channels, ``H == W``).
        """
        flat = self.sample(class_id, generator=generator)
        size = round((flat.shape[1] // 3) ** 0.5)
        images = flat.reshape(-1, 3, size, size)
        return ((images + 1.0) * 0.5).clamp(0.0, 1.0)

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        *,
        device: str | torch.device | None = None,
    ) -> ConditionalImplicitKuramotoGenerator:
        """Load a released checkpoint by name (e.g. `imagenet64/n16384`)."""
        if name not in _PRETRAINED:
            raise ValueError(
                f"Unknown pretrained name {name!r}. Available: {', '.join(PRETRAINED_NAMES)}."
            )
        from huggingface_hub import hf_hub_download

        filename, family = _PRETRAINED[name]
        path = hf_hub_download(_HF_REPO, filename)
        # weights_only=True: the file is a remote pickle (RCE surface); the
        # released payloads load cleanly under the safe unpickler.
        state = torch.load(path, map_location="cpu", weights_only=True)
        builder = build_cifar10_model if family == "cifar10" else build_imagenet64_model
        model = build_from_config(builder, state["config"])
        model.load_state_dict(state["model"])
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return model.to(device)


def prepare_class_ids_for_generation(
    *,
    num_samples: int,
    num_classes_per_step: int,
    num_total_classes: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample `num_classes_per_step` classes and tile them to fill `num_samples`.

    Each chosen class gets `num_samples // num_classes_per_step` samples; the
    first `num_samples % num_classes_per_step` chosen classes get one extra.
    Used so each generation step covers few classes densely (enough per-class
    samples for the drift target) rather than all classes too sparsely.
    """
    if not 1 <= num_classes_per_step <= num_total_classes:
        raise ValueError(
            "Need 1 <= num_classes_per_step <= num_total_classes, got "
            f"{num_classes_per_step} and {num_total_classes}."
        )
    if num_classes_per_step > num_samples:
        raise ValueError(
            f"num_classes_per_step ({num_classes_per_step}) cannot exceed "
            f"num_samples ({num_samples})."
        )
    chosen = torch.randperm(num_total_classes, device=device, generator=generator)[
        :num_classes_per_step
    ]
    base = num_samples // num_classes_per_step
    remainder = num_samples % num_classes_per_step
    counts = torch.full((num_classes_per_step,), base, device=device, dtype=torch.long)
    counts[:remainder] += 1
    return chosen.repeat_interleave(counts)


def _infer_cifar_lohe_spatial_layout(
    *,
    n_oscillators: int,
    requested_grid: int | None,
) -> tuple[int, int, int]:
    """Infer `(height, width, num_upsamples)` for CIFAR spatial-token Lohe."""
    grid = math.isqrt(n_oscillators) if requested_grid is None else int(requested_grid)
    if grid < 1 or grid * grid != int(n_oscillators):
        raise ValueError(
            "Spatial-token Lohe requires n_oscillators to equal grid^2. "
            f"Got n_oscillators={n_oscillators}, grid={grid}."
        )
    if 32 % grid != 0:
        raise ValueError(f"CIFAR spatial Lohe grid must divide 32, got grid={grid}.")
    scale = 32 // grid
    if scale < 1 or scale & (scale - 1):
        raise ValueError(
            "CIFAR spatial Lohe grid must require a power-of-two upsample to 32. "
            f"Got grid={grid}, scale={scale}."
        )
    return grid, grid, int(math.log2(scale))


def build_cifar10_model(
    *,
    n_oscillators: int = 4096,
    n_conditional_oscillators: int = 8,
    class_dropout_prob: float = 0.1,
    num_steps: int = 25,
    decoder_in_channels: int | None = None,
    parameterization: Parameterization = "standard",
    relativization: Relativization = "ref_oscillator",
    encoding: Encoding = "sin_cos",
    solver: Solver = "rk4",
    dynamics: DynamicsKind = "kuramoto",
    lohe_dim: int = 2,
    lohe_spatial_decoder: bool = False,
    lohe_decoder_grid: int | None = None,
    lohe_latent_dim: int = 0,
    lohe_latent_class_dim: int = 64,
    lohe_latent_pos_dim: int = 32,
    lohe_latent_hidden_dim: int = 512,
    lohe_latent_scale: float = 0.3,
) -> ConditionalImplicitKuramotoGenerator:
    """Build the CIFAR-10 class-conditional model used in the release."""
    if dynamics not in ("kuramoto", "lohe_fixed"):
        raise ValueError(f"dynamics must be 'kuramoto' or 'lohe_fixed', got {dynamics!r}.")
    # sin_cos concatenates sin and cos (2 * n); raw/sin pass n features through.
    feature_dim = (
        int(n_oscillators) * int(lohe_dim)
        if dynamics == "lohe_fixed"
        else (2 if encoding == "sin_cos" else 1) * int(n_oscillators)
    )
    decoder_in_height = 4
    decoder_in_width = 4
    decoder_num_upsamples = 3
    if bool(lohe_spatial_decoder):
        if dynamics != "lohe_fixed":
            raise ValueError("--lohe-spatial-decoder is only valid with dynamics='lohe_fixed'.")
        decoder_in_height, decoder_in_width, decoder_num_upsamples = (
            _infer_cifar_lohe_spatial_layout(
                n_oscillators=int(n_oscillators),
                requested_grid=lohe_decoder_grid,
            )
        )
        if decoder_in_channels is not None and int(decoder_in_channels) != int(lohe_dim):
            raise ValueError(
                "Spatial-token Lohe uses lohe_dim as decoder_in_channels. "
                f"Got decoder_in_channels={decoder_in_channels}, lohe_dim={lohe_dim}."
            )
        decoder_in_channels = int(lohe_dim)
    elif decoder_in_channels is None:
        spatial_features = decoder_in_height * decoder_in_width
        if feature_dim % spatial_features != 0:
            raise ValueError(
                f"feature_dim={feature_dim} must be divisible by "
                f"{spatial_features} when decoder_in_channels is not set."
            )
        decoder_in_channels = feature_dim // spatial_features

    if dynamics == "kuramoto":
        dynamics_module: nn.Module = ConditionalKuramotoDynamics(
            n_oscillators=int(n_oscillators),
            n_conditional_oscillators=int(n_conditional_oscillators),
            num_classes=10,
            init_k_scale=1.0,
            init_freq_scale=1.0,
            parameterization=parameterization,
        )
        # Compile the velocity function: called 4 * num_steps times per integration
        # with fixed shape (batch, n + n_cond), so Inductor fuses sin/cos +
        # the matmuls + the einsum drive into a handful of kernels.
        dynamics_module = torch.compile(dynamics_module)
        readout: nn.Module = ReadoutTransform(
            encoding=encoding,
            relativization=relativization,
        )
    else:
        dynamics_module = ConditionalFixedAnchorLoheDynamics(
            n_oscillators=int(n_oscillators),
            n_anchors=int(n_conditional_oscillators),
            oscillator_dim=int(lohe_dim),
            num_classes=10,
            latent_dim=int(lohe_latent_dim),
            latent_class_embed_dim=int(lohe_latent_class_dim),
            latent_pos_embed_dim=int(lohe_latent_pos_dim),
            latent_hidden_dim=int(lohe_latent_hidden_dim),
            latent_delta_scale=float(lohe_latent_scale),
            init_k_scale=1.0,
            parameterization=parameterization,
        )
        if bool(lohe_spatial_decoder):
            readout = LoheSpatialTokenReadout(
                n_oscillators=int(n_oscillators),
                oscillator_dim=int(lohe_dim),
                height=decoder_in_height,
                width=decoder_in_width,
            )
        else:
            readout = IdentityReadout()
    decoder = ResizeConvDecoder(
        feature_dim=feature_dim,
        output_dim=3 * 32 * 32,
        in_channels=int(decoder_in_channels),
        in_height=decoder_in_height,
        in_width=decoder_in_width,
        out_channels=3,
        num_upsamples=decoder_num_upsamples,
        final_activation="tanh",
        init_output_gain=0.5,
    )
    decoder = torch.compile(decoder)
    return ConditionalImplicitKuramotoGenerator(
        dynamics=dynamics_module,
        readout=readout,
        decoder=decoder,
        class_dropout_prob=float(class_dropout_prob),
        integration_time=1.0,
        num_steps=int(num_steps),
        solver=solver,
    )


def build_imagenet64_model(
    *,
    n_oscillators: int = 16384,
    n_conditional_oscillators: int = 1,
    class_dropout_prob: float = 0.1,
    num_steps: int = 10,
    decoder_in_channels: int | None = None,
    parameterization: Parameterization = "mup",
    relativization: Relativization = "ref_oscillator",
) -> ConditionalImplicitKuramotoGenerator:
    """Build the ImageNet-64 class-conditional model (1000 classes, 10-step euler)."""
    feature_dim = 2 * int(n_oscillators)
    decoder_in_height = 4
    decoder_in_width = 4
    if decoder_in_channels is None:
        spatial_features = decoder_in_height * decoder_in_width
        if feature_dim % spatial_features != 0:
            raise ValueError(
                "2 * n_oscillators must be divisible by "
                f"{spatial_features} when decoder_in_channels is not set."
            )
        decoder_in_channels = feature_dim // spatial_features

    dynamics = ConditionalKuramotoDynamics(
        n_oscillators=int(n_oscillators),
        n_conditional_oscillators=int(n_conditional_oscillators),
        num_classes=1000,
        init_k_scale=1.0,
        init_freq_scale=1.0,
        parameterization=parameterization,
    )
    dynamics = torch.compile(dynamics)
    readout = ReadoutTransform(
        encoding="sin_cos",
        relativization=relativization,
    )
    decoder = ResizeConvDecoder(
        feature_dim=feature_dim,
        output_dim=3 * 64 * 64,
        in_channels=int(decoder_in_channels),
        in_height=decoder_in_height,
        in_width=decoder_in_width,
        out_channels=3,
        num_upsamples=4,
        final_activation="tanh",
        init_output_gain=1.0,
    )
    decoder = torch.compile(decoder)
    return ConditionalImplicitKuramotoGenerator(
        dynamics=dynamics,
        readout=readout,
        decoder=decoder,
        class_dropout_prob=float(class_dropout_prob),
        integration_time=1.0,
        num_steps=int(num_steps),
        solver="euler",
    )


# Only this repo id changes when checkpoints move to the official hub; the
# filenames and names below are stable.
_HF_REPO = "un-ai/Un-0"

# name -> (filename, family)
_PRETRAINED: dict[str, tuple[str, str]] = {
    "cifar10/n1024": ("cifar10_n1024.pt", "cifar10"),
    "cifar10/n2048": ("cifar10_n2048.pt", "cifar10"),
    "cifar10/n4096": ("cifar10_n4096.pt", "cifar10"),
    "imagenet64/n6656": ("imagenet64_n6656.pt", "imagenet64"),
    "imagenet64/n10240": ("imagenet64_n10240.pt", "imagenet64"),
    "imagenet64/n16384": ("imagenet64_n16384.pt", "imagenet64"),
}
PRETRAINED_NAMES = tuple(sorted(_PRETRAINED))


def build_from_config(
    builder: Callable[..., ConditionalImplicitKuramotoGenerator],
    config: dict,
) -> ConditionalImplicitKuramotoGenerator:
    """Build a model with `builder`, passing only the arch keys it accepts.

    Checkpoint configs carry many training-only keys the builders do not take;
    those are dropped, and any arch key absent from `config` falls back to the
    builder's own default.
    """
    accepted = set(inspect.signature(builder).parameters)
    kwargs = {key: value for key, value in config.items() if key in accepted}
    return builder(**kwargs)
