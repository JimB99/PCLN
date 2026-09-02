"""Temporal predictive coding for language latents.

Standard PCN in this repo reconstructed each token from itself (a denoiser).
Language is a time series: the generative model should predict z_t from z_<t.

This module implements:
- causal_shift: right-shift latents with a learned start vector
- TemporalPCNBlock: next-step latent prediction + precision-weighted updates
- HierarchicalPCNBlock: slower causal pool predicts local means (Rao–Ballard)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_shift(z: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    """Shift sequence right by one step; position 0 is `start`.

    Args:
        z: (batch, seq_len, d_model)
        start: (d_model,) or (batch, d_model) or (batch, 1, d_model)
    """
    if start.dim() == 1:
        start = start.view(1, 1, -1).expand(z.size(0), 1, z.size(-1))
    elif start.dim() == 2:
        start = start.unsqueeze(1)
    start = start.to(device=z.device, dtype=z.dtype)
    if z.size(1) == 0:
        return start[:, :0]
    return torch.cat([start, z[:, :-1]], dim=1)


def causal_moving_mean(z: torch.Tensor, window: int) -> torch.Tensor:
    """Causal mean of the last `window` tokens, inclusive of the current step."""
    window = max(int(window), 1)
    batch, seq_len, _ = z.shape
    cumulative = torch.cumsum(z, dim=1)
    prev = torch.zeros_like(cumulative)
    if window < seq_len:
        prev[:, window:] = cumulative[:, :-window]
    sums = cumulative - prev
    counts = torch.arange(1, seq_len + 1, device=z.device, dtype=z.dtype)
    counts = counts.clamp(max=window).view(1, seq_len, 1)
    return sums / counts


class TemporalPCNBlock(nn.Module):
    """Refine latents by minimizing next-step prediction error.

    ẑ_t = f(z_{t-1});  e_t = z_t - ẑ_t;  z ← z - α · Π · e
    """

    def __init__(
        self,
        d_model: int,
        K: int = 2,
        alpha: float = 0.1,
        adaptive_k: bool = False,
        error_stop_threshold: float = 0.01,
        use_precision_net: bool = True,
        dropout: float = 0.0,
        residual_mode: str = "refined",
    ):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.alpha = alpha
        self.adaptive_k = adaptive_k
        self.error_stop_threshold = error_stop_threshold
        self.residual_mode = residual_mode

        self.start = nn.Parameter(torch.zeros(d_model))
        self.generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.log_precision = nn.Parameter(torch.zeros(d_model))
        if use_precision_net:
            self.precision_net: nn.Module | None = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.Softplus(),
            )
        else:
            self.precision_net = None
        self.norm = nn.LayerNorm(d_model)

    def _precision(self, error: torch.Tensor) -> torch.Tensor:
        base = F.softplus(self.log_precision) + 0.05
        if self.precision_net is None:
            return base
        return base * (1.0 + self.precision_net(error.abs()))

    def forward(self, z: torch.Tensor, training: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        k_max = self.K if training or self.adaptive_k else self.K + 1
        z_refined = z
        errors_list: list[torch.Tensor] = []

        for _ in range(k_max):
            z_pred = self.generator(causal_shift(z_refined, self.start))
            error = z_refined - z_pred
            errors_list.append(error)
            precision = self._precision(error)
            z_refined = z_refined - self.alpha * precision * error
            if (not training) and self.adaptive_k:
                if error.pow(2).mean() < self.error_stop_threshold:
                    break

        errors = torch.stack(errors_list, dim=1)
        if self.residual_mode == "sum":
            return self.norm(z + z_refined), errors
        return self.norm(z_refined), errors


class HierarchicalPCNBlock(nn.Module):
    """Slow pathway: a causal pooled state predicts the next local mean."""

    def __init__(
        self,
        d_model: int,
        timescale: int = 4,
        K: int = 2,
        alpha: float = 0.1,
        dropout: float = 0.0,
        residual_mode: str = "refined",
    ):
        super().__init__()
        self.d_model = d_model
        self.timescale = max(int(timescale), 1)
        self.K = K
        self.alpha = alpha
        self.residual_mode = residual_mode
        self.start = nn.Parameter(torch.zeros(d_model))
        self.slow_gen = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor, training: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        k_steps = self.K if training else self.K + 1
        z_refined = z
        errors_list: list[torch.Tensor] = []

        for _ in range(k_steps):
            slow = causal_moving_mean(z_refined, self.timescale)
            slow_pred = self.slow_gen(causal_shift(slow, self.start))
            error = slow - slow_pred
            errors_list.append(error)
            z_refined = z_refined - self.alpha * error

        errors = torch.stack(errors_list, dim=1)
        if self.residual_mode == "sum":
            return self.norm(z + z_refined), errors
        return self.norm(z_refined), errors
