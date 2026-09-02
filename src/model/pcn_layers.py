"""Predictive coding refinement layers for PCLN."""

from __future__ import annotations

import torch
import torch.nn as nn


class PCNRefinementLayer(nn.Module):
    """K steps of z ← z - α · (z - f(z)) with learned per-dim damping."""

    def __init__(self, d_model: int, K: int = 2, alpha: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.alpha = alpha

        self.generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.damping = nn.Parameter(torch.ones(d_model) * 0.1)

    def forward(self, z, training: bool = True):
        z_refined = z
        errors_list = []
        k_steps = self.K if training else self.K + 1

        for _ in range(k_steps):
            z_pred = self.generator(z_refined)
            error = z_refined - z_pred
            errors_list.append(error)
            z_refined = z_refined - self.alpha * error * (1.0 + self.damping)

        errors = torch.stack(errors_list, dim=1)
        return z_refined, errors


class PCNBlock(nn.Module):
    """PCN refinement + LayerNorm. Output is the refined state, not z + refined."""

    def __init__(self, d_model: int, K: int = 2, alpha: float = 0.1, residual_mode: str = "refined"):
        super().__init__()
        self.refine_layer = PCNRefinementLayer(d_model, K, alpha)
        self.norm = nn.LayerNorm(d_model)
        self.residual_mode = residual_mode

    def forward(self, z, training: bool = True):
        z_refined, errors = self.refine_layer(z, training=training)
        if self.residual_mode == "sum":
            return self.norm(z + z_refined), errors
        return self.norm(z_refined), errors
