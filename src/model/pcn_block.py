"""A minimal Predictive Coding Network (PCN) block implementation.

This implementation is deliberately lightweight and works with either
NumPy (default) or PyTorch (if available). It's intended for quick
experimentation and smoke-testing — not for production training.
"""
from __future__ import annotations

import math
import typing as t

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

import numpy as np


class PredictiveCodingBlock:
    """Minimal PCN block.

    Args:
        dim: dimensionality of the latent state.
        use_torch: prefer torch tensors when True and torch is available.
    """

    def __init__(self, dim: int, use_torch: bool = False, seed: int = 0):
        self.dim = int(dim)
        self.use_torch = use_torch and _TORCH_AVAILABLE
        self.rng = np.random.RandomState(seed)

        # Initialize simple linear generative weights (z_hat = W z + b)
        W = self.rng.normal(scale=0.1, size=(self.dim, self.dim)).astype(np.float32)
        b = np.zeros((self.dim,), dtype=np.float32)

        if self.use_torch:
            self.W = torch.nn.Parameter(torch.from_numpy(W))
            self.b = torch.nn.Parameter(torch.from_numpy(b))
        else:
            self.W = W
            self.b = b

    def generative(self, z):
        """Compute generative prediction z_hat from latent z."""
        if self.use_torch:
            return z.matmul(self.W.t()) + self.b
        else:
            return z.dot(self.W.T) + self.b

    def refine(self, z, K: int = 3, alpha: float = 0.1):
        """Iterative refinement loop: z <- z - alpha * (z - z_hat).

        Works with NumPy arrays or torch tensors depending on `use_torch`.
        Returns the refined latent state.
        """
        if self.use_torch:
            for _ in range(K):
                z_hat = self.generative(z)
                err = z - z_hat
                z = z - alpha * err
            return z
        else:
            z = z.astype(np.float32)
            for _ in range(K):
                z_hat = self.generative(z)
                err = z - z_hat
                z = z - float(alpha) * err
            return z

    def random_state(self, batch_size: int = 1):
        """Return a random latent state (batch_size, dim)."""
        if self.use_torch:
            return torch.randn(batch_size, self.dim)
        else:
            return self.rng.randn(batch_size, self.dim).astype(np.float32)


def demo_numpy_run():
    block = PredictiveCodingBlock(dim=8, use_torch=False, seed=1)
    z = block.random_state(batch_size=4)
    refined = block.refine(z, K=5, alpha=0.2)
    print("NumPy demo — refined shape:", refined.shape)


if __name__ == "__main__":
    demo_numpy_run()
