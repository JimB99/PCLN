"""Decoder for PCLN: maps refined latent beliefs back to token logits."""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleDecoder(nn.Module):
    """Linear decoder from latents to vocabulary logits."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(d_model, vocab_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(latent))


class MemoryAugmentedDecoder(nn.Module):
    """Fuse retrieved memory with latents, then project to logits.

    Memory may be per-token (batch, seq, d_model) or pooled (batch, mem, d_model).
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        use_memory_fusion: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_memory_fusion = use_memory_fusion

        if use_memory_fusion:
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, latent: torch.Tensor, memory: torch.Tensor | None = None) -> torch.Tensor:
        if memory is not None and self.use_memory_fusion and memory.numel() > 0:
            if memory.dim() == 3 and memory.size(1) == latent.size(1):
                mem_expanded = memory
            else:
                mem_avg = memory.mean(dim=1, keepdim=True)
                mem_expanded = mem_avg.expand_as(latent)
            combined = torch.cat([latent, mem_expanded], dim=-1)
            decoded = self.norm(self.fusion(combined))
        else:
            decoded = self.norm(latent)
        return self.linear(decoded)
