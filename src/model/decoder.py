"""Decoder for PCLN: maps refined latent beliefs back to token logits.

Includes options for simple linear decoder or more complex architectures.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleDecoder(nn.Module):
    """Simple linear + softmax decoder.
    
    Maps latent beliefs to vocabulary logits.
    
    Args:
        d_model: dimension of latent beliefs.
        vocab_size: output vocabulary size.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.linear = nn.Linear(d_model, vocab_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent beliefs to token logits.
        
        Args:
            latent: (batch, seq_len, d_model) refined latent beliefs.
        Returns:
            logits: (batch, seq_len, vocab_size) token logits.
        """
        x = self.norm(latent)
        logits = self.linear(x)
        return logits


class MemoryAugmentedDecoder(nn.Module):
    """Decoder that fuses retrieved memories with latent beliefs.
    
    Optionally incorporates episodic/semantic memory before decoding.
    
    Args:
        d_model: dimension of latent beliefs.
        vocab_size: output vocabulary size.
        use_memory_fusion: whether to fuse memory with latent.
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
            # Fusion module: combine latent + memory
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
        
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, latent: torch.Tensor, memory: torch.Tensor | None = None) -> torch.Tensor:
        """Decode with optional memory fusion.
        
        Args:
            latent: (batch, seq_len, d_model) latent beliefs.
            memory: (batch, mem_len, d_model) or None. Retrieved memories.
        Returns:
            logits: (batch, seq_len, vocab_size) token logits.
        """
        if memory is not None and self.use_memory_fusion and memory.numel() > 0:
            # Average memory across retrieval dimension
            mem_avg = memory.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
            
            # Broadcast and concatenate
            latent_expanded = latent  # (batch, seq_len, d_model)
            mem_expanded = mem_avg.expand_as(latent_expanded)  # (batch, seq_len, d_model)
            combined = torch.cat([latent_expanded, mem_expanded], dim=-1)  # (batch, seq_len, 2*d_model)
            
            # Fuse
            fused = self.fusion(combined)
            decoded = self.norm(fused)
        else:
            decoded = self.norm(latent)

        logits = self.linear(decoded)
        return logits
