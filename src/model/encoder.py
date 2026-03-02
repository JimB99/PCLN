"""Sparse Transformer Encoder for PCLN.

Fast/amortized feedforward encoder that projects tokens to latent beliefs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional embeddings."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                             -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """Add positional encoding to embeddings.
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + pos_encoding: (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoder(nn.Module):
    """Lightweight transformer encoder for fast inference.
    
    Args:
        vocab_size: vocabulary size.
        d_model: embedding/latent dimension.
        nhead: number of attention heads.
        num_layers: number of transformer layers.
        d_ff: feedforward hidden dimension.
        dropout: dropout rate.
        max_len: max sequence length for positional encoding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        d_ff: int | None = None,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, mask=None):
        """Encode token sequence to latent beliefs.
        
        Args:
            tokens: (batch, seq_len) token indices.
            mask: (batch, seq_len) or None. True = attend, False = mask out.
        Returns:
            latent: (batch, seq_len, d_model) latent belief states.
        """
        # Embed and add positional encoding
        x = self.embedding(tokens) * (self.d_model ** 0.5)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Create attention mask if provided
        if mask is not None:
            # PyTorch uses True for positions to mask out
            attn_mask = ~mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            attn_mask = attn_mask.expand(mask.size(0), 1, mask.size(1), mask.size(1))
        else:
            attn_mask = None

        x = self.encoder(x, src_key_padding_mask=~mask if mask is not None else None)
        x = self.norm(x)
        return x
