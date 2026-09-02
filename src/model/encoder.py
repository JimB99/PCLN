"""Causal Transformer encoder: tokens → latent beliefs."""

from __future__ import annotations

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional embeddings."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """Add positional encoding to embeddings. x: (batch, seq_len, d_model)."""
        return x + self.pe[:, : x.size(1), :]


def _causal_attn_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Boolean mask: True = cannot attend (future positions)."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class TransformerEncoder(nn.Module):
    """Lightweight transformer encoder for amortized inference.

    For language modeling, `causal=True` so position t cannot see tokens > t.
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
        causal: bool = True,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.causal = causal
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
            mask: (batch, seq_len) or None. True/1 = attend, False/0 = pad.
        """
        x = self.embedding(tokens) * (self.d_model ** 0.5)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        seq_len = x.size(1)
        src_mask = _causal_attn_mask(seq_len, x.device) if self.causal else None

        padding = None
        if mask is not None:
            if mask.dtype == torch.bool:
                padding = ~mask
            else:
                padding = mask == 0

        x = self.encoder(x, mask=src_mask, src_key_padding_mask=padding)
        x = self.norm(x)
        return x
