"""Memory system for PCLN: episodic (KV) and semantic (learned slots)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicMemory(nn.Module):
    """Ring-buffer KV store with cosine retrieval of *values* (not keys)."""

    def __init__(self, d_model: int, max_memory_size: int = 128):
        super().__init__()
        self.d_model = d_model
        self.max_memory_size = max_memory_size

        self.register_buffer("memory_keys", torch.zeros(max_memory_size, d_model))
        self.register_buffer("memory_values", torch.zeros(max_memory_size, d_model))
        self.register_buffer("memory_valid", torch.zeros(max_memory_size, dtype=torch.bool))
        self.register_buffer("memory_ptr", torch.tensor(0, dtype=torch.long))

    def store(self, key: torch.Tensor, value: torch.Tensor):
        """Store a key-value pair (no gradients through the buffer)."""
        with torch.no_grad():
            if key.dim() == 1:
                key = key.unsqueeze(0)
                value = value.unsqueeze(0)

            batch_size = key.size(0)
            for i in range(batch_size):
                ptr = int(self.memory_ptr.item())
                self.memory_keys[ptr] = key[i]
                self.memory_values[ptr] = value[i]
                self.memory_valid[ptr] = True
                self.memory_ptr = (self.memory_ptr + 1) % self.max_memory_size

    def retrieve(self, query: torch.Tensor, topk: int = 1) -> torch.Tensor:
        """Retrieve values whose keys are closest to `query`.

        Args:
            query: (d_model,), (batch, d_model), or (batch, seq, d_model)
            topk: number of matches to mix
        Returns:
            (d_model,) / (batch, d_model) / (batch, seq, d_model) weighted values
        """
        squeeze_batch = query.dim() == 1
        squeeze_time = query.dim() < 3
        if query.dim() == 1:
            query = query.view(1, 1, -1)
        elif query.dim() == 2:
            query = query.unsqueeze(1)

        batch, seq_len, _ = query.shape
        n_valid = int(self.memory_valid.sum().item())
        if n_valid == 0:
            out = torch.zeros(batch, seq_len, self.d_model, device=query.device, dtype=query.dtype)
            if squeeze_time:
                out = out.squeeze(1)
            if squeeze_batch:
                out = out.squeeze(0)
            return out

        keys = self.memory_keys
        values = self.memory_values
        qn = F.normalize(query, dim=-1)
        kn = F.normalize(keys, dim=-1)
        scores = torch.matmul(qn, kn.transpose(0, 1))
        scores = scores.masked_fill(~self.memory_valid.view(1, 1, -1), -1e9)
        k = min(int(topk), n_valid)
        weights, indices = torch.topk(scores, k=k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        gathered = values[indices]
        retrieved = (gathered * weights.unsqueeze(-1)).sum(dim=-2)

        if squeeze_time:
            retrieved = retrieved.squeeze(1)
        if squeeze_batch:
            retrieved = retrieved.squeeze(0)
        return retrieved


class SemanticMemory(nn.Module):
    """Learned slots retrieved by soft attention over the query."""

    def __init__(self, num_slots: int = 64, d_model: int = 128):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        self.slots = nn.Embedding(num_slots, d_model)
        self.query_proj = nn.Linear(d_model, d_model)

    def retrieve(self, query: torch.Tensor, topk: int = 1) -> torch.Tensor:
        """Args:
            query: (batch, d_model) or (batch, seq, d_model)
        Returns:
            same leading dims as query, last dim d_model
        """
        del topk
        q = self.query_proj(query)
        slots = self.slots.weight
        scores = torch.matmul(q, slots.transpose(0, 1))
        scores = F.softmax(scores, dim=-1)
        return torch.matmul(scores, slots)


class MemoryModule(nn.Module):
    """Episodic + semantic memory. Retrieve first, then optionally store."""

    def __init__(
        self,
        d_model: int,
        episodic_size: int = 128,
        semantic_slots: int = 64,
        retrieve_topk: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.retrieve_topk = retrieve_topk
        self.episodic = EpisodicMemory(d_model, episodic_size)
        self.semantic = SemanticMemory(semantic_slots, d_model)

    def forward(
        self,
        latent: torch.Tensor,
        store: bool = False,
        surprise: torch.Tensor | None = None,
        surprise_threshold: float = 0.0,
    ) -> torch.Tensor:
        """Retrieve per-token memory; optionally write the sequence after retrieve.

        Args:
            latent: (batch, seq_len, d_model) or (batch, d_model)
            store: if True, write key=mean, value=last token (or the vector itself)
            surprise: scalar or tensor; store is skipped when below threshold
            surprise_threshold: 0 disables the gate (store whenever `store` is True)
        Returns:
            (batch, seq_len, d_model) fused memory, or (batch, 1, d_model) if 2D input
        """
        if latent.dim() == 3:
            per_token = latent
            store_key = latent.mean(dim=1)
            store_value = latent[:, -1, :]
        else:
            per_token = latent.unsqueeze(1)
            store_key = latent
            store_value = latent

        episodic = self.episodic.retrieve(per_token, topk=self.retrieve_topk)
        semantic = self.semantic.retrieve(per_token)
        retrieved = 0.5 * (episodic + semantic)

        should_store = store
        if should_store and surprise is not None and surprise_threshold > 0:
            should_store = bool((surprise > surprise_threshold).reshape(-1).any().item())
        if should_store:
            self.episodic.store(store_key, store_value)

        return retrieved
