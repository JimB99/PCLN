"""Memory system for PCLN: episodic and semantic memory.

Stores and retrieves compressed latent states for improved reasoning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicMemory(nn.Module):
    """Episodic memory: stores compressed latent belief snapshots.
    
    Simple KV-store implementation using attention-based retrieval.
    
    Args:
        d_model: dimension of latent states.
        max_memory_size: maximum number of stored episodes.
    """

    def __init__(self, d_model: int, max_memory_size: int = 128):
        super().__init__()
        self.d_model = d_model
        self.max_memory_size = max_memory_size
        
        # Memory storage (will be updated dynamically)
        self.register_buffer("memory_keys", torch.zeros(max_memory_size, d_model))
        self.register_buffer("memory_values", torch.zeros(max_memory_size, d_model))
        self.register_buffer("memory_valid", torch.zeros(max_memory_size, dtype=torch.bool))
        self.register_buffer("memory_ptr", torch.tensor(0, dtype=torch.long))

    def store(self, key: torch.Tensor, value: torch.Tensor):
        """Store a key-value pair in memory (without gradients).
        
        Args:
            key: (d_model,) or (batch, d_model) key for retrieval.
            value: (d_model,) or (batch, d_model) value to store.
        """
        with torch.no_grad():
            if key.dim() == 1:
                key = key.unsqueeze(0)
                value = value.unsqueeze(0)

            batch_size = key.size(0)
            for i in range(batch_size):
                ptr = self.memory_ptr.item()
                self.memory_keys[ptr] = key[i]
                self.memory_values[ptr] = value[i]
                self.memory_valid[ptr] = True
                self.memory_ptr = (self.memory_ptr + 1) % self.max_memory_size

    def retrieve(self, query: torch.Tensor, topk: int = 1) -> torch.Tensor:
        """Retrieve values similar to query.
        
        Args:
            query: (batch, d_model) or (d_model,) query vector.
            topk: number of top matches to retrieve.
        Returns:
            retrieved_values: (batch, topk, d_model) or (topk, d_model) retrieved values.
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = query.size(0)
        topk = min(topk, self.memory_valid.sum().item())

        if topk == 0:
            # Return zeros if no valid memories
            return torch.zeros(batch_size, 1, self.d_model, device=query.device)

        # Compute similarities
        valid_keys = self.memory_keys[self.memory_valid]
        similarities = torch.matmul(query, valid_keys.t())  # (batch, num_valid)
        
        # Get top-k
        _, top_indices = torch.topk(similarities, k=topk, dim=1)
        retrieved = valid_keys[top_indices]  # (batch, topk, d_model)

        if squeeze_output:
            retrieved = retrieved.squeeze(0)

        return retrieved


class SemanticMemory(nn.Module):
    """Semantic memory: slow-updated learnable memory for facts.
    
    Implemented as a frozen embedding lookup table updated asynchronously.
    
    Args:
        num_slots: number of semantic memory slots.
        d_model: dimension of each slot.
    """

    def __init__(self, num_slots: int = 64, d_model: int = 128):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        
        # Semantic memory slots (learned during training)
        self.slots = nn.Embedding(num_slots, d_model)
        
        # Attention weights for slot retrieval
        self.query_proj = nn.Linear(d_model, d_model)

    def retrieve(self, query: torch.Tensor, topk: int = 1) -> torch.Tensor:
        """Retrieve semantic memory slots via attention.
        
        Args:
            query: (batch, d_model) latent query.
            topk: number of slots to retrieve.
        Returns:
            retrieved: (batch, topk, d_model) retrieved memory slots.
        """
        batch_size = query.size(0)
        
        # Project query
        q = self.query_proj(query)  # (batch, d_model)
        
        # Get all semantic slots
        all_slots = self.slots.weight  # (num_slots, d_model)
        
        # Compute attention scores
        scores = torch.matmul(q, all_slots.t())  # (batch, num_slots)
        scores = F.softmax(scores, dim=1)
        
        # Weighted retrieval
        retrieved = torch.matmul(scores, all_slots)  # (batch, d_model)
        
        return retrieved.unsqueeze(1)  # (batch, 1, d_model)


class MemoryModule(nn.Module):
    """Combined episodic and semantic memory system.
    
    Args:
        d_model: latent dimension.
        episodic_size: max episodic memory capacity.
        semantic_slots: number of semantic memory slots.
    """

    def __init__(
        self,
        d_model: int,
        episodic_size: int = 128,
        semantic_slots: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.episodic = EpisodicMemory(d_model, episodic_size)
        self.semantic = SemanticMemory(semantic_slots, d_model)

    def forward(self, latent: torch.Tensor, store: bool = False) -> torch.Tensor:
        """Retrieve memories relevant to latent state.
        
        Args:
            latent: (batch, seq_len, d_model) or (batch, d_model) latent state.
            store: if True, store latent in episodic memory.
        Returns:
            retrieved: (batch, topk, d_model) concatenated episodic + semantic memories.
        """
        # Average over sequence length if needed
        if latent.dim() == 3:
            latent_query = latent.mean(dim=1)  # (batch, d_model)
        else:
            latent_query = latent

        # Retrieve from both memory systems
        episodic_retrieved = self.episodic.retrieve(latent_query, topk=1)  # (batch, 1, d_model)
        semantic_retrieved = self.semantic.retrieve(latent_query, topk=1)  # (batch, 1, d_model)

        # Store if requested
        if store:
            self.episodic.store(latent_query, latent_query)

        # Concatenate and return
        retrieved = torch.cat([episodic_retrieved, semantic_retrieved], dim=1)  # (batch, 2, d_model)
        return retrieved
