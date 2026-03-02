"""Full PCLN model: Predictive Coding Language Model.

Combines transformer encoder, PCN refinement, memory, and decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import TransformerEncoder
from .pcn_layers import PCNBlock
from .sparse_moe import SparseMoEPCNBlock
from .dynamic_neurons import DynamicNeuronBlock
from .memory import MemoryModule
from .decoder import MemoryAugmentedDecoder


class PCLN(nn.Module):
    """Full Predictive Coding Language Model.
    
    Architecture:
    1. Token embedding → Transformer encoder (amortized inference)
    2. PCN refinement blocks (iterative belief refinement)
       - Standard: simple feedforward-based PCN
       - Sparse MoE: gated expert routing with pruning
    3. Memory retrieval (episodic + semantic)
    4. Memory-augmented decoder
    
    Args:
        vocab_size: vocabulary size.
        d_model: embedding/latent dimension.
        nhead: number of attention heads in encoder.
        num_encoder_layers: number of transformer encoder layers.
        num_pcn_blocks: number of PCN refinement blocks.
        K_pcn: refinement steps per PCN block during training.
        alpha_pcn: refinement step size.
        d_ff: feedforward dimension in encoder.
        dropout: dropout rate.
        episodic_memory_size: max episodic memory capacity.
        semantic_slots: number of semantic memory slots.
        use_memory: whether to use memory module.
        use_sparse_moe: if True, use Sparse MoE blocks instead of standard PCN.
        use_dynamic_neurons: if True, use dynamic neuron blocks (replaces dense FC).
        num_experts: number of experts per MoE block (if use_sparse_moe=True).
        top_k_experts: number of experts to activate per token (if use_sparse_moe=True).
        num_neurons: number of neurons in dynamic neuron layer (if use_dynamic_neurons=True).
        top_k_neurons: number of neurons to activate per token (if use_dynamic_neurons=True).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_pcn_blocks: int = 1,
        K_pcn: int = 2,
        alpha_pcn: float = 0.1,
        d_ff: int | None = None,
        dropout: float = 0.1,
        episodic_memory_size: int = 128,
        semantic_slots: int = 64,
        use_memory: bool = True,
        use_sparse_moe: bool = False,
        use_dynamic_neurons: bool = False,
        num_experts: int = 4,
        top_k_experts: int = 2,
        num_neurons: int = 256,
        top_k_neurons: int = 32,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_pcn_blocks = num_pcn_blocks
        self.use_memory = use_memory
        self.use_sparse_moe = use_sparse_moe
        self.use_dynamic_neurons = use_dynamic_neurons
        self.K_pcn = K_pcn

        # Encoder: fast/amortized inference
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_seq_len,
        )

        # PCN refinement blocks: choose variant
        # Priority: dynamic_neurons > sparse_moe > standard
        if use_dynamic_neurons:
            self.pcn_blocks = nn.ModuleList([
                DynamicNeuronBlock(
                    d_model=d_model,
                    num_neurons=num_neurons,
                    top_k_neurons=top_k_neurons,
                    K=K_pcn,
                    alpha=alpha_pcn,
                )
                for _ in range(num_pcn_blocks)
            ])
        elif use_sparse_moe:
            self.pcn_blocks = nn.ModuleList([
                SparseMoEPCNBlock(
                    d_model=d_model,
                    num_experts=num_experts,
                    top_k=top_k_experts,
                    K_refine=K_pcn,
                    alpha=alpha_pcn,
                )
                for _ in range(num_pcn_blocks)
            ])
        else:
            self.pcn_blocks = nn.ModuleList([
                PCNBlock(d_model=d_model, K=K_pcn, alpha=alpha_pcn)
                for _ in range(num_pcn_blocks)
            ])

        # Memory module (optional)
        if use_memory:
            self.memory = MemoryModule(
                d_model=d_model,
                episodic_size=episodic_memory_size,
                semantic_slots=semantic_slots,
            )

        # Decoder
        self.decoder = MemoryAugmentedDecoder(
            d_model=d_model,
            vocab_size=vocab_size,
            use_memory_fusion=use_memory,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_errors: bool = False,
    ) -> dict:
        """Forward pass through PCLN.
        
        Args:
            tokens: (batch, seq_len) token indices.
            mask: (batch, seq_len) or None. 1 = attend, 0 = mask out.
            return_errors: if True, return PCN prediction errors for auxiliary loss.
        Returns:
            output dict with keys:
            - logits: (batch, seq_len, vocab_size) output logits.
            - latent: (batch, seq_len, d_model) final refined latent state.
            - errors: (list of tensors) PCN prediction errors if return_errors=True.
            - load_balance_loss: weighted auxiliary loss from MoE blocks (if use_sparse_moe=True).
        """
        # Step 1: Encoder (fast feedforward)
        latent = self.encoder(tokens, mask=mask)  # (batch, seq_len, d_model)

        # Step 2: PCN refinement
        errors_all = []
        load_balance_loss = torch.tensor(0.0, device=latent.device, dtype=latent.dtype)

        for pcn_block in self.pcn_blocks:
            if self.use_dynamic_neurons:
                latent, errors, lb_loss = pcn_block(latent, training=self.training)
                load_balance_loss = load_balance_loss + lb_loss
                errors_all.append(errors)
            elif self.use_sparse_moe:
                latent, errors, lb_loss, expert_usage = pcn_block(
                    latent, training=self.training
                )
                load_balance_loss = load_balance_loss + lb_loss
                errors_all.append(errors)
            else:
                latent, errors = pcn_block(latent, training=self.training)
                errors_all.append(errors)

        # Step 3: Memory retrieval (optional)
        memory = None
        if self.use_memory:
            memory = self.memory(latent, store=self.training)  # (batch, 2, d_model)

        # Step 4: Decode
        logits = self.decoder(latent, memory=memory)  # (batch, seq_len, vocab_size)

        output = {
            "logits": logits,
            "latent": latent,
            "load_balance_loss": load_balance_loss,
        }
        if return_errors:
            output["errors"] = errors_all

        return output

    def generate(
        self,
        tokens: torch.Tensor,
        max_len: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.
        
        Args:
            tokens: (batch, seq_len) initial tokens.
            max_len: maximum length to generate.
            temperature: sampling temperature (higher = more random).
            top_k: if set, use top-k sampling.
        Returns:
            generated: (batch, orig_len + max_len) generated sequence.
        """
        device = tokens.device
        generated = tokens.clone()

        for _ in range(max_len):
            # Forward pass
            output = self(generated, return_errors=False)
            logits = output["logits"][:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            generated = torch.cat([generated, next_token], dim=1)

        return generated
