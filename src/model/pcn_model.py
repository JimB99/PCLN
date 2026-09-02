"""Full PCLN model: Predictive Coding Language Model.

Pipeline: causal encoder → PCN refinement → memory → decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import TransformerEncoder
from .pcn_layers import PCNBlock
from .sparse_moe import SparseMoEPCNBlock
from .dynamic_neurons import DynamicNeuronBlock
from .temporal_pcn import HierarchicalPCNBlock, TemporalPCNBlock
from .memory import MemoryModule
from .decoder import MemoryAugmentedDecoder
from .generation import (
    apply_repetition_penalty,
    ban_repeating_ngrams,
    sample_next_token,
)


class PCLN(nn.Module):
    """Predictive Coding Language Model."""

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
        causal: bool = True,
        tie_embeddings: bool = False,
        use_temporal_pcn: bool = False,
        use_hierarchical_pcn: bool = False,
        adaptive_k: bool = False,
        error_stop_threshold: float = 0.01,
        timescale: int = 4,
        surprise_store_threshold: float = 0.0,
        memory_topk: int = 4,
        pcn_residual_mode: str = "refined",
        per_token_memory: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_memory = use_memory
        self.use_sparse_moe = use_sparse_moe
        self.use_dynamic_neurons = use_dynamic_neurons
        self.K_pcn = K_pcn
        self.causal = causal
        self.tie_embeddings = tie_embeddings
        self.surprise_store_threshold = surprise_store_threshold
        self.max_seq_len = max_seq_len
        self.pcn_residual_mode = pcn_residual_mode
        self.per_token_memory = per_token_memory

        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_seq_len,
            causal=causal,
        )

        makers: list = []
        if use_temporal_pcn:
            makers.append(
                lambda: TemporalPCNBlock(
                    d_model=d_model,
                    K=K_pcn,
                    alpha=alpha_pcn,
                    adaptive_k=adaptive_k,
                    error_stop_threshold=error_stop_threshold,
                    dropout=dropout,
                    residual_mode=pcn_residual_mode,
                )
            )
        if use_hierarchical_pcn:
            makers.append(
                lambda: HierarchicalPCNBlock(
                    d_model=d_model,
                    timescale=timescale,
                    K=K_pcn,
                    alpha=alpha_pcn,
                    dropout=dropout,
                    residual_mode=pcn_residual_mode,
                )
            )
        if use_dynamic_neurons:
            makers.append(
                lambda: DynamicNeuronBlock(
                    d_model=d_model,
                    num_neurons=num_neurons,
                    top_k_neurons=top_k_neurons,
                    K=K_pcn,
                    alpha=alpha_pcn,
                    residual_mode=pcn_residual_mode,
                )
            )
        if use_sparse_moe:
            makers.append(
                lambda: SparseMoEPCNBlock(
                    d_model=d_model,
                    num_experts=num_experts,
                    top_k=top_k_experts,
                    K_refine=K_pcn,
                    alpha=alpha_pcn,
                    residual_mode=pcn_residual_mode,
                )
            )
        if not makers:
            blocks = [
                PCNBlock(
                    d_model=d_model,
                    K=K_pcn,
                    alpha=alpha_pcn,
                    residual_mode=pcn_residual_mode,
                )
                for _ in range(num_pcn_blocks)
            ]
        elif len(makers) == 1:
            blocks = [makers[0]() for _ in range(num_pcn_blocks)]
        else:
            blocks = [make() for make in makers]
        self.pcn_blocks = nn.ModuleList(blocks)
        self.num_pcn_blocks = len(blocks)

        if use_memory:
            self.memory = MemoryModule(
                d_model=d_model,
                episodic_size=episodic_memory_size,
                semantic_slots=semantic_slots,
                retrieve_topk=memory_topk,
            )

        self.decoder = MemoryAugmentedDecoder(
            d_model=d_model,
            vocab_size=vocab_size,
            use_memory_fusion=use_memory,
        )
        if tie_embeddings:
            self.decoder.linear.weight = self.encoder.embedding.weight

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_errors: bool = False,
        store_memory: bool = False,
    ) -> dict:
        latent = self.encoder(tokens, mask=mask)

        errors_all = []
        load_balance_loss = tokens.new_zeros(())
        load_balance_loss = load_balance_loss.to(dtype=latent.dtype)

        for pcn_block in self.pcn_blocks:
            result = pcn_block(latent, training=self.training)
            if len(result) == 4:
                latent, errors, lb_loss, _expert_usage = result
                load_balance_loss = load_balance_loss + lb_loss
            elif len(result) == 3:
                latent, errors, lb_loss = result
                load_balance_loss = load_balance_loss + lb_loss
            else:
                latent, errors = result
            errors_all.append(errors)

        memory = None
        if self.use_memory:
            surprise = None
            if errors_all:
                surprise = errors_all[-1].pow(2).mean()
            mem_src = latent if self.per_token_memory else latent.mean(dim=1)
            memory = self.memory(
                mem_src,
                store=store_memory,
                surprise=surprise,
                surprise_threshold=self.surprise_store_threshold,
            )

        logits = self.decoder(latent, memory=memory)
        output = {
            "logits": logits,
            "latent": latent,
            "load_balance_loss": load_balance_loss,
        }
        if return_errors:
            output["errors"] = errors_all
        return output

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_len: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        store_memory: bool = False,
    ) -> torch.Tensor:
        """Autoregressive generation with optional nucleus / repetition controls."""
        generated = tokens.clone()
        history = generated[0].tolist() if generated.size(0) == 1 else []
        max_ctx = self.encoder.pos_encoding.pe.size(1)

        for _ in range(max_len):
            ctx = generated[:, -max_ctx:]
            logits = self(ctx, return_errors=False, store_memory=store_memory)["logits"][:, -1, :]
            if generated.size(0) == 1:
                logits = apply_repetition_penalty(logits, history[-32:], repetition_penalty)
                logits = ban_repeating_ngrams(logits, history, no_repeat_ngram_size)
            next_token = sample_next_token(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            generated = torch.cat([generated, next_token], dim=1)
            if generated.size(0) == 1:
                history.append(int(next_token.item()))

        return generated
