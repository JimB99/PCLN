"""Build PCLN from argparse/checkpoint namespaces with stable field names."""

from __future__ import annotations

from typing import Any

from .pcn_model import PCLN


def pcln_kwargs_from_args(
    args: Any,
    vocab_size: int,
    *,
    default_causal: bool = True,
) -> dict:
    """Map train/chat args onto PCLN constructor kwargs.

    `default_causal` is True for new training and False when loading old
    checkpoints that were trained bidirectionally.
    """
    seq_len = int(getattr(args, "seq_len", 512) or 512)
    raw_max = getattr(args, "max_seq_len", None)
    max_seq_len = int(raw_max) if raw_max else max(512, seq_len)
    return {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_encoder_layers": args.num_encoder_layers,
        "num_pcn_blocks": args.num_pcn_blocks,
        "K_pcn": args.K_pcn,
        "alpha_pcn": args.alpha_pcn,
        "dropout": args.dropout,
        "use_memory": args.use_memory,
        "episodic_memory_size": args.episodic_memory_size,
        "semantic_slots": args.semantic_slots,
        "use_sparse_moe": getattr(args, "use_sparse_moe", False),
        "use_dynamic_neurons": getattr(args, "use_dynamic_neurons", False),
        "num_experts": getattr(args, "num_experts", 4),
        "top_k_experts": getattr(args, "top_k_experts", 2),
        "num_neurons": getattr(args, "num_neurons", 256),
        "top_k_neurons": getattr(args, "top_k_neurons", 32),
        "max_seq_len": max_seq_len,
        "causal": getattr(args, "causal", default_causal),
        "tie_embeddings": getattr(args, "tie_embeddings", False),
        "use_temporal_pcn": getattr(args, "use_temporal_pcn", False),
        "use_hierarchical_pcn": getattr(args, "use_hierarchical_pcn", False),
        "adaptive_k": getattr(args, "adaptive_k", False),
        "error_stop_threshold": getattr(args, "error_stop_threshold", 0.01),
        "timescale": getattr(args, "timescale", 4),
        "surprise_store_threshold": getattr(args, "surprise_store_threshold", 0.0),
        "memory_topk": getattr(args, "memory_topk", 4),
        "pcn_residual_mode": getattr(
            args,
            "pcn_residual_mode",
            "refined" if default_causal else "sum",
        ),
        "per_token_memory": getattr(args, "per_token_memory", default_causal),
    }


def build_pcln(args: Any, vocab_size: int, *, default_causal: bool = True) -> PCLN:
    return PCLN(**pcln_kwargs_from_args(args, vocab_size, default_causal=default_causal))
