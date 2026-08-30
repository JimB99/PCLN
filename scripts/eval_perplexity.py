#!/usr/bin/env python3
"""Evaluate validation/test perplexity for a trained PCLN checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import PCLN
from src.data import get_wikitext2_dataloader, get_wikitext2_char_dataloader, get_char_level_dataloader


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[PCLN, dict]:
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ck["args"]
    state = ck["model_state"]
    vocab_size = state["encoder.embedding.weight"].shape[0]
    model = PCLN(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_pcn_blocks=args.num_pcn_blocks,
        K_pcn=args.K_pcn,
        alpha_pcn=args.alpha_pcn,
        dropout=args.dropout,
        use_memory=args.use_memory,
        episodic_memory_size=args.episodic_memory_size,
        semantic_slots=args.semantic_slots,
        use_sparse_moe=getattr(args, "use_sparse_moe", False),
        use_dynamic_neurons=getattr(args, "use_dynamic_neurons", False),
        num_experts=getattr(args, "num_experts", 4),
        top_k_experts=getattr(args, "top_k_experts", 2),
        num_neurons=getattr(args, "num_neurons", 256),
        top_k_neurons=getattr(args, "top_k_neurons", 32),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ck


def eval_split(
    model: PCLN,
    loader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for tokens, targets in loader:
            tokens = tokens.to(device)
            targets = targets.to(device)
            logits = model(tokens, return_errors=False)["logits"]
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            ntok = targets.numel()
            total_loss += loss.item() * ntok
            total_tokens += ntok
    avg_nll = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_nll, 20.0))
    return avg_nll, ppl


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PCLN perplexity evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt = Path(args.checkpoint)
    model, ck = load_model(ckpt, device)
    train_args = ck["args"]
    vocab_maps = ck.get("vocab_mappings", {})
    chunk_stride = getattr(train_args, "chunk_stride", None)

    if getattr(train_args, "use_char_level", False):
        loader, _, _ = get_wikitext2_char_dataloader(
            split=args.split,
            seq_len=train_args.seq_len,
            batch_size=train_args.batch_size,
            max_samples=None,
            shuffle=False,
            char2id=vocab_maps.get("char2id"),
            chunk_stride=chunk_stride,
        )
    else:
        loader, _, _ = get_wikitext2_dataloader(
            split=args.split,
            seq_len=train_args.seq_len,
            batch_size=train_args.batch_size,
            vocab_size=train_args.vocab_size,
            max_samples=None,
            shuffle=False,
            word2id=vocab_maps.get("word2id"),
            chunk_stride=chunk_stride,
        )

    criterion = nn.CrossEntropyLoss()
    nll, ppl = eval_split(model, loader, device, criterion)
    print(f"Checkpoint: {ckpt}")
    print(f"Split: {args.split}")
    print(f"NLL: {nll:.4f}")
    print(f"Perplexity: {ppl:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
