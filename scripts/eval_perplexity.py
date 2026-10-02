#!/usr/bin/env python3
"""Evaluate validation/test perplexity for a trained PCLN checkpoint.

Default evaluation uses non-overlapping chunks (stride = seq_len) so overlapping
training windows cannot inflate or double-count test perplexity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import PCLN, build_pcln
from src.data import get_wikitext2_dataloader, get_wikitext2_char_dataloader


def resolve_eval_chunk_stride(train_args, use_train_stride: bool = False) -> int:
    """Stride for val/test chunks.

    Default: seq_len (no overlap), ignoring training chunk_stride.
    use_train_stride=True restores the training stride (not for published PPL).
    """
    seq_len = int(getattr(train_args, "seq_len", 64) or 64)
    if use_train_stride:
        stride = getattr(train_args, "chunk_stride", None)
        if stride is not None and int(stride) > 0:
            return int(stride)
        return seq_len
    eval_stride = getattr(train_args, "eval_chunk_stride", None)
    if eval_stride is not None and int(eval_stride) > 0:
        return int(eval_stride)
    return seq_len


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[PCLN, dict]:
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ck["args"]
    state = ck["model_state"]
    vocab_size = state["encoder.embedding.weight"].shape[0]
    model = build_pcln(args, vocab_size, default_causal=False).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ck


def eval_split(
    model: PCLN,
    loader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, int]:
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
    return avg_nll, ppl, total_tokens


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PCLN perplexity evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--use-train-stride",
        action="store_true",
        help="Use the training chunk stride (overlapping). Do not use for published PPL.",
    )
    parser.add_argument(
        "--eval-chunk-stride",
        type=int,
        default=None,
        help="Override eval stride. Default is seq_len (non-overlapping).",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt = Path(args.checkpoint)
    model, ck = load_model(ckpt, device)
    train_args = ck["args"]
    vocab_maps = ck.get("vocab_mappings", {})
    tokenization = vocab_maps.get("tokenization", "unknown")
    if tokenization == "dummy":
        print("ERROR: checkpoint tokenization is dummy; refusing eval.", file=sys.stderr)
        return 1

    if args.eval_chunk_stride is not None and args.eval_chunk_stride > 0:
        chunk_stride = int(args.eval_chunk_stride)
    else:
        chunk_stride = resolve_eval_chunk_stride(train_args, use_train_stride=args.use_train_stride)

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
    nll, ppl, ntok = eval_split(model, loader, device, criterion)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    causal = bool(getattr(train_args, "causal", False))
    vocab_size = int(model.encoder.embedding.weight.shape[0])

    print(f"Checkpoint: {ckpt}")
    print(f"Split: {args.split}")
    print(f"Causal: {causal}")
    print(f"Parameters: {n_params}")
    print(f"Vocab size: {vocab_size}")
    print(f"Tokenization: {tokenization}")
    print(f"Eval stride: {chunk_stride}")
    print(f"Tokens: {ntok}")
    print(f"NLL: {nll:.4f}")
    print(f"Perplexity: {ppl:.2f}")

    payload = {
        "checkpoint": str(ckpt),
        "split": args.split,
        "causal": causal,
        "parameters": n_params,
        "vocab_size": vocab_size,
        "tokenization": tokenization,
        "eval_stride": chunk_stride,
        "tokens": ntok,
        "nll": nll,
        "perplexity": ppl,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
