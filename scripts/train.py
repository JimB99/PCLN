"""Minimal training runner for PCLN experiments.

This script runs a tiny demo training loop with GPU support.

Modes:
- Torch + GPU: If torch+CUDA available, runs optimized GPU training (mixed precision).
- Torch + CPU: Falls back to CPU training if GPU not available.
- NumPy: Final fallback for minimal testing without torch.

Run with: `python scripts/train.py --epochs 3 --batch-size 8`
To see device/memory: add --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for local imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from src.model.pcn_block import PredictiveCodingBlock
import numpy as np


def run_torch_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # mixed precision only on GPU
    vocab = 32
    emb_dim = 16
    seq_len = 8
    model_name = "[torch+cuda]" if device.type == "cuda" else "[torch+cpu]"

    if args.verbose:
        print(f"Device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    model = nn.Sequential(
        nn.Embedding(vocab, emb_dim),
        nn.Flatten(),
        nn.Linear(emb_dim * seq_len, vocab),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    for ep in range(args.epochs):
        tokens = torch.randint(0, vocab, (args.batch_size, seq_len), dtype=torch.long, device=device)
        target = tokens[:, 0]

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(tokens)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(tokens)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

        if args.verbose and device.type == "cuda":
            mem_used = torch.cuda.memory_allocated() / 1e9
            print(f"{model_name} epoch={ep+1}/{args.epochs} loss={loss.item():.4f} gpu_mem={mem_used:.3f}GB")
        else:
            print(f"{model_name} epoch={ep+1}/{args.epochs} loss={loss.item():.4f}")


def run_numpy_demo(args):
    block = PredictiveCodingBlock(dim=32, use_torch=False, seed=42)
    batch = 16
    z = block.random_state(batch_size=batch)
    for ep in range(args.epochs):
        refined = block.refine(z, K=5, alpha=0.1)
        # toy loss: mean squared error to zero
        loss = float((refined ** 2).mean())
        print(f"[numpy] epoch={ep+1}/{args.epochs} loss={loss:.6f}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--verbose", action="store_true", help="Print device/memory info.")
    args = parser.parse_args(argv)

    print("Project root:", ROOT)
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            print("Torch + CUDA available — running GPU-optimized training.")
        else:
            print("Torch available (CPU only) — running CPU training.")
        run_torch_demo(args)
    else:
        print("Torch not available — running NumPy PCN demo.")
        run_numpy_demo(args)


if __name__ == "__main__":
    main()
