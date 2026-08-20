"""Full training loop for PCLN with logging and checkpointing.

Usage:
    python scripts/train_full.py --epochs 5 --batch-size 16 --dataset dummy
    python scripts/train_full.py --epochs 20 --batch-size 8 --dataset wikitext2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import PCLN
from src.data import (
    get_dummy_dataloader,
    get_wikitext2_dataloader,
    get_wikitext2_char_dataloader,
)


class Trainer:
    """Training harness for PCLN."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.log_file = self.checkpoint_dir / "training.log"
        self.log("=" * 80)
        self.log(f"PCLN Training started at {datetime.now().isoformat()}")
        self.log(f"Device: {self.device}")
        self.log(f"Args: {args}")

        # Load data
        self._load_data()

        # Build model
        self.model = PCLN(
            vocab_size=self.vocab_size,
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
            use_sparse_moe=args.use_sparse_moe,
            use_dynamic_neurons=args.use_dynamic_neurons,
            num_experts=args.num_experts,
            top_k_experts=args.top_k_experts,
            num_neurons=args.num_neurons,
            top_k_neurons=args.top_k_neurons,
        ).to(self.device)

        self.log(f"Model parameters: {self._count_params()}")

        # Optimizer and loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )
        self.criterion = nn.CrossEntropyLoss()

        # State
        self.step = 0
        self.epoch = 0

    def _load_data(self):
        """Load training and validation data."""
        self.vocab_mappings = {"tokenization": "dummy"}

        if self.args.data_file:
            from src.data import get_text_file_dataloader, get_char_level_dataloader

            data_type = "char-level" if self.args.use_char_level else "word-level"
            self.log(f"Loading custom text file ({data_type}): {self.args.data_file}...")

            if self.args.use_char_level:
                train_loader, self.vocab_size, self.vocab_mappings = get_char_level_dataloader(
                    file_path=self.args.data_file,
                    seq_len=self.args.seq_len,
                    batch_size=self.args.batch_size,
                    max_samples=self.args.num_train_samples,
                    shuffle=True,
                )
                val_loader, _, _ = get_char_level_dataloader(
                    file_path=self.args.data_file,
                    seq_len=self.args.seq_len,
                    batch_size=self.args.batch_size,
                    max_samples=self.args.num_val_samples,
                    shuffle=False,
                )
            else:
                train_loader, self.vocab_size, self.vocab_mappings = get_text_file_dataloader(
                    file_path=self.args.data_file,
                    seq_len=self.args.seq_len,
                    batch_size=self.args.batch_size,
                    vocab_size=self.args.vocab_size,
                    max_samples=self.args.num_train_samples,
                    shuffle=True,
                )
                val_loader, _, _ = get_text_file_dataloader(
                    file_path=self.args.data_file,
                    seq_len=self.args.seq_len,
                    batch_size=self.args.batch_size,
                    vocab_size=self.args.vocab_size,
                    max_samples=self.args.num_val_samples,
                    shuffle=False,
                )
        elif self.args.dataset == "dummy":
            self.log("Loading dummy dataset...")
            train_loader, self.vocab_size, self.vocab_mappings = get_dummy_dataloader(
                num_samples=self.args.num_train_samples,
                seq_len=self.args.seq_len,
                batch_size=self.args.batch_size,
                vocab_size=self.args.vocab_size,
                shuffle=True,
            )
            val_loader, _, _ = get_dummy_dataloader(
                num_samples=self.args.num_val_samples,
                seq_len=self.args.seq_len,
                batch_size=self.args.batch_size,
                vocab_size=self.args.vocab_size,
                shuffle=False,
            )
        else:
            tokenization = "char-level" if self.args.use_char_level else "word-level"
            self.log(f"Loading WikiText-2 dataset ({tokenization})...")
            if self.args.use_char_level:
                train_loader, self.vocab_size, self.vocab_mappings = get_wikitext2_char_dataloader(
                    split="train",
                    seq_len=self.args.seq_len,
                    batch_size=self.args.batch_size,
                    max_samples=self.args.num_train_samples,
                    shuffle=True,
                )
                val_loader, _, _ = get_wikitext2_char_dataloader(
                    split="validation",
                    seq_len=self.args.seq_len,
                    batch_size=self.args.batch_size,
                    max_samples=self.args.num_val_samples,
                    shuffle=False,
                )
            else:
                train_loader, self.vocab_size, self.vocab_mappings = get_wikitext2_dataloader(
                    split="train",
                    seq_len=self.args.seq_len,
                    batch_size=self.args.batch_size,
                    vocab_size=self.args.vocab_size,
                    max_samples=self.args.num_train_samples,
                    shuffle=True,
                )
                val_loader, _, _ = get_wikitext2_dataloader(
                    split="validation",
                    seq_len=self.args.seq_len,
                    batch_size=self.args.batch_size,
                    vocab_size=self.args.vocab_size,
                    max_samples=self.args.num_val_samples,
                    shuffle=False,
                )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log(f"Vocab size: {self.vocab_size}")
        self.log(f"Tokenization: {self.vocab_mappings.get('tokenization', 'unknown')}")
        self.log(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def log(self, msg: str):
        """Log message to console and file."""
        print(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")

    def train_epoch(self):
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        total_error_loss = 0.0
        total_moe_loss = 0.0
        num_batches = 0

        for batch_idx, (tokens, targets) in enumerate(self.train_loader):
            tokens = tokens.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(tokens, return_errors=True)
            logits = output["logits"]  # (batch, seq_len, vocab_size)
            errors = output["errors"]
            load_balance_loss = output["load_balance_loss"]

            # Main loss: next-token prediction
            # Targets shape is (batch,), need to match with logits shape
            # Use the last token's logits for prediction
            logits_last = logits[:, -1, :]  # (batch, vocab_size)
            loss_main = self.criterion(logits_last, targets)

            # Auxiliary loss: minimize PCN prediction errors
            loss_error = 0.0
            if errors:
                for error_tensor in errors:
                    loss_error += (error_tensor ** 2).mean()
                loss_error = loss_error / len(errors)

            # Combined loss: main + PCN error + MoE load-balance
            total_task_loss = (
                loss_main
                + self.args.lambda_error * loss_error
                + self.args.lambda_moe * load_balance_loss
            )

            total_task_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            total_loss += loss_main.item()
            total_error_loss += loss_error.item() if isinstance(loss_error, torch.Tensor) else loss_error
            total_moe_loss += load_balance_loss.item()
            num_batches += 1
            self.step += 1

            if (batch_idx + 1) % self.args.log_interval == 0:
                avg_loss = total_loss / num_batches
                avg_error = total_error_loss / num_batches
                avg_moe = total_moe_loss / num_batches
                self.log(
                    f"Epoch {self.epoch+1} Batch {batch_idx+1}/{len(self.train_loader)} "
                    f"loss={avg_loss:.4f} error_loss={avg_error:.4f} moe_loss={avg_moe:.4f} step={self.step}"
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for tokens, targets in self.val_loader:
            tokens = tokens.to(self.device)
            targets = targets.to(self.device)

            output = self.model(tokens, return_errors=False)
            logits = output["logits"]  # (batch, seq_len, vocab_size)

            # Use last token logits for prediction
            logits_last = logits[:, -1, :]  # (batch, vocab_size)
            loss = self.criterion(logits_last, targets)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "args": self.args,
            "vocab_mappings": self.vocab_mappings,  # Include vocab for text-based chat
        }

        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch{self.epoch}.pt"
        torch.save(checkpoint, ckpt_path)
        self.log(f"Saved checkpoint: {ckpt_path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.log(f"Saved best model: {best_path}")

    def train(self):
        """Main training loop."""
        best_val_loss = float("inf")

        for epoch in range(self.args.epochs):
            self.epoch = epoch
            self.log(f"\n--- Epoch {epoch+1}/{self.args.epochs} ---")

            # Train
            start_time = time.time()
            train_loss = self.train_epoch()
            self.scheduler.step()
            train_time = time.time() - start_time

            # Validate
            val_loss = self.validate()
            self.log(
                f"Epoch {epoch+1} finished | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"time={train_time:.2f}s"
            )

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            self.save_checkpoint(is_best=is_best)

        self.log("\n" + "=" * 80)
        self.log(f"Training completed at {datetime.now().isoformat()}")
        self.log(f"Best val loss: {best_val_loss:.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train PCLN model")
    
    # Data
    parser.add_argument("--dataset", default="dummy", choices=["dummy", "wikitext2"])
    parser.add_argument("--use-char-level", action="store_true", default=False, help="Use character-level tokenization instead of word-level")
    parser.add_argument("--data-file", type=str, default=None, help="Path to custom text file (if provided, overrides dataset)")
    parser.add_argument("--num-train-samples", type=int, default=1000)
    parser.add_argument("--num-val-samples", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    
    # Model
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-pcn-blocks", type=int, default=1)
    parser.add_argument("--K-pcn", type=int, default=2, help="PCN refinement steps")
    parser.add_argument("--alpha-pcn", type=float, default=0.1, help="PCN step size")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--episodic-memory-size", type=int, default=64)
    parser.add_argument("--semantic-slots", type=int, default=32)
    
    # Training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-error", type=float, default=0.1, help="Weight for PCN error loss")
    parser.add_argument("--lambda-moe", type=float, default=0.1, help="Weight for MoE load-balance loss")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    
    # Sparse MoE
    parser.add_argument("--use-sparse-moe", action="store_true", default=False, help="Use Sparse MoE PCN blocks")
    parser.add_argument("--num-experts", type=int, default=4, help="Number of experts per MoE block")
    parser.add_argument("--top-k-experts", type=int, default=2, help="Number of experts to activate per token")
    
    # Dynamic Neurons
    parser.add_argument("--use-dynamic-neurons", action="store_true", default=False, help="Use dynamic neuron PCN blocks (replaces dense FC)")
    parser.add_argument("--num-neurons", type=int, default=256, help="Number of neurons in dynamic neuron layer")
    parser.add_argument("--top-k-neurons", type=int, default=32, help="Number of neurons to activate per token")
    
    args = parser.parse_args(argv)
    
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
