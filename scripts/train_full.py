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

from src.model import build_pcln
from src.data import (
    get_dummy_dataloader,
    get_wikitext2_dataloader,
    get_wikitext2_char_dataloader,
)


def _resolve_max_samples(n: int) -> int | None:
    """0 or negative means use all available sequences."""
    return None if n <= 0 else n


def _should_stop_early(worse_epochs: int, patience: int) -> bool:
    """True when validation has not improved for `patience` epochs. patience<=0 disables."""
    if patience <= 0:
        return False
    return worse_epochs >= patience


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

        self.model = build_pcln(args, self.vocab_size, default_causal=True).to(self.device)

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
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=float(getattr(args, "label_smoothing", 0.0) or 0.0)
        )

        # State
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.worse_epochs = 0

        if getattr(args, "resume", False):
            self._load_resume_checkpoint()

    def _compute_main_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Next-token loss at every position (or legacy single-target batches)."""
        if targets.dim() == 1:
            return self.criterion(logits[:, -1, :], targets)
        return self.criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

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
                    max_samples=_resolve_max_samples(self.args.num_train_samples),
                    shuffle=True,
                    chunk_stride=self.args.chunk_stride,
                )
                val_loader, _, _ = get_char_level_dataloader(
                    file_path=self.args.data_file,
                    seq_len=self.args.seq_len,
                    batch_size=self.args.batch_size,
                    max_samples=_resolve_max_samples(self.args.num_val_samples),
                    shuffle=False,
                    char2id=self.vocab_mappings.get("char2id"),
                    chunk_stride=self.args.chunk_stride,
                )
            else:
                train_loader, self.vocab_size, self.vocab_mappings = get_text_file_dataloader(
                    file_path=self.args.data_file,
                    seq_len=self.args.seq_len,
                    batch_size=self.args.batch_size,
                    vocab_size=self.args.vocab_size,
                    max_samples=_resolve_max_samples(self.args.num_train_samples),
                    shuffle=True,
                    chunk_stride=self.args.chunk_stride,
                )
                val_loader, _, _ = get_text_file_dataloader(
                    file_path=self.args.data_file,
                    seq_len=self.args.seq_len,
                    batch_size=self.args.batch_size,
                    vocab_size=self.args.vocab_size,
                    max_samples=_resolve_max_samples(self.args.num_val_samples),
                    shuffle=False,
                    word2id=self.vocab_mappings.get("word2id"),
                    chunk_stride=self.args.chunk_stride,
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
                    max_samples=_resolve_max_samples(self.args.num_train_samples),
                    shuffle=True,
                    chunk_stride=self.args.chunk_stride,
                )
                val_loader, _, _ = get_wikitext2_char_dataloader(
                    split="validation",
                    seq_len=self.args.seq_len,
                    batch_size=self.args.batch_size,
                    max_samples=_resolve_max_samples(self.args.num_val_samples),
                    shuffle=False,
                    char2id=self.vocab_mappings.get("char2id"),
                    chunk_stride=self.args.chunk_stride,
                )
            else:
                train_loader, self.vocab_size, self.vocab_mappings = get_wikitext2_dataloader(
                    split="train",
                    seq_len=self.args.seq_len,
                    batch_size=self.args.batch_size,
                    vocab_size=self.args.vocab_size,
                    max_samples=_resolve_max_samples(self.args.num_train_samples),
                    shuffle=True,
                    chunk_stride=self.args.chunk_stride,
                )
                val_loader, _, _ = get_wikitext2_dataloader(
                    split="validation",
                    seq_len=self.args.seq_len,
                    batch_size=self.args.batch_size,
                    vocab_size=self.args.vocab_size,
                    max_samples=_resolve_max_samples(self.args.num_val_samples),
                    shuffle=False,
                    word2id=self.vocab_mappings.get("word2id"),
                    chunk_stride=self.args.chunk_stride,
                )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log(f"Vocab size: {self.vocab_size}")
        self.log(f"Tokenization: {self.vocab_mappings.get('tokenization', 'unknown')}")
        self.log(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _load_resume_checkpoint(self):
        """Load best or latest checkpoint and continue from the next epoch."""
        best_path = self.checkpoint_dir / "best_model.pt"
        candidates = list(self.checkpoint_dir.glob("checkpoint_epoch*.pt"))

        path = None
        best_epoch = -1
        if best_path.exists():
            ck_meta = torch.load(best_path, map_location="cpu", weights_only=False)
            best_epoch = ck_meta.get("epoch", -1)
            path = best_path

        for candidate in candidates:
            epoch_num = int(candidate.stem.replace("checkpoint_epoch", ""))
            if epoch_num > best_epoch:
                best_epoch = epoch_num
                path = candidate

        if path is None:
            self.log("Resume requested but no checkpoint found; starting fresh.")
            return

        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model_state"])
        if "optimizer_state" in ck:
            try:
                self.optimizer.load_state_dict(ck["optimizer_state"])
            except Exception as e:
                self.log(f"Could not load optimizer state: {e}")
        self.epoch = ck.get("epoch", -1) + 1
        self.step = ck.get("step", 0)
        self.best_val_loss = ck.get("best_val_loss", float("inf"))
        if self.best_val_loss == float("inf") and self.log_file.exists():
            import re
            matches = re.findall(
                r"val_loss=([\d.]+)",
                self.log_file.read_text(encoding="utf-8", errors="ignore"),
            )
            if matches:
                self.best_val_loss = min(float(x) for x in matches)
        self.log(
            f"Resumed from {path.name} -> starting at epoch {self.epoch + 1}/{self.args.epochs} "
            f"(best val {self.best_val_loss:.4f})"
        )

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

            # Main loss: next-token prediction at each position
            loss_main = self._compute_main_loss(logits, targets)

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

            loss = self._compute_main_loss(logits, targets)

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
            "vocab_mappings": self.vocab_mappings,
            "best_val_loss": self.best_val_loss,
        }

        if getattr(self.args, "save_epoch_checkpoints", False):
            ckpt_path = self.checkpoint_dir / f"checkpoint_epoch{self.epoch}.pt"
            torch.save(checkpoint, ckpt_path)
            self.log(f"Saved checkpoint: {ckpt_path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.log(f"Saved best model: {best_path}")

    def train(self):
        """Main training loop."""
        for epoch in range(self.epoch, self.args.epochs):
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
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.worse_epochs = 0
            else:
                self.worse_epochs += 1
            self.save_checkpoint(is_best=is_best)
            if _should_stop_early(self.worse_epochs, int(getattr(self.args, "patience", 0) or 0)):
                self.log(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no val improvement for {self.worse_epochs} epochs)"
                )
                break

        self.log("\n" + "=" * 80)
        self.log(f"Training completed at {datetime.now().isoformat()}")
        self.log(f"Best val loss: {self.best_val_loss:.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train PCLN model")
    
    # Data
    parser.add_argument("--dataset", default="dummy", choices=["dummy", "wikitext2"])
    parser.add_argument("--use-char-level", action="store_true", default=False, help="Use character-level tokenization instead of word-level")
    parser.add_argument("--data-file", type=str, default=None, help="Path to custom text file (if provided, overrides dataset)")
    parser.add_argument("--num-train-samples", type=int, default=1000,
                        help="Max train sequences; 0 = use all available")
    parser.add_argument("--num-val-samples", type=int, default=100,
                        help="Max val sequences; 0 = use all available")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument(
        "--chunk-stride",
        type=int,
        default=None,
        help="Token stride between training chunks; default seq_len (no overlap). Use seq_len/2 for overlap.",
    )
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from best_model.pt or latest checkpoint_epoch*.pt in checkpoint-dir",
    )
    parser.add_argument(
        "--save-epoch-checkpoints",
        action="store_true",
        help="Save checkpoint_epochN.pt each epoch (uses more disk; default: best only)",
    )
    
    # Sparse MoE
    parser.add_argument("--use-sparse-moe", action="store_true", default=False, help="Use Sparse MoE PCN blocks")
    parser.add_argument("--num-experts", type=int, default=4, help="Number of experts per MoE block")
    parser.add_argument("--top-k-experts", type=int, default=2, help="Number of experts to activate per token")
    
    # Dynamic Neurons
    parser.add_argument("--use-dynamic-neurons", action="store_true", default=False, help="Use dynamic neuron PCN blocks (replaces dense FC)")
    parser.add_argument("--num-neurons", type=int, default=256, help="Number of neurons in dynamic neuron layer")
    parser.add_argument("--top-k-neurons", type=int, default=32, help="Number of neurons to activate per token")

    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True,
                        help="Causal encoder (required for honest next-token LM)")
    parser.add_argument("--tie-embeddings", action=argparse.BooleanOptionalAction, default=True,
                        help="Share input embedding and output projection weights")
    parser.add_argument("--use-temporal-pcn", action="store_true", default=False,
                        help="Next-step latent predictive coding instead of self-reconstruction")
    parser.add_argument("--use-hierarchical-pcn", action="store_true", default=False,
                        help="Slow-timescale causal pooled predictive coding")
    parser.add_argument("--adaptive-k", action="store_true", default=False,
                        help="Stop PCN refinement early at eval when error is small")
    parser.add_argument("--error-stop-threshold", type=float, default=0.01)
    parser.add_argument("--timescale", type=int, default=4, help="Hierarchical pooling window")
    parser.add_argument("--surprise-store-threshold", type=float, default=0.0,
                        help="Only write episodic memory when PCN error exceeds this (0=always on store)")
    parser.add_argument("--memory-topk", type=int, default=4)
    parser.add_argument(
        "--pcn-residual-mode",
        choices=["refined", "sum"],
        default="refined",
        help="refined=LayerNorm(z_refined); sum=legacy LayerNorm(z + z_refined)",
    )
    parser.add_argument(
        "--per-token-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retrieve memory at each position (disable to match old pooled memory)",
    )
    parser.add_argument("--patience", type=int, default=0, help="Early-stop patience in epochs (0=off)")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Positional encoding length (default max(512, seq_len))")

    args = parser.parse_args(argv)
    if args.max_seq_len is None:
        args.max_seq_len = max(512, args.seq_len)
    
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
