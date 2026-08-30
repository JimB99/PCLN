"""Tests for full-sequence next-token targets and loss."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.data import TextFileDataset, get_text_file_dataloader, _build_sequence_chunks
from scripts.train_full import Trainer, _resolve_max_samples


class TestFullSequenceTargets(unittest.TestCase):
    def test_dataset_returns_seq_len_targets(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("one two three four five six seven eight nine ten " * 20)
            path = f.name
        try:
            ds = TextFileDataset(path, seq_len=4, vocab_size=32, max_samples=5)
            inputs, targets = ds[0]
            self.assertEqual(inputs.shape[0], 4)
            self.assertEqual(targets.shape[0], 4)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_resolve_max_samples(self):
        self.assertIsNone(_resolve_max_samples(0))
        self.assertIsNone(_resolve_max_samples(-1))
        self.assertEqual(_resolve_max_samples(100), 100)

    def test_compute_main_loss_full_sequence(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("alpha beta gamma delta epsilon " * 30)
            path = f.name
        try:
            loader, vocab_size, _ = get_text_file_dataloader(
                file_path=path, seq_len=8, batch_size=4, vocab_size=64, max_samples=10
            )
            tokens, targets = next(iter(loader))
            self.assertEqual(tokens.shape[1], 8)
            self.assertEqual(targets.shape[1], 8)

            class Args:
                dataset = "dummy"
                use_char_level = False
                data_file = None
                num_train_samples = 10
                num_val_samples = 5
                seq_len = 8
                vocab_size = 64
                batch_size = 4
                d_model = 64
                nhead = 4
                num_encoder_layers = 1
                num_pcn_blocks = 1
                K_pcn = 1
                alpha_pcn = 0.1
                dropout = 0.1
                use_memory = False
                episodic_memory_size = 32
                semantic_slots = 16
                epochs = 1
                learning_rate = 1e-3
                weight_decay = 1e-5
                grad_clip = 1.0
                lambda_error = 0.1
                lambda_moe = 0.1
                log_interval = 100
                checkpoint_dir = tempfile.mkdtemp()
                use_sparse_moe = False
                num_experts = 4
                top_k_experts = 2
                use_dynamic_neurons = False
                num_neurons = 64
                top_k_neurons = 8
                chunk_stride = None
                save_epoch_checkpoints = False
                resume = False

            trainer = Trainer(Args())
            logits = torch.randn(4, 8, trainer.vocab_size)
            targets = torch.randint(0, trainer.vocab_size, (4, 8))
            loss = trainer._compute_main_loss(logits, targets)
            self.assertTrue(torch.isfinite(loss))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_chunk_stride_increases_sequences(self):
        tokens = list(range(200))
        no_overlap = _build_sequence_chunks(tokens, seq_len=8, stride=None)
        overlap = _build_sequence_chunks(tokens, seq_len=8, stride=4)
        self.assertGreater(len(overlap), len(no_overlap))

    def test_text_file_dataset_stride(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 40)
            path = f.name
        try:
            ds_full = TextFileDataset(path, seq_len=8, vocab_size=64, chunk_stride=8)
            ds_half = TextFileDataset(path, seq_len=8, vocab_size=64, chunk_stride=4, word2id=ds_full.word2id)
            self.assertGreater(len(ds_half.data), len(ds_full.data))
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
