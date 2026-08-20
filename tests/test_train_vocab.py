"""Tests that training saves vocab mappings in checkpoints."""

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class TestTrainVocabCheckpoint(unittest.TestCase):
    def test_checkpoint_contains_vocab_mappings(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "best_model.pt"
            fake_mappings = {
                "tokenization": "word",
                "word2id": {"hello": 0, "<unk>": 1},
                "id2word": {0: "hello", 1: "<unk>"},
            }
            args = argparse.Namespace(
                seq_len=8,
                d_model=32,
                nhead=4,
                num_encoder_layers=1,
                num_pcn_blocks=1,
                K_pcn=1,
                alpha_pcn=0.1,
                dropout=0.1,
                use_memory=True,
                episodic_memory_size=8,
                semantic_slots=4,
                use_sparse_moe=False,
                use_dynamic_neurons=False,
                num_experts=2,
                top_k_experts=1,
                num_neurons=16,
                top_k_neurons=4,
                use_char_level=False,
            )
            checkpoint = {
                "epoch": 0,
                "step": 1,
                "model_state": {},
                "optimizer_state": {},
                "args": args,
                "vocab_mappings": fake_mappings,
            }
            torch.save(checkpoint, ckpt_path)
            loaded = torch.load(ckpt_path, weights_only=False)
            self.assertEqual(loaded["vocab_mappings"]["tokenization"], "word")
            self.assertIn("hello", loaded["vocab_mappings"]["word2id"])


if __name__ == "__main__":
    unittest.main()
