"""Eval-stride and seed helpers for quoteable WikiText perplexity."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.eval_perplexity import resolve_eval_chunk_stride
from scripts.train_full import set_seed


class TestEvalChunkStride(unittest.TestCase):
    def test_default_ignores_train_overlap(self):
        args = SimpleNamespace(seq_len=128, chunk_stride=64, eval_chunk_stride=None)
        self.assertEqual(resolve_eval_chunk_stride(args, use_train_stride=False), 128)

    def test_use_train_stride_keeps_overlap(self):
        args = SimpleNamespace(seq_len=128, chunk_stride=64, eval_chunk_stride=None)
        self.assertEqual(resolve_eval_chunk_stride(args, use_train_stride=True), 64)

    def test_explicit_eval_stride_wins(self):
        args = SimpleNamespace(seq_len=128, chunk_stride=64, eval_chunk_stride=128)
        self.assertEqual(resolve_eval_chunk_stride(args, use_train_stride=False), 128)


class TestSetSeed(unittest.TestCase):
    def test_set_seed_is_callable(self):
        set_seed(42)


if __name__ == "__main__":
    unittest.main()
