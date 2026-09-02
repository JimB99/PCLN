"""Dummy LM data should be a shifted sequence, not independent random targets."""

from __future__ import annotations

import unittest

import torch

from src.data import DummyDataset
from scripts.train_full import _should_stop_early


class TestDummyAutoregressive(unittest.TestCase):
    def test_targets_are_inputs_shifted_by_one(self):
        ds = DummyDataset(num_samples=4, seq_len=8, vocab_size=32)
        inp, tgt = ds[0]
        self.assertEqual(inp.shape, (8,))
        self.assertEqual(tgt.shape, (8,))
        self.assertTrue(torch.equal(tgt[:-1], inp[1:]))


class TestEarlyStopping(unittest.TestCase):
    def test_stops_after_patience_without_improvement(self):
        self.assertFalse(_should_stop_early(worse_epochs=2, patience=3))
        self.assertTrue(_should_stop_early(worse_epochs=3, patience=3))
        self.assertFalse(_should_stop_early(worse_epochs=5, patience=0))


if __name__ == "__main__":
    unittest.main()
