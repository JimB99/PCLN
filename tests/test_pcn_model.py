"""Unit tests for PCLN model."""

import unittest

import torch

from src.model import PCLN
from src.model.dynamic_neurons import DynamicNeuronBlock
from src.model.sparse_moe import SparseMoEPCNBlock


class TestPCLNForward(unittest.TestCase):
    def test_tiny_forward_pass(self):
        model = PCLN(vocab_size=32, d_model=32, nhead=4, num_encoder_layers=1, num_pcn_blocks=1)
        tokens = torch.randint(0, 32, (2, 8))
        out = model(tokens, return_errors=True)
        self.assertEqual(out["logits"].shape, (2, 8, 32))
        self.assertIn("errors", out)
        self.assertIn("load_balance_loss", out)

    def test_composable_blocks_when_both_flags(self):
        model = PCLN(
            vocab_size=32,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            use_dynamic_neurons=True,
            use_sparse_moe=True,
            num_neurons=16,
            top_k_neurons=4,
            num_experts=2,
            top_k_experts=1,
        )
        self.assertEqual(len(model.pcn_blocks), 2)
        self.assertIsInstance(model.pcn_blocks[0], DynamicNeuronBlock)
        self.assertIsInstance(model.pcn_blocks[1], SparseMoEPCNBlock)

        tokens = torch.randint(0, 32, (2, 8))
        out = model(tokens, return_errors=True)
        self.assertEqual(out["logits"].shape, (2, 8, 32))
        self.assertEqual(len(out["errors"]), 2)
        self.assertGreaterEqual(out["load_balance_loss"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
