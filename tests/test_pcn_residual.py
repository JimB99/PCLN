"""PCN residual must not double-count the input state."""

from __future__ import annotations

import unittest

import torch

from src.model.pcn_layers import PCNBlock, PCNRefinementLayer


class TestPCNResidual(unittest.TestCase):
    def test_block_output_is_normalized_refined_state_not_z_plus_z(self):
        torch.manual_seed(0)
        block = PCNBlock(d_model=8, K=1, alpha=0.0)
        # alpha=0 → refined == input (generator unused for the update).
        z = torch.randn(2, 4, 8)
        out, _ = block(z, training=True)
        # With zero step size, output should be LayerNorm(z), not LayerNorm(2z).
        expected = torch.nn.functional.layer_norm(z, (8,))
        self.assertTrue(torch.allclose(out, expected, atol=1e-5, rtol=1e-5))

    def test_refinement_reports_errors(self):
        layer = PCNRefinementLayer(d_model=8, K=2, alpha=0.1)
        z = torch.randn(1, 3, 8)
        refined, errors = layer(z, training=True)
        self.assertEqual(refined.shape, z.shape)
        self.assertEqual(errors.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
