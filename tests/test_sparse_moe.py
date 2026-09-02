"""Vectorized sparse MoE PCN block."""

from __future__ import annotations

import unittest

import torch

from src.model.sparse_moe import SparseMoEPCNBlock


class TestVectorizedSparseMoE(unittest.TestCase):
    def test_forward_shapes(self):
        block = SparseMoEPCNBlock(d_model=16, num_experts=4, top_k=2, K_refine=2)
        z = torch.randn(3, 11, 16)
        out, errors, lb, usage = block(z, training=True)
        self.assertEqual(out.shape, z.shape)
        self.assertEqual(errors.shape[0], 3)
        self.assertEqual(errors.shape[-1], 16)
        self.assertGreaterEqual(float(lb), 0.0)
        self.assertEqual(usage.numel(), 4)
        self.assertTrue(torch.isfinite(out).all())

    def test_backward_through_experts(self):
        block = SparseMoEPCNBlock(d_model=8, num_experts=3, top_k=2, K_refine=1)
        z = torch.randn(2, 5, 8, requires_grad=True)
        out, errors, lb, _ = block(z, training=True)
        (out.pow(2).mean() + errors.pow(2).mean() + lb).backward()
        self.assertIsNotNone(z.grad)
        self.assertTrue(any(p.grad is not None for p in block.experts.parameters()))

    def test_eval_does_not_require_training_flag_for_shapes(self):
        block = SparseMoEPCNBlock(d_model=8, num_experts=2, top_k=1, K_refine=2)
        block.eval()
        z = torch.randn(1, 4, 8)
        with torch.no_grad():
            out, errors, lb, _ = block(z, training=False)
        self.assertEqual(out.shape, z.shape)
        self.assertEqual(float(lb), 0.0)


if __name__ == "__main__":
    unittest.main()
