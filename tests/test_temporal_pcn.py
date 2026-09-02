"""Temporal and hierarchical predictive-coding blocks."""

from __future__ import annotations

import unittest

import torch

from src.model.temporal_pcn import HierarchicalPCNBlock, TemporalPCNBlock, causal_shift


class TestCausalShift(unittest.TestCase):
    def test_shifts_time_right_and_fills_start(self):
        z = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
        start = torch.zeros(3)
        shifted = causal_shift(z, start)
        self.assertTrue(torch.allclose(shifted[:, 0], start))
        self.assertTrue(torch.allclose(shifted[:, 1], z[:, 0]))
        self.assertTrue(torch.allclose(shifted[:, 2], z[:, 1]))
        self.assertTrue(torch.allclose(shifted[:, 3], z[:, 2]))


class TestTemporalPCNBlock(unittest.TestCase):
    def test_output_and_error_shapes(self):
        block = TemporalPCNBlock(d_model=16, K=2, alpha=0.1)
        z = torch.randn(2, 7, 16)
        out, errors = block(z, training=True)
        self.assertEqual(out.shape, z.shape)
        self.assertEqual(errors.shape[0], 2)
        self.assertEqual(errors.shape[-1], 16)
        self.assertTrue(torch.isfinite(out).all())

    def test_position_zero_independent_of_future_latents(self):
        torch.manual_seed(0)
        block = TemporalPCNBlock(d_model=8, K=2, alpha=0.2)
        block.eval()
        z = torch.randn(1, 6, 8)
        z2 = z.clone()
        z2[:, -1] = torch.randn(8)
        with torch.no_grad():
            a, _ = block(z, training=False)
            b, _ = block(z2, training=False)
        self.assertTrue(torch.allclose(a[:, 0], b[:, 0], atol=1e-5, rtol=1e-5))

    def test_gradients_flow(self):
        block = TemporalPCNBlock(d_model=8, K=2, alpha=0.1)
        z = torch.randn(2, 5, 8, requires_grad=True)
        out, errors = block(z, training=True)
        loss = out.pow(2).mean() + errors.pow(2).mean()
        loss.backward()
        self.assertIsNotNone(z.grad)
        self.assertTrue(any(p.grad is not None for p in block.parameters() if p.requires_grad))

    def test_adaptive_k_stops_when_error_is_tiny(self):
        block = TemporalPCNBlock(
            d_model=8, K=6, alpha=0.1, adaptive_k=True, error_stop_threshold=1e6
        )
        block.eval()
        z = torch.randn(1, 4, 8)
        with torch.no_grad():
            _out, errors = block(z, training=False)
        # High threshold means first-step error is already "small" → stop before K.
        self.assertLess(errors.shape[1], 6)


class TestHierarchicalPCNBlock(unittest.TestCase):
    def test_output_shape(self):
        block = HierarchicalPCNBlock(d_model=12, timescale=4, K=2, alpha=0.1)
        z = torch.randn(2, 9, 12)
        out, errors = block(z, training=True)
        self.assertEqual(out.shape, z.shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(errors).all())

    def test_future_tokens_do_not_change_early_slow_state(self):
        torch.manual_seed(3)
        block = HierarchicalPCNBlock(d_model=8, timescale=3, K=1, alpha=0.1)
        block.eval()
        z = torch.randn(1, 8, 8)
        z2 = z.clone()
        z2[:, -1] = torch.randn(8)
        with torch.no_grad():
            a, _ = block(z, training=False)
            b, _ = block(z2, training=False)
        self.assertTrue(torch.allclose(a[:, 0], b[:, 0], atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
