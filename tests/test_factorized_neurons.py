"""Factorized dynamic neurons: sparse routing with rank-1 units."""

from __future__ import annotations

import unittest

import torch

from src.model.dynamic_neurons import DynamicNeuronBlock, DynamicNeuronLayer


class TestFactorizedDynamicNeurons(unittest.TestCase):
    def test_output_shape(self):
        layer = DynamicNeuronLayer(in_features=16, out_features=24, num_neurons=32, top_k_neurons=4)
        x = torch.randn(2, 5, 16)
        out, usage = layer(x, return_neuron_usage=True)
        self.assertEqual(out.shape, (2, 5, 24))
        self.assertIn("load_balance_loss", usage)
        self.assertGreaterEqual(usage["load_balance_loss"].item(), 0.0)

    def test_parameter_count_is_factorized_not_dense_outer_product(self):
        in_f, out_f, n = 32, 64, 128
        layer = DynamicNeuronLayer(in_f, out_f, num_neurons=n, top_k_neurons=8)
        n_params = sum(p.numel() for p in layer.parameters())
        dense_outer = n * in_f * out_f
        self.assertLess(n_params, dense_outer // 4)

    def test_unused_neuron_projection_is_not_a_full_in_by_n_out_linear(self):
        layer = DynamicNeuronLayer(8, 8, num_neurons=16, top_k_neurons=4)
        self.assertFalse(hasattr(layer, "neurons") and isinstance(layer.neurons, torch.nn.Linear))

    def test_block_returns_three_values(self):
        block = DynamicNeuronBlock(d_model=16, num_neurons=24, top_k_neurons=4, K=2)
        z = torch.randn(2, 6, 16)
        out, errors, lb = block(z, training=True)
        self.assertEqual(out.shape, z.shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreaterEqual(float(lb), 0.0)

    def test_gradients_flow(self):
        layer = DynamicNeuronLayer(8, 8, num_neurons=16, top_k_neurons=4)
        x = torch.randn(2, 3, 8, requires_grad=True)
        out, usage = layer(x, return_neuron_usage=True)
        (out.sum() + usage["load_balance_loss"]).backward()
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
