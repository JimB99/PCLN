"""Unit tests for src.model.pcn_block.

Tests cover:
- Initialization with different dimensions and seeds
- NumPy mode (default)
- PyTorch mode (if available)
- Refinement iterations
- Random state generation
"""

import unittest
import numpy as np

from src.model import PredictiveCodingBlock


class TestPCNBlockNumPy(unittest.TestCase):
    """Test PCN block in NumPy mode."""

    def test_init(self):
        """Test initialization."""
        block = PredictiveCodingBlock(dim=16, use_torch=False, seed=42)
        self.assertEqual(block.dim, 16)
        self.assertFalse(block.use_torch)

    def test_generative_shape(self):
        """Test generative function returns correct shape."""
        block = PredictiveCodingBlock(dim=8, use_torch=False, seed=42)
        z = np.random.randn(4, 8).astype(np.float32)
        z_hat = block.generative(z)
        self.assertEqual(z_hat.shape, z.shape)

    def test_refine_shape(self):
        """Test refinement loop returns correct shape."""
        block = PredictiveCodingBlock(dim=8, use_torch=False, seed=42)
        z = np.random.randn(4, 8).astype(np.float32)
        refined = block.refine(z, K=3, alpha=0.1)
        self.assertEqual(refined.shape, z.shape)

    def test_refine_reduces_error(self):
        """Test that refinement reduces prediction error."""
        block = PredictiveCodingBlock(dim=8, use_torch=False, seed=42)
        z = block.random_state(batch_size=2)
        
        # Initial error
        z_hat_init = block.generative(z)
        error_init = float(np.mean((z - z_hat_init) ** 2))
        
        # Refine a few steps
        refined = block.refine(z.copy(), K=5, alpha=0.1)
        z_hat_final = block.generative(refined)
        error_final = float(np.mean((refined - z_hat_final) ** 2))
        
        # Error should decrease or stay similar (not increase significantly)
        self.assertLessEqual(error_final, error_init + 1e-6)

    def test_random_state_shape(self):
        """Test random state generation."""
        block = PredictiveCodingBlock(dim=16, use_torch=False, seed=42)
        z = block.random_state(batch_size=8)
        self.assertEqual(z.shape, (8, 16))
        self.assertTrue(np.isfinite(z).all())

    def test_deterministic_with_seed(self):
        """Test reproducibility with fixed seed."""
        block1 = PredictiveCodingBlock(dim=8, use_torch=False, seed=123)
        z1 = block1.random_state(batch_size=2)
        
        block2 = PredictiveCodingBlock(dim=8, use_torch=False, seed=123)
        z2 = block2.random_state(batch_size=2)
        
        np.testing.assert_array_equal(z1, z2)


class TestPCNBlockTorch(unittest.TestCase):
    """Test PCN block in PyTorch mode (if available)."""

    def setUp(self):
        """Check torch availability."""
        try:
            import torch
            self.torch_available = True
        except ImportError:
            self.torch_available = False

    def test_torch_mode_init(self):
        """Test torch mode initialization."""
        if not self.torch_available:
            self.skipTest("torch not available")
        
        block = PredictiveCodingBlock(dim=16, use_torch=True, seed=42)
        self.assertEqual(block.dim, 16)
        # use_torch may be False if torch unavailable globally, but we set it
        # only in the init; the setter depends on _TORCH_AVAILABLE at import time

    def test_torch_generative_shape(self):
        """Test torch mode generative function."""
        if not self.torch_available:
            self.skipTest("torch not available")
        
        import torch
        block = PredictiveCodingBlock(dim=8, use_torch=True, seed=42)
        if not block.use_torch:
            self.skipTest("torch mode not supported (torch unavailable at import)")
        
        z = torch.randn(4, 8)
        z_hat = block.generative(z)
        self.assertEqual(z_hat.shape, z.shape)

    def test_torch_refine_shape(self):
        """Test torch mode refinement."""
        if not self.torch_available:
            self.skipTest("torch not available")
        
        import torch
        block = PredictiveCodingBlock(dim=8, use_torch=True, seed=42)
        if not block.use_torch:
            self.skipTest("torch mode not supported (torch unavailable at import)")
        
        z = torch.randn(4, 8)
        refined = block.refine(z, K=3, alpha=0.1)
        self.assertEqual(refined.shape, z.shape)


if __name__ == "__main__":
    unittest.main()
