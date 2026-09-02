"""Episodic/semantic memory behavior."""

from __future__ import annotations

import unittest

import torch

from src.model import PCLN
from src.model.memory import EpisodicMemory, MemoryModule, SemanticMemory


class TestEpisodicMemory(unittest.TestCase):
    def test_retrieve_returns_stored_values_not_keys(self):
        mem = EpisodicMemory(d_model=4, max_memory_size=8)
        key = torch.tensor([1.0, 0.0, 0.0, 0.0])
        value = torch.tensor([0.0, 1.0, 0.0, 0.0])
        mem.store(key, value)
        got = mem.retrieve(key, topk=1)
        self.assertEqual(got.shape[-1], 4)
        retrieved = got.reshape(-1, 4)[0]
        self.assertTrue(torch.allclose(retrieved, value, atol=1e-5))
        self.assertFalse(torch.allclose(retrieved, key, atol=1e-5))

    def test_empty_memory_returns_zeros(self):
        mem = EpisodicMemory(d_model=4, max_memory_size=8)
        query = torch.randn(2, 4)
        got = mem.retrieve(query, topk=1)
        self.assertEqual(got.shape[0], 2)
        self.assertTrue(torch.allclose(got, torch.zeros_like(got)))

    def test_store_after_retrieve_does_not_include_current_write(self):
        mem = EpisodicMemory(d_model=4, max_memory_size=8)
        query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        empty = mem.retrieve(query, topk=1)
        self.assertTrue(torch.allclose(empty, torch.zeros_like(empty)))
        mem.store(query[0], torch.tensor([0.0, 0.0, 1.0, 0.0]))
        filled = mem.retrieve(query, topk=1)
        self.assertFalse(torch.allclose(filled, empty))


class TestMemoryModule(unittest.TestCase):
    def test_per_token_retrieve_shape(self):
        module = MemoryModule(d_model=8, episodic_size=16, semantic_slots=4)
        latent = torch.randn(2, 5, 8)
        retrieved = module(latent, store=False)
        self.assertEqual(retrieved.shape[:2], (2, 5))
        self.assertEqual(retrieved.shape[-1], 8)

    def test_surprise_gate_skips_low_surprise_store(self):
        module = MemoryModule(d_model=4, episodic_size=8, semantic_slots=2)
        latent = torch.randn(1, 3, 4)
        ptr_before = int(module.episodic.memory_ptr.item())
        module(latent, store=True, surprise=torch.tensor(0.0), surprise_threshold=1.0)
        self.assertEqual(int(module.episodic.memory_ptr.item()), ptr_before)

    def test_surprise_gate_stores_high_surprise(self):
        module = MemoryModule(d_model=4, episodic_size=8, semantic_slots=2)
        latent = torch.randn(1, 3, 4)
        ptr_before = int(module.episodic.memory_ptr.item())
        module(latent, store=True, surprise=torch.tensor(2.0), surprise_threshold=1.0)
        self.assertNotEqual(int(module.episodic.memory_ptr.item()), ptr_before)


class TestPCLNMemoryDuringTraining(unittest.TestCase):
    def test_training_forward_does_not_write_episodic_memory(self):
        model = PCLN(
            vocab_size=16,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_pcn_blocks=1,
            K_pcn=1,
            use_memory=True,
            episodic_memory_size=8,
            semantic_slots=4,
            causal=True,
        )
        model.train()
        ptr_before = int(model.memory.episodic.memory_ptr.item())
        tokens = torch.randint(0, 16, (2, 6))
        model(tokens, return_errors=False)
        self.assertEqual(int(model.memory.episodic.memory_ptr.item()), ptr_before)


class TestSemanticMemory(unittest.TestCase):
    def test_retrieve_shape(self):
        sem = SemanticMemory(num_slots=8, d_model=16)
        query = torch.randn(3, 16)
        got = sem.retrieve(query)
        self.assertEqual(got.shape, (3, 16))


if __name__ == "__main__":
    unittest.main()
