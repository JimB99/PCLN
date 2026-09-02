"""Shared generation utilities and train/chat tokenizer alignment."""

from __future__ import annotations

import unittest

import torch

from src.data.text_codec import WordCodec, encode_text, word_tokenize
from src.model.generation import apply_top_p, sample_next_token


class TestWordCodec(unittest.TestCase):
    def test_tokenize_matches_str_split(self):
        text = "Hello, world! The king said."
        self.assertEqual(word_tokenize(text), text.split())

    def test_roundtrip_known_words(self):
        codec = WordCodec(
            word2id={"hello": 0, "world": 1, "<unk>": 2},
            id2word={0: "hello", 1: "world", 2: "<unk>"},
            unk_id=2,
        )
        ids = codec.encode("hello world")
        self.assertEqual(ids, [0, 1])
        self.assertEqual(codec.decode(ids), "hello world")

    def test_encode_text_char_mode(self):
        ids = encode_text("ab", tokenization="char", char2id={"a": 1, "b": 2})
        self.assertEqual(ids, [1, 2])


class TestNucleusSampling(unittest.TestCase):
    def test_top_p_keeps_at_least_one_token(self):
        logits = torch.tensor([[0.0, -100.0, -100.0]])
        filtered = apply_top_p(logits, top_p=0.1)
        self.assertTrue(torch.isfinite(filtered[0, 0]))
        self.assertFalse(torch.isfinite(filtered[0, 1]))

    def test_sample_next_token_shape(self):
        logits = torch.randn(2, 10)
        token = sample_next_token(logits, temperature=0.8, top_k=5, top_p=0.9)
        self.assertEqual(token.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
