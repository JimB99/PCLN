"""Unit tests for data loaders."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data import (
    CharLevelDataset,
    get_char_level_dataloader,
    get_wikitext2_char_dataloader,
    get_wikitext2_dataloader,
    resolve_wikitext2_text_file,
    vocab_mappings_from_dataset,
)


class TestCharLevelDataset(unittest.TestCase):
    def test_vocab_from_tiny_text(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("abc\n")
            path = f.name
        try:
            ds = CharLevelDataset(path, seq_len=2, max_samples=1)
            self.assertGreaterEqual(ds.vocab_size, 3)
            mappings = vocab_mappings_from_dataset(ds)
            self.assertEqual(mappings["tokenization"], "char")
            self.assertIn("a", mappings["char2id"])
        finally:
            Path(path).unlink(missing_ok=True)


class TestWikiTextFallback(unittest.TestCase):
    def test_resolve_falls_back_to_shakespeare(self):
        with patch("src.data.download_wikitext2_direct", return_value=None):
            with patch("src.data.download_tiny_shakespeare") as mock_shake:
                mock_shake.return_value = "data/tiny_shakespeare.txt"
                path = resolve_wikitext2_text_file(split="train")
                self.assertEqual(path, "data/tiny_shakespeare.txt")


class TestLoaderReturnShape(unittest.TestCase):
    def test_char_loader_returns_three_values(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("hello world " * 50)
            path = f.name
        try:
            loader, vocab_size, mappings = get_char_level_dataloader(
                file_path=path, seq_len=8, batch_size=4, max_samples=5
            )
            self.assertGreater(vocab_size, 0)
            self.assertEqual(mappings["tokenization"], "char")
            self.assertGreater(len(loader), 0)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_wikitext_word_loader_uses_local_shakespeare_if_present(self):
        shake = Path("data/tiny_shakespeare.txt")
        if not shake.exists():
            self.skipTest("tiny_shakespeare.txt not present")
        loader, vocab_size, mappings = get_wikitext2_dataloader(
            split="train", seq_len=8, batch_size=4, max_samples=5, vocab_size=128
        )
        self.assertGreater(vocab_size, 0)
        self.assertEqual(mappings["tokenization"], "word")


if __name__ == "__main__":
    unittest.main()
