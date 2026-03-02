"""Data loading for WikiText-2, Tiny Shakespeare, and other datasets."""

from __future__ import annotations

import os
from pathlib import Path
from collections import Counter
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class TextFileDataset(Dataset):
    """Generic text file dataset with tokenization and fixed sequence length.
    
    Args:
        file_path: path to text file.
        seq_len: sequence length (context window).
        vocab_size: vocabulary size for encoding.
        max_lines: max lines to read (None = all).
        max_samples: max samples to use (None = all).
    """

    def __init__(
        self,
        file_path: str,
        seq_len: int = 64,
        vocab_size: int = 1024,
        max_lines: int | None = None,
        max_samples: int | None = None,
    ):
        self.file_path = file_path
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        print(f"Loading text file: {file_path}...")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            if max_lines is not None:
                lines = lines[:max_lines]

        # Concatenate all text
        full_text = " ".join(lines)

        # Build vocab from text (simple word-level)
        print("Building vocabulary...")
        words = full_text.split()
        word_counts = Counter(words)

        # Keep top vocab_size words
        self.word2id = {w: i for i, (w, _) in enumerate(word_counts.most_common(vocab_size))}
        # Use vocab_size as <unk> token
        self.word2id["<unk>"] = vocab_size
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = vocab_size + 1

        # Tokenize
        print("Tokenizing text...")
        tokens = [self.word2id.get(w, vocab_size) for w in words]

        # Create chunks of seq_len
        self.data = []
        for i in range(0, len(tokens) - self.seq_len, self.seq_len):
            chunk = tokens[i : i + self.seq_len + 1]  # +1 for target
            if len(chunk) == self.seq_len + 1:
                self.data.append(chunk)

        if max_samples is not None:
            self.data = self.data[:max_samples]

        print(f"Created {len(self.data)} sequences of length {seq_len+1}")
        print(f"Vocabulary size: {self.vocab_size}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        input_tokens = torch.tensor(chunk[:-1], dtype=torch.long)
        target_token = torch.tensor(chunk[-1], dtype=torch.long)
        return input_tokens, target_token


def download_tiny_shakespeare(output_path: str = "data/tiny_shakespeare.txt"):
    """Download Tiny Shakespeare dataset.
    
    Args:
        output_path: where to save the file.
    """
    import urllib.request
    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"File already exists: {output_path}")
        return str(output_path)
    
    print(f"Downloading Tiny Shakespeare to {output_path}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded {output_path.stat().st_size / 1024:.1f} KB")
    return str(output_path)


def download_wikitext2_direct(split: str = "train", output_dir: str = "data/wikitext2"):
    """Download WikiText-2 directly from HuggingFace Hub (bypasses fsspec).
    
    This avoids Windows fsspec glob pattern issues by downloading files directly.
    
    Args:
        split: 'train', 'validation', or 'test'.
        output_dir: directory to cache downloaded files.
        
    Returns:
        path to the text file.
    """
    import urllib.request
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Map split names to file names on HuggingFace
    split_map = {
        "train": "wikitext-2-v1/wikitext-2-train.txt.gz",
        "validation": "wikitext-2-v1/wikitext-2-valid.txt.gz",
        "test": "wikitext-2-v1/wikitext-2-test.txt.gz",
    }
    
    if split not in split_map:
        raise ValueError(f"Unknown split: {split}. Must be one of {list(split_map.keys())}")
    
    file_path = output_path / f"{split}.txt"
    
    if file_path.exists():
        print(f"File already exists: {file_path}")
        return str(file_path)
    
    # Download from HuggingFace CDN
    url = f"https://huggingface.co/datasets/wikitext/resolve/main/{split_map[split]}"
    temp_gz = output_path / f"{split}.txt.gz"
    
    try:
        print(f"Downloading WikiText-2 ({split}) from HuggingFace...")
        urllib.request.urlretrieve(url, temp_gz)
        print(f"Downloaded {temp_gz.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Decompress
        import gzip
        print(f"Decompressing to {file_path}...")
        with gzip.open(temp_gz, 'rb') as f_in:
            with open(file_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Clean up gz
        temp_gz.unlink()
        
        print(f"WikiText-2 {split} ready at: {file_path}")
        return str(file_path)
        
    except Exception as e:
        print(f"Failed to download WikiText-2 directly: {e}")
        # Clean up partial downloads
        if temp_gz.exists():
            temp_gz.unlink()
        return None



    """WikiText-2 dataset with multiple fallback strategies.
    
    Args:
        split: 'train', 'validation', or 'test'.
        seq_len: sequence length (context window).
        vocab_size: vocabulary size for encoding.
        max_samples: maximum number of samples to use.
    """

    def __init__(
        self,
        split: str = "train",
        seq_len: int = 64,
        vocab_size: int = 1024,
        cache_dir: str | None = None,
        max_samples: int | None = None,
    ):
        self.split = split
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Try multiple loading strategies
        print(f"Loading WikiText-2 ({split})...")
        dataset = None
        
        # Strategy 1: Try direct download (bypasses fsspec, works on Windows)
        try:
            print("  Attempting: Direct download from HuggingFace...")
            wikitext_path = download_wikitext2_direct(split=split)
            if wikitext_path and Path(wikitext_path).exists():
                print(f"  ✓ Downloaded direct")
                dataset_obj = TextFileDataset(
                    file_path=wikitext_path,
                    seq_len=seq_len,
                    vocab_size=vocab_size,
                    max_samples=max_samples,
                )
                self.data = dataset_obj.data
                self.word2id = dataset_obj.word2id
                self.id2word = dataset_obj.id2word
                self.vocab_size = dataset_obj.vocab_size
                return
        except Exception as e:
            print(f"  ✗ Failed: {type(e).__name__}: {str(e)[:60]}...")

        # Strategy 2: Try with HF datasets (may fail on Windows)
        if dataset is None:
            try:
                from datasets import load_dataset
                print("  Attempting: HuggingFace datasets library...")
                dataset = load_dataset("wikitext", "wikitext-2", split=split, trust_remote_code=True)
                print("  ✓ Loaded via HuggingFace datasets")
            except Exception as e:
                print(f"  ✗ Failed: {type(e).__name__}: {str(e)[:60]}...")

        # Strategy 3: Fallback to Tiny Shakespeare
        if dataset is None:
            print("  Attempting: Tiny Shakespeare fallback...")
            try:
                shakespeare_path = download_tiny_shakespeare()
                print(f"  Using Tiny Shakespeare instead...")
                
                dataset_obj = TextFileDataset(
                    file_path=shakespeare_path,
                    seq_len=seq_len,
                    vocab_size=vocab_size,
                    max_lines=None,
                    max_samples=max_samples,
                )
                # Store the object to delegate to it
                self.data = dataset_obj.data
                self.word2id = dataset_obj.word2id
                self.id2word = dataset_obj.id2word
                self.vocab_size = dataset_obj.vocab_size
                return
            except Exception as e:
                print(f"  ✗ Tiny Shakespeare failed: {e}")

        # If we got here and have a dataset, process it
        if dataset is None:
            raise RuntimeError("All dataset loading strategies failed!")

        # Process WikiText-2
        texts = dataset["text"]
        full_text = " ".join(texts)

        # Build vocab
        print("Building vocabulary...")
        words = full_text.split()
        word_counts = Counter(words)

        # Keep top vocab_size words
        self.word2id = {w: i for i, (w, _) in enumerate(word_counts.most_common(vocab_size))}
        self.word2id["<unk>"] = vocab_size
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = vocab_size + 1

        # Tokenize
        print("Tokenizing text...")
        tokens = [self.word2id.get(w, vocab_size) for w in words]

        # Create chunks
        self.data = []
        for i in range(0, len(tokens) - self.seq_len, self.seq_len):
            chunk = tokens[i : i + self.seq_len + 1]
            if len(chunk) == self.seq_len + 1:
                self.data.append(chunk)

        if max_samples is not None:
            self.data = self.data[:max_samples]

        print(f"Created {len(self.data)} sequences of length {seq_len+1}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        input_tokens = torch.tensor(chunk[:-1], dtype=torch.long)
        target_token = torch.tensor(chunk[-1], dtype=torch.long)
        return input_tokens, target_token


class CharLevelDataset(Dataset):
    """Character-level tokenization dataset.
    
    Uses character-level tokenization instead of word-level.
    - Vocabulary size: ~100 (ASCII letters, digits, punctuation)
    - No <unk> tokens (every character is representable)
    - Better for generation of arbitrary text patterns
    
    Args:
        file_path: path to text file.
        seq_len: sequence length (context window).
        max_chars: if set, limit to first N characters (for testing).
        max_samples: max samples to use.
    """

    def __init__(
        self,
        file_path: str,
        seq_len: int = 64,
        max_chars: int | None = None,
        max_samples: int | None = None,
    ):
        self.file_path = file_path
        self.seq_len = seq_len

        print(f"Loading text file (char-level): {file_path}...")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        if max_chars is not None:
            text = text[:max_chars]

        # Build character vocabulary
        print("Building character vocabulary...")
        unique_chars = sorted(set(text))
        self.char2id = {ch: i for i, ch in enumerate(unique_chars)}
        self.id2char = {v: k for k, v in self.char2id.items()}
        self.vocab_size = len(unique_chars)

        print(f"Character vocabulary size: {self.vocab_size}")
        print(f"Example chars: {unique_chars[:20]}")

        # Convert text to token IDs
        print("Tokenizing text...")
        tokens = [self.char2id[ch] for ch in text]

        # Create sequences
        self.data = []
        for i in range(0, len(tokens) - self.seq_len, self.seq_len):
            chunk = tokens[i : i + self.seq_len + 1]
            if len(chunk) == self.seq_len + 1:
                self.data.append(chunk)

        if max_samples is not None:
            self.data = self.data[:max_samples]

        print(f"Created {len(self.data)} sequences of length {self.seq_len + 1}")
        print(f"Total text length: {len(text)} characters, {len(tokens)} tokens")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        input_tokens = torch.tensor(chunk[:-1], dtype=torch.long)
        target_token = torch.tensor(chunk[-1], dtype=torch.long)
        return input_tokens, target_token


def get_wikitext2_dataloader(
    split: str = "train",
    seq_len: int = 64,
    batch_size: int = 16,
    vocab_size: int = 1024,
    max_samples: int | None = None,
    num_workers: int = 0,
    shuffle: bool = True,
) -> Tuple[DataLoader, int]:
    """Get WikiText-2 dataloader (with automatic Tiny Shakespeare fallback).
    
    Args:
        split: 'train', 'validation', or 'test'.
        seq_len: sequence length.
        batch_size: batch size.
        vocab_size: vocabulary size.
        max_samples: max samples to use.
        num_workers: number of data loading workers.
        shuffle: whether to shuffle data.
    Returns:
        DataLoader and vocab_size.
    """
    try:
        # Try to get WikiText-2 file
        file_path = download_wikitext2_direct(split=split, output_dir="data/wikitext2")
        dataset = TextFileDataset(
            file_path=file_path,
            seq_len=seq_len,
            vocab_size=vocab_size,
            max_samples=max_samples,
        )
    except Exception as e:
        # Fallback to Tiny Shakespeare
        print(f"Failed to load WikiText-2: {e}")
        print("Falling back to Tiny Shakespeare dataset...")
        file_path = download_tiny_shakespeare()
        dataset = TextFileDataset(
            file_path=file_path,
            seq_len=seq_len,
            vocab_size=vocab_size,
            max_samples=max_samples,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and split == "train",
        num_workers=num_workers,
        drop_last=split == "train",
    )

    return dataloader, dataset.vocab_size


def get_text_file_dataloader(
    file_path: str,
    seq_len: int = 64,
    batch_size: int = 16,
    vocab_size: int = 1024,
    max_samples: int | None = None,
    shuffle: bool = True,
) -> Tuple[DataLoader, int]:
    """Get dataloader from a custom text file.
    
    Args:
        file_path: path to text file.
        seq_len: sequence length.
        batch_size: batch size.
        vocab_size: vocabulary size.
        max_samples: max samples to use.
        shuffle: whether to shuffle data.
    Returns:
        DataLoader and vocab_size.
    """
    dataset = TextFileDataset(
        file_path=file_path,
        seq_len=seq_len,
        vocab_size=vocab_size,
        max_samples=max_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )

    return dataloader, dataset.vocab_size


def get_char_level_dataloader(
    file_path: str,
    seq_len: int = 64,
    batch_size: int = 16,
    max_chars: int | None = None,
    max_samples: int | None = None,
    num_workers: int = 0,
    shuffle: bool = True,
) -> Tuple[DataLoader, int]:
    """Get character-level tokenized dataloader.
    
    Character-level tokenization provides:
    - Small vocabulary (~100 chars)
    - No <unk> tokens needed
    - Better generation of arbitrary text patterns
    
    Args:
        file_path: path to text file.
        seq_len: sequence length.
        batch_size: batch size.
        max_chars: max characters to use (for testing).
        max_samples: max samples to use.
        num_workers: number of data loading workers.
        shuffle: whether to shuffle data.
    Returns:
        DataLoader and vocab_size (number of unique characters).
    """
    dataset = CharLevelDataset(
        file_path=file_path,
        seq_len=seq_len,
        max_chars=max_chars,
        max_samples=max_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )

    return dataloader, dataset.vocab_size


class DummyDataset(Dataset):
    """Dummy random dataset for quick testing.
    
    Args:
        num_samples: number of random samples.
        seq_len: sequence length.
        vocab_size: vocabulary size.
    """

    def __init__(self, num_samples: int = 100, seq_len: int = 64, vocab_size: int = 256):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_tokens = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        target_token = torch.randint(0, self.vocab_size, (1,), dtype=torch.long).squeeze(0)
        return input_tokens, target_token


def get_dummy_dataloader(
    num_samples: int = 100,
    seq_len: int = 64,
    batch_size: int = 16,
    vocab_size: int = 256,
    shuffle: bool = True,
) -> Tuple[DataLoader, int]:
    """Get a dummy dataloader for quick testing.
    
    Returns:
        (DataLoader, vocab_size) tuple.
    """
    dataset = DummyDataset(num_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloader, vocab_size
