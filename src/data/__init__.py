"""Data loading for WikiText-2, Tiny Shakespeare, and other datasets."""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

LoaderResult = Tuple[DataLoader, int, dict]


def vocab_mappings_from_dataset(dataset: Dataset) -> dict:
    """Build checkpoint-safe vocab mappings from a dataset instance."""
    if hasattr(dataset, "char2id"):
        return {
            "tokenization": "char",
            "char2id": dict(dataset.char2id),
            "id2char": {int(k): v for k, v in dataset.id2char.items()},
        }
    if hasattr(dataset, "word2id"):
        return {
            "tokenization": "word",
            "word2id": dict(dataset.word2id),
            "id2word": {int(k): v for k, v in dataset.id2word.items()},
        }
    return {"tokenization": "dummy"}


def _make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def _pack_loader_result(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int = 0,
) -> LoaderResult:
    loader = _make_dataloader(dataset, batch_size, shuffle, drop_last, num_workers)
    return loader, dataset.vocab_size, vocab_mappings_from_dataset(dataset)


class TextFileDataset(Dataset):
    """Generic text file dataset with word-level tokenization."""

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

        print(f"Loading text file: {file_path}...")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            if max_lines is not None:
                lines = lines[:max_lines]

        full_text = " ".join(lines)

        print("Building vocabulary...")
        words = full_text.split()
        word_counts = Counter(words)

        self.word2id = {w: i for i, (w, _) in enumerate(word_counts.most_common(vocab_size))}
        self.word2id["<unk>"] = vocab_size
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = vocab_size + 1

        print("Tokenizing text...")
        tokens = [self.word2id.get(w, vocab_size) for w in words]

        self.data = []
        for i in range(0, len(tokens) - self.seq_len, self.seq_len):
            chunk = tokens[i : i + self.seq_len + 1]
            if len(chunk) == self.seq_len + 1:
                self.data.append(chunk)

        if max_samples is not None:
            self.data = self.data[:max_samples]

        print(f"Created {len(self.data)} sequences of length {seq_len + 1}")
        print(f"Vocabulary size: {self.vocab_size}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        input_tokens = torch.tensor(chunk[:-1], dtype=torch.long)
        target_token = torch.tensor(chunk[-1], dtype=torch.long)
        return input_tokens, target_token


def download_tiny_shakespeare(output_path: str = "data/tiny_shakespeare.txt") -> str:
    """Download Tiny Shakespeare dataset."""
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


def download_wikitext2_direct(split: str = "train", output_dir: str = "data/wikitext2") -> str | None:
    """Download WikiText-2 directly from HuggingFace Hub (bypasses fsspec)."""
    import gzip
    import urllib.request

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": ("wikitext-2-v1/wikitext-2-train.txt", "wiki.train.tokens"),
        "validation": ("wikitext-2-v1/wikitext-2-valid.txt", "wiki.valid.tokens"),
        "test": ("wikitext-2-v1/wikitext-2-test.txt", "wiki.test.tokens"),
    }

    if split not in split_map:
        raise ValueError(f"Unknown split: {split}. Must be one of {list(split_map.keys())}")

    hf_rel, zip_member = split_map[split]
    file_path = output_path / f"{split}.txt"

    if file_path.exists():
        print(f"File already exists: {file_path}")
        return str(file_path)

    hf_urls = [
        f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/{hf_rel}.gz",
        f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/{hf_rel}",
        f"https://huggingface.co/datasets/wikitext/resolve/main/{hf_rel}.gz",
    ]
    s3_zip_url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"

    for url in hf_urls:
        temp_download = output_path / f"{split}.download"
        try:
            print(f"Downloading WikiText-2 ({split}) from {url}...")
            urllib.request.urlretrieve(url, temp_download)
            if url.endswith(".gz"):
                print(f"Decompressing to {file_path}...")
                with gzip.open(temp_download, "rb") as f_in:
                    with open(file_path, "wb") as f_out:
                        f_out.write(f_in.read())
            else:
                temp_download.replace(file_path)
            temp_download.unlink(missing_ok=True)
            print(f"WikiText-2 {split} ready at: {file_path}")
            return str(file_path)
        except Exception as e:
            print(f"  Failed: {e}")
            temp_download.unlink(missing_ok=True)

    temp_zip = output_path / "wikitext-2-v1.zip"
    try:
        print("Downloading WikiText-2 zip from S3...")
        urllib.request.urlretrieve(s3_zip_url, temp_zip)
        import zipfile

        with zipfile.ZipFile(temp_zip, "r") as zf:
            member = f"wikitext-2/{zip_member}"
            with zf.open(member) as src, open(file_path, "wb") as dst:
                dst.write(src.read())
        temp_zip.unlink(missing_ok=True)
        print(f"WikiText-2 {split} ready at: {file_path}")
        return str(file_path)
    except Exception as e:
        print(f"Failed to download WikiText-2 from S3 zip: {e}")
        temp_zip.unlink(missing_ok=True)
        return None


def resolve_wikitext2_text_file(split: str = "train", output_dir: str = "data/wikitext2") -> str:
    """Return a local text file path for WikiText-2 or Tiny Shakespeare fallback."""
    file_path = download_wikitext2_direct(split=split, output_dir=output_dir)
    if file_path and Path(file_path).exists():
        return file_path
    print("WikiText-2 unavailable, falling back to Tiny Shakespeare...")
    return download_tiny_shakespeare()


class WikiText2Dataset(Dataset):
    """WikiText-2 dataset with multiple fallback strategies."""

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
        output_dir = cache_dir or "data/wikitext2"

        print(f"Loading WikiText-2 ({split})...")
        dataset = None

        try:
            print("  Attempting: Direct download from HuggingFace...")
            wikitext_path = download_wikitext2_direct(split=split, output_dir=output_dir)
            if wikitext_path and Path(wikitext_path).exists():
                print("  Downloaded direct")
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
            print(f"  Failed: {type(e).__name__}: {str(e)[:60]}...")

        if dataset is None:
            try:
                from datasets import load_dataset

                print("  Attempting: HuggingFace datasets library...")
                dataset = load_dataset("wikitext", "wikitext-2", split=split, trust_remote_code=True)
                print("  Loaded via HuggingFace datasets")
            except Exception as e:
                print(f"  Failed: {type(e).__name__}: {str(e)[:60]}...")

        if dataset is None:
            print("  Attempting: Tiny Shakespeare fallback...")
            shakespeare_path = download_tiny_shakespeare()
            dataset_obj = TextFileDataset(
                file_path=shakespeare_path,
                seq_len=seq_len,
                vocab_size=vocab_size,
                max_samples=max_samples,
            )
            self.data = dataset_obj.data
            self.word2id = dataset_obj.word2id
            self.id2word = dataset_obj.id2word
            self.vocab_size = dataset_obj.vocab_size
            return

        texts = dataset["text"]
        full_text = " ".join(texts)

        print("Building vocabulary...")
        words = full_text.split()
        word_counts = Counter(words)

        self.word2id = {w: i for i, (w, _) in enumerate(word_counts.most_common(vocab_size))}
        self.word2id["<unk>"] = vocab_size
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = vocab_size + 1

        print("Tokenizing text...")
        tokens = [self.word2id.get(w, vocab_size) for w in words]

        self.data = []
        for i in range(0, len(tokens) - self.seq_len, self.seq_len):
            chunk = tokens[i : i + self.seq_len + 1]
            if len(chunk) == self.seq_len + 1:
                self.data.append(chunk)

        if max_samples is not None:
            self.data = self.data[:max_samples]

        print(f"Created {len(self.data)} sequences of length {seq_len + 1}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        input_tokens = torch.tensor(chunk[:-1], dtype=torch.long)
        target_token = torch.tensor(chunk[-1], dtype=torch.long)
        return input_tokens, target_token


class CharLevelDataset(Dataset):
    """Character-level tokenization dataset."""

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
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        if max_chars is not None:
            text = text[:max_chars]

        print("Building character vocabulary...")
        unique_chars = sorted(set(text))
        self.char2id = {ch: i for i, ch in enumerate(unique_chars)}
        self.id2char = {v: k for k, v in self.char2id.items()}
        self.vocab_size = len(unique_chars)

        print(f"Character vocabulary size: {self.vocab_size}")

        print("Tokenizing text...")
        tokens = [self.char2id[ch] for ch in text]

        self.data = []
        for i in range(0, len(tokens) - self.seq_len, self.seq_len):
            chunk = tokens[i : i + self.seq_len + 1]
            if len(chunk) == self.seq_len + 1:
                self.data.append(chunk)

        if max_samples is not None:
            self.data = self.data[:max_samples]

        print(f"Created {len(self.data)} sequences of length {self.seq_len + 1}")

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
) -> LoaderResult:
    """Word-level WikiText-2 dataloader with Tiny Shakespeare fallback."""
    file_path = resolve_wikitext2_text_file(split=split)
    dataset = TextFileDataset(
        file_path=file_path,
        seq_len=seq_len,
        vocab_size=vocab_size,
        max_samples=max_samples,
    )
    return _pack_loader_result(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and split == "train",
        drop_last=split == "train",
        num_workers=num_workers,
    )


def get_wikitext2_char_dataloader(
    split: str = "train",
    seq_len: int = 64,
    batch_size: int = 16,
    max_samples: int | None = None,
    num_workers: int = 0,
    shuffle: bool = True,
) -> LoaderResult:
    """Character-level WikiText-2 dataloader with Tiny Shakespeare fallback."""
    file_path = resolve_wikitext2_text_file(split=split)
    dataset = CharLevelDataset(
        file_path=file_path,
        seq_len=seq_len,
        max_samples=max_samples,
    )
    return _pack_loader_result(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and split == "train",
        drop_last=split == "train",
        num_workers=num_workers,
    )


def get_text_file_dataloader(
    file_path: str,
    seq_len: int = 64,
    batch_size: int = 16,
    vocab_size: int = 1024,
    max_samples: int | None = None,
    shuffle: bool = True,
) -> LoaderResult:
    """Word-level dataloader from a custom text file."""
    dataset = TextFileDataset(
        file_path=file_path,
        seq_len=seq_len,
        vocab_size=vocab_size,
        max_samples=max_samples,
    )
    return _pack_loader_result(dataset, batch_size, shuffle, drop_last=True)


def get_char_level_dataloader(
    file_path: str,
    seq_len: int = 64,
    batch_size: int = 16,
    max_chars: int | None = None,
    max_samples: int | None = None,
    num_workers: int = 0,
    shuffle: bool = True,
) -> LoaderResult:
    """Character-level dataloader from a custom text file."""
    dataset = CharLevelDataset(
        file_path=file_path,
        seq_len=seq_len,
        max_chars=max_chars,
        max_samples=max_samples,
    )
    return _pack_loader_result(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
    )


class DummyDataset(Dataset):
    """Dummy random dataset for quick testing."""

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
) -> LoaderResult:
    """Dummy dataloader for quick testing."""
    dataset = DummyDataset(num_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size)
    return _pack_loader_result(dataset, batch_size, shuffle, drop_last=True)
