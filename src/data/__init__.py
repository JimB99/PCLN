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


def _build_sequence_chunks(
    tokens: list[int],
    seq_len: int,
    stride: int | None = None,
) -> list[list[int]]:
    """Split token stream into (seq_len+1) chunks with optional overlap."""
    step = stride if stride is not None and stride > 0 else seq_len
    chunks: list[list[int]] = []
    for i in range(0, len(tokens) - seq_len, step):
        chunk = tokens[i : i + seq_len + 1]
        if len(chunk) == seq_len + 1:
            chunks.append(chunk)
    return chunks


class TextFileDataset(Dataset):
    """Generic text file dataset with word-level tokenization."""

    def __init__(
        self,
        file_path: str,
        seq_len: int = 64,
        vocab_size: int = 1024,
        max_lines: int | None = None,
        max_samples: int | None = None,
        word2id: dict[str, int] | None = None,
        chunk_stride: int | None = None,
    ):
        self.file_path = file_path
        self.seq_len = seq_len
        self.chunk_stride = chunk_stride

        print(f"Loading text file: {file_path}...")
        read_chars = None
        if max_samples is not None:
            # Enough text for target sequences plus vocabulary estimation.
            read_chars = max_samples * (seq_len + 1) * 12

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            if read_chars is not None:
                full_text = f.read(read_chars)
            else:
                lines = f.readlines()
                if max_lines is not None:
                    lines = lines[:max_lines]
                full_text = " ".join(lines)

        print("Building vocabulary...")
        if word2id is not None:
            self.word2id = dict(word2id)
            self.id2word = {v: k for k, v in self.word2id.items()}
            self.vocab_size = max(self.word2id.values()) + 1
            unk_id = self.word2id.get("<unk>", self.vocab_size - 1)
        else:
            words = full_text.split()
            word_counts = Counter(words)
            self.word2id = {w: i for i, (w, _) in enumerate(word_counts.most_common(vocab_size))}
            self.word2id["<unk>"] = vocab_size
            self.id2word = {v: k for k, v in self.word2id.items()}
            self.vocab_size = vocab_size + 1
            unk_id = vocab_size

        print("Tokenizing text...")
        words = full_text.split()
        tokens = [self.word2id.get(w, unk_id) for w in words]

        self.data = _build_sequence_chunks(tokens, self.seq_len, self.chunk_stride)

        if max_samples is not None:
            self.data = self.data[:max_samples]

        stride_note = f", stride={self.chunk_stride or self.seq_len}"
        print(f"Created {len(self.data)} sequences of length {seq_len + 1}{stride_note}")
        print(f"Vocabulary size: {self.vocab_size}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        input_tokens = torch.tensor(chunk[:-1], dtype=torch.long)
        target_tokens = torch.tensor(chunk[1:], dtype=torch.long)
        return input_tokens, target_tokens


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


def _url_download(url: str, dest: Path) -> None:
    """Download a URL to a local file, following redirects."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "PCLN/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        out.write(resp.read())


def _parquet_to_text(parquet_path: Path, text_path: Path) -> None:
    """Convert a HuggingFace WikiText parquet shard to a plain-text file."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    with open(text_path, "w", encoding="utf-8") as out:
        for line in table.column("text").to_pylist():
            out.write(f"{line}\n")


def download_wikitext2_direct(split: str = "train", output_dir: str = "data/wikitext2") -> str | None:
    """Download WikiText-2 directly from HuggingFace Hub (bypasses fsspec)."""
    import gzip
    import zipfile

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": {
            "parquet": "wikitext-2-v1/train-00000-of-00001.parquet",
            "txt": "wikitext-2-v1/wikitext-2-train.txt",
            "zip_member": "wiki.train.tokens",
        },
        "validation": {
            "parquet": "wikitext-2-v1/validation-00000-of-00001.parquet",
            "txt": "wikitext-2-v1/wikitext-2-valid.txt",
            "zip_member": "wiki.valid.tokens",
        },
        "test": {
            "parquet": "wikitext-2-v1/test-00000-of-00001.parquet",
            "txt": "wikitext-2-v1/wikitext-2-test.txt",
            "zip_member": "wiki.test.tokens",
        },
    }

    if split not in split_map:
        raise ValueError(f"Unknown split: {split}. Must be one of {list(split_map.keys())}")

    meta = split_map[split]
    file_path = output_path / f"{split}.txt"

    if file_path.exists():
        print(f"File already exists: {file_path}")
        return str(file_path)

    hf_base = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main"
    temp_download = output_path / f"{split}.download"

    parquet_url = f"{hf_base}/{meta['parquet']}"
    try:
        print(f"Downloading WikiText-2 ({split}) parquet from HuggingFace...")
        _url_download(parquet_url, temp_download)
        print(f"Converting parquet to {file_path}...")
        _parquet_to_text(temp_download, file_path)
        temp_download.unlink(missing_ok=True)
        print(f"WikiText-2 {split} ready at: {file_path}")
        return str(file_path)
    except Exception as e:
        print(f"  Parquet download failed: {e}")
        temp_download.unlink(missing_ok=True)

    hf_rel = meta["txt"]
    hf_urls = [
        f"{hf_base}/{hf_rel}.gz",
        f"{hf_base}/{hf_rel}",
        f"https://huggingface.co/datasets/wikitext/resolve/main/{hf_rel}.gz",
    ]
    s3_zip_url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"

    for url in hf_urls:
        try:
            print(f"Downloading WikiText-2 ({split}) from {url}...")
            _url_download(url, temp_download)
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
        _url_download(s3_zip_url, temp_zip)
        with zipfile.ZipFile(temp_zip, "r") as zf:
            member = f"wikitext-2/{meta['zip_member']}"
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
        word2id: dict[str, int] | None = None,
        chunk_stride: int | None = None,
    ):
        self.split = split
        self.seq_len = seq_len
        self.chunk_stride = chunk_stride
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
                    chunk_stride=self.chunk_stride,
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
                chunk_stride=self.chunk_stride,
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

        self.data = _build_sequence_chunks(tokens, self.seq_len, self.chunk_stride)

        if max_samples is not None:
            self.data = self.data[:max_samples]

        stride_note = f", stride={self.chunk_stride or self.seq_len}"
        print(f"Created {len(self.data)} sequences of length {seq_len + 1}{stride_note}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        input_tokens = torch.tensor(chunk[:-1], dtype=torch.long)
        target_tokens = torch.tensor(chunk[1:], dtype=torch.long)
        return input_tokens, target_tokens


class CharLevelDataset(Dataset):
    """Character-level tokenization dataset."""

    def __init__(
        self,
        file_path: str,
        seq_len: int = 64,
        max_chars: int | None = None,
        max_samples: int | None = None,
        char2id: dict[str, int] | None = None,
        chunk_stride: int | None = None,
    ):
        self.file_path = file_path
        self.seq_len = seq_len
        self.chunk_stride = chunk_stride

        print(f"Loading text file (char-level): {file_path}...")
        if max_chars is None and max_samples is not None:
            max_chars = max_samples * (seq_len + 1) + 5000

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(max_chars) if max_chars is not None else f.read()

        print("Building character vocabulary...")
        if char2id is not None:
            self.char2id = dict(char2id)
            self.id2char = {v: k for k, v in self.char2id.items()}
            self.vocab_size = len(self.char2id)
        else:
            unique_chars = sorted(set(text))
            self.char2id = {ch: i for i, ch in enumerate(unique_chars)}
            self.id2char = {v: k for k, v in self.char2id.items()}
            self.vocab_size = len(unique_chars)

        print(f"Character vocabulary size: {self.vocab_size}")

        print("Tokenizing text...")
        tokens = [self.char2id[ch] for ch in text]

        self.data = _build_sequence_chunks(tokens, self.seq_len, self.chunk_stride)

        if max_samples is not None:
            self.data = self.data[:max_samples]

        stride_note = f", stride={self.chunk_stride or self.seq_len}"
        print(f"Created {len(self.data)} sequences of length {self.seq_len + 1}{stride_note}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        input_tokens = torch.tensor(chunk[:-1], dtype=torch.long)
        target_tokens = torch.tensor(chunk[1:], dtype=torch.long)
        return input_tokens, target_tokens


def get_wikitext2_dataloader(
    split: str = "train",
    seq_len: int = 64,
    batch_size: int = 16,
    vocab_size: int = 1024,
    max_samples: int | None = None,
    num_workers: int = 0,
    shuffle: bool = True,
    word2id: dict[str, int] | None = None,
    chunk_stride: int | None = None,
) -> LoaderResult:
    """Word-level WikiText-2 dataloader with Tiny Shakespeare fallback."""
    file_path = resolve_wikitext2_text_file(split=split)
    dataset = TextFileDataset(
        file_path=file_path,
        seq_len=seq_len,
        vocab_size=vocab_size,
        max_samples=max_samples,
        word2id=word2id,
        chunk_stride=chunk_stride,
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
    char2id: dict[str, int] | None = None,
    chunk_stride: int | None = None,
) -> LoaderResult:
    """Character-level WikiText-2 dataloader with Tiny Shakespeare fallback."""
    file_path = resolve_wikitext2_text_file(split=split)
    dataset = CharLevelDataset(
        file_path=file_path,
        seq_len=seq_len,
        max_samples=max_samples,
        char2id=char2id,
        chunk_stride=chunk_stride,
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
    word2id: dict[str, int] | None = None,
    chunk_stride: int | None = None,
) -> LoaderResult:
    """Word-level dataloader from a custom text file."""
    dataset = TextFileDataset(
        file_path=file_path,
        seq_len=seq_len,
        vocab_size=vocab_size,
        max_samples=max_samples,
        word2id=word2id,
        chunk_stride=chunk_stride,
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
    char2id: dict[str, int] | None = None,
    chunk_stride: int | None = None,
) -> LoaderResult:
    """Character-level dataloader from a custom text file."""
    dataset = CharLevelDataset(
        file_path=file_path,
        seq_len=seq_len,
        max_chars=max_chars,
        max_samples=max_samples,
        char2id=char2id,
        chunk_stride=chunk_stride,
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
        target_tokens = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        return input_tokens, target_tokens


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
