"""Train/chat text codecs. Tokenization must match the dataset splitters."""

from __future__ import annotations


def word_tokenize(text: str) -> list[str]:
    """Same rule as TextFileDataset: whitespace split, punctuation kept."""
    return text.split()


class WordCodec:
    def __init__(self, word2id: dict[str, int], id2word: dict[int, str], unk_id: int):
        self.word2id = word2id
        self.id2word = {int(k): v for k, v in id2word.items()}
        self.unk_id = int(unk_id)

    def encode(self, text: str) -> list[int]:
        tokens = [self.word2id.get(w, self.unk_id) for w in word_tokenize(text)]
        return tokens if tokens else [self.unk_id]

    def decode(self, token_ids: list[int]) -> str:
        words = []
        for token_id in token_ids:
            word = self.id2word.get(int(token_id), "")
            if word and word != "<unk>":
                words.append(word)
        return " ".join(words)


class CharCodec:
    def __init__(self, char2id: dict[str, int], id2char: dict[int, str], unk_id: int = 0):
        self.char2id = char2id
        self.id2char = {int(k): v for k, v in id2char.items()}
        self.unk_id = int(unk_id)

    def encode(self, text: str) -> list[int]:
        tokens = [self.char2id.get(ch, self.unk_id) for ch in text]
        return tokens if tokens else [self.unk_id]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.id2char.get(int(tid), "") for tid in token_ids)


def encode_text(
    text: str,
    *,
    tokenization: str,
    word2id: dict[str, int] | None = None,
    char2id: dict[str, int] | None = None,
    unk_id: int = 0,
) -> list[int]:
    if tokenization == "char":
        mapping = char2id or {}
        tokens = [mapping.get(ch, unk_id) for ch in text]
        return tokens if tokens else [unk_id]
    mapping = word2id or {}
    tokens = [mapping.get(w, unk_id) for w in word_tokenize(text)]
    return tokens if tokens else [unk_id]
