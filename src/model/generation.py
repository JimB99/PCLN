"""Shared decoding utilities (repetition penalty, n-gram ban, nucleus sample)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_history: list[int],
    penalty: float,
) -> torch.Tensor:
    """Reduce logits for tokens that appeared recently (HF-style)."""
    if penalty <= 0 or not token_history:
        return logits
    out = logits.clone()
    for tid in set(token_history):
        score = out[:, tid]
        out[:, tid] = torch.where(score > 0, score / penalty, score * penalty)
    return out


def ban_repeating_ngrams(
    logits: torch.Tensor,
    token_history: list[int],
    ngram_size: int,
) -> torch.Tensor:
    """Block tokens that would repeat an n-gram already seen in history."""
    if ngram_size <= 1 or len(token_history) < ngram_size - 1:
        return logits
    out = logits.clone()
    prefix = tuple(token_history[-(ngram_size - 1) :])
    banned: set[int] = set()
    for i in range(len(token_history) - ngram_size + 1):
        if tuple(token_history[i : i + ngram_size - 1]) == prefix:
            banned.add(token_history[i + ngram_size - 1])
    for tid in banned:
        out[:, tid] = float("-inf")
    return out


def apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits
    out = logits.clone()
    k = min(top_k, out.size(-1))
    threshold, _ = torch.topk(out, k, dim=-1)
    out[out < threshold[:, [-1]]] = float("-inf")
    return out


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens with mass >= top_p."""
    if top_p is None or top_p >= 1.0:
        return logits
    top_p = max(float(top_p), 0.0)
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    remove = cdf > top_p
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample (batch, 1) token ids from filtered logits."""
    scaled = logits / max(float(temperature), 1e-5)
    scaled = apply_top_k(scaled, top_k)
    scaled = apply_top_p(scaled, top_p)
    probs = F.softmax(scaled, dim=-1)
    finite = torch.isfinite(probs).all(dim=-1)
    safe = finite & (probs.sum(dim=-1) > 0)
    sampled = torch.multinomial(torch.where(safe.unsqueeze(-1), probs, torch.ones_like(probs)), 1)
    greedy = torch.argmax(logits, dim=-1, keepdim=True)
    return torch.where(safe.unsqueeze(-1), sampled, greedy)
