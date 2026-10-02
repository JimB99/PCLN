# Experiments and benchmark results

Single reference for recorded training runs. **Checkpoints live under `results/` (gitignored).**

**Cite only the Causal ablation table.** Everything under Legacy used a non-causal encoder, overlapping eval, a 1k-sample cap, and/or incomparable tokenization.

Custom **10k word vocabulary**. These perplexities are **not** comparable to published WikiText-2 literature numbers (typically ~33k vocab).

---

## Causal ablation (2026-10-02)

Question: at ~4.2–4.5M parameters, WikiText-2 word LM, does a PCN refinement block beat a matched causal transformer encoder + decoder?

| Field | Value |
|-------|-------|
| Date | 2026-10-02 |
| Machine | NVIDIA GeForce GTX 1650 (4 GB) |
| PyTorch | 2.10.0+cu130 |
| Seed | 42 (Python/NumPy/PyTorch; CUDA kernels not forced deterministic) |
| Dataset | WikiText-2 word-level, full split |
| Vocab | custom 10k + unk (**10001**) |
| `d_model` / layers / heads | 256 / 2 / 4 |
| Seq len | 128 |
| Train stride | 64 (overlap) |
| Eval stride | 128 (non-overlapping val and test) |
| Batch | 8 |
| Epochs / patience | 28 / 6 |
| Memory | **off** |
| Label smoothing | 0 |
| Adaptive-K | off |
| Eval tokens | val 213760, test 241152 |

| Run | PCN | Params | Best epoch | Train-loop best val | Val NLL | Val PPL | Test NLL | Test PPL |
|-----|-----|--------|------------|---------------------|---------|---------|----------|----------|
| wiki_tf | none | 4,150,801 | 26 | 4.5921 | 4.5920 | 98.70 | 4.5231 | **92.12** |
| wiki_pcn | self-PCN | 4,414,481 | 22 | 4.6462 | 4.6462 | 104.19 | 4.5719 | 96.73 |
| wiki_temporal | temporal | 4,546,321 | 23 | 4.4104 | 4.4104 | 82.30 | 4.3509 | **77.55** |

Command: `python scripts/run_causal_ablation.py --max-hours 6.5`.

Checkpoints: `results/causal_ablation/<id>/best_model.pt`.

### Interpretation

- **Self-PCN lost.** `wiki_pcn` test PPL 96.73 vs transformer 92.12 despite +264k parameters. Latent self-reconstruction did not help next-token modeling at this scale.
- **Temporal PCN won this protocol.** Test PPL 77.55 vs 92.12 is a **15.8% relative** drop. Temporal has +396k parameters vs the transformer (~9.5% more). The gain is therefore not a free lunch, but it is larger than the parameter gap and goes the opposite direction of self-PCN.
- Do **not** call this SOTA or compare it to AWD-LSTM / GPT-2 WikiText tables.
- Generations still degenerate without repetition penalty; samples are qualitative only (`results/causal_ablation/generation_samples.md`, gitignored).

`wiki_temporal` was killed during a first attempt at epoch 8 (agent session crash, Windows exit `1073807364`). That incomplete checkpoint (best at epoch 7) had test PPL 88.88. The numbers above are from a **fresh 28-epoch rerun**, not a resume.

---

## Legacy, not for citation

### Comparability problems

| Issue | Implication |
|-------|-------------|
| **Non-causal encoder** | Sprint WikiText runs (Aug 2026) used a bidirectional encoder with full-sequence loss. Not valid causal LM perplexity. |
| **Overlapping eval** | Some sprint evals reused train `chunk_stride`, double-counting tokens. |
| **Phase 4 micro-runs** | 1000 train samples, 8 epochs — not full-data. |
| **Feb 2026 benchmarks** | Invalid (char-level and exp4 bugs). Do not cite. |

### Phase 4 micro-benchmarks (2026-08-30)

1000 train / 100 val sequences, 8 epochs, GTX 1650 low-VRAM. Mixed word vs char losses in one table — not an ablation.

| Run | Best val loss | Notes |
|-----|---------------|-------|
| exp1_baseline | 4.1315 | Word-level WikiText-2 |
| exp2_char_level | 4.9394 | Char-level WikiText-2 |
| exp3_dynamic_neurons | 3.6437 | 128 neurons, low-VRAM |
| exp4_all_features | 5.0518 | Char + dynamic (128) + MoE |

Dynamic neurons looked better here and **lost** on full-data sprint training. Gate mass was near-uniform (~1.5–1.6%).

### Sprint training (2026-08-30, full WikiText-2, non-causal)

`d_model=256`, `seq_len=128`, `batch_size=8`. Bidirectional encoder. Do not cite as LM perplexity.

| Run | Best val loss | Epochs | Notes |
|-----|---------------|--------|-------|
| wiki_baseline | 4.1192 | 30 | Vocab 10,001 |
| wiki_stride64 | 4.0334 | 40 | Chunk stride 64 |
| wiki_dynamic | 4.2523 | 15 | Dynamic neurons 128, top-32 |
| shakespeare_char | 2.0880 | 40 | Tiny Shakespeare, char vocab |

An internal overlapping eval of `wiki_stride64` reported a test NLL around 3.97. That figure used the invalid protocol above.

Qualitative leftovers that still match the new table: extra epochs past ~30 tended to overfit; dynamic neurons did not beat a dense block at full data with a shorter budget.

---

## Recording new results

After `python scripts/run_causal_ablation.py`, replace or append a dated section. See [REPRODUCING.md](REPRODUCING.md).
