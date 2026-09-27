# Experiments and benchmark results

Single reference for recorded training and micro-benchmark runs. **Checkpoints live under `results/` (gitignored).**

## Comparability

| Issue | Implication |
|-------|-------------|
| **Non-causal encoder** | Sprint WikiText runs (Aug 2026) used a bidirectional encoder with full-sequence loss. Numbers are **not** valid causal LM perplexities for external comparison. |
| **Causal defaults** | New training (`train_full.py` defaults) uses a causal encoder; retrain before publishing perplexity. |
| **Phase 4 micro-runs** | 1000 train samples, 8 epochs, mixed loss settings vs sprint—use only for ablations, not vs sprint. |
| **Feb 2026 benchmarks** | Invalid (char-level and exp4 bugs fixed Aug 2026). Do not cite. |

---

## Run metadata (Phase 4 micro-benchmarks)

| Field | Value |
|-------|-------|
| Date | 2026-08-30 |
| Machine | GTX 1650 (4 GB VRAM), low-VRAM mode |
| PyTorch | 2.10.0+cu130 |
| Dataset | WikiText-2 (`data/wikitext2/`, HF parquet download) |
| Train samples | 1000 (default) |
| Epochs | 8 |

### Phase 4 validation loss (1k samples, 8 epochs)

| Run | Best val loss | Notes |
|-----|---------------|-------|
| exp1_baseline | 4.1315 | Word-level WikiText-2 |
| exp2_char_level | 4.9394 | Char-level WikiText-2 |
| exp3_dynamic_neurons | 3.6437 | 128 neurons, low-VRAM |
| exp4_all_features | 5.0518 | Char + dynamic (128) + MoE |

Best val loss in this suite: **exp3_dynamic_neurons** (3.6437 vs exp1 4.1315, ~11.8% relative). Short training and non-sprint protocol—do not equate with full-data sprint winners.

### Generation quality (Phase 4)

Prompt `"the king said"` (word-level), `--max-len 40`. Models are under-trained; outputs show repetition and noise. One illustrative degeneration: word-level baseline loops high-frequency tokens (`said` / punctuation) instead of coherent prose—motivating repetition penalty and n-gram blocking in `chat.py`.

Full samples: `results/generation_samples.md` when generated locally.

### Phase 4 notes

- Low-VRAM mode: exp3/exp4 use 128 neurons / top-32 (512-neuron configs exceed 4 GB VRAM).
- Neuron analysis (exp3): gate norms near-uniform (~1.6% top neuron mass)—limited specialization at this scale.

---

## Sprint training (2026-08-30, full WikiText-2)

Full-sequence next-token loss, shared train/val vocab, full WikiText-2 (not 1k cap). Config: `d_model=256`, `seq_len=128`, `batch_size=8`.

| Run | Best val loss | Train loss (final) | Epochs | Notes |
|-----|---------------|-------------------|--------|-------|
| wiki_baseline | 4.1192 | 3.39 | 30 | Vocab 10,001 |
| wiki_stride64 | **4.0334** | — | 40 | Chunk stride 64; **best val in sprint** |
| wiki_dynamic | 4.2523 | 3.81 | 15 | Dynamic neurons 128, top-32 |
| shakespeare_char | 2.0880 | 2.12 | 40 | Tiny Shakespeare, char vocab |

### Test perplexity (wiki_stride64)

| Split | NLL | Perplexity |
|-------|-----|------------|
| Test | 3.9703 | **53.0** |

Command: `python scripts/eval_perplexity.py --checkpoint results/sprint/wiki_stride64/best_model.pt --split test`

### Findings

1. **Overlapping chunks:** `--chunk-stride 64` improved best val loss from 4.1192 (wiki_baseline) to **4.0334** and test perplexity from ~58.2 to **53.0** vs the non-overlap baseline checkpoint path.
2. **Dynamic neurons at full scale:** With 15 epochs, wiki_dynamic **did not** beat wiki_baseline (4.25 vs 4.12)—needs matched epoch budget or architecture tuning before claiming a win (consistent with Phase 4 micro-scale win not transferring).
3. **Extra epochs on wiki_baseline:** Resuming 23→50 epochs lowered train loss but **did not** improve val (overfitting); early stopping around epoch 30 would have been sufficient.

### Dynamic neurons (wiki_dynamic)

Neuron gate analysis (`scripts/exp3_neuron_analysis.py`): top gate neuron ~1.5% of norm mass—near-uniform routing at this scale.

---

## Recording new results

After `python scripts/run_benchmarks.py` and `analyze_benchmarks.py`, append a dated section here with machine, PyTorch, dataset, and val/test metrics. See [REPRODUCING.md](REPRODUCING.md).
