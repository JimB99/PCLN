# PCLN Phase 5 Quality Sprint — Results

**Date:** 2026-08-30 | **Machine:** GTX 1650 4GB | **Duration:** ~6 hours total

## What we changed

| Change | Why |
|--------|-----|
| **Full-sequence loss** | Train every next-token position (was last-token only) |
| **Shared train/val vocab** | Validation uses training `word2id` / `char2id` |
| **`--num-train-samples 0`** | Full WikiText-2 (~16k sequences/epoch) vs 1k cap |
| **WikiText parquet download** | HF shards → `data/wikitext2/` |

Code: `scripts/train_full.py`, `src/data/__init__.py`, `scripts/run_sprint.py`, `tests/test_full_sequence_loss.py`

---

## Training results

| Run | Best val loss | Train loss (final) | Epochs | Wall time | Checkpoint |
|-----|---------------|-------------------|--------|-----------|------------|
| **wiki_baseline** | **4.1192** | 3.39 | 30 | 51 min | `results/sprint/wiki_baseline/best_model.pt` (84 MB) |
| **shakespeare_char** | **2.0880** | 2.12 | 40 | 29 min | `results/sprint/shakespeare_char/best_model.pt` (26 MB) |
| **wiki_dynamic** | 4.2523 | 3.81 | 15 | 278 min | `results/sprint/wiki_dynamic/best_model.pt` (473 MB) |

Config shared: `d_model=256`, `seq_len=128`, `batch_size=8`, full-sequence loss.

Wiki runs: vocab 10,001, full WikiText-2 train/val. Dynamic: 128 neurons, top-32.

---

## vs Phase 4 micro-benchmarks (Aug 2026)

| | Phase 4 | Sprint |
|--|---------|--------|
| Data | 1k samples, Shakespeare fallback | Full WikiText-2 |
| Loss | Last token only | All positions |
| Vocab | 257 words | 10,001 words |
| Best val (baseline) | 4.1315 | 4.1192 |
| Generation | Punctuation noise | Recognizable phrases (repetition) |

**Dynamic neurons:** At micro-scale exp3 won (3.64 vs 4.13). At full scale with 15 epochs, dynamic **did not** beat baseline (4.25 vs 4.12) — needs more epochs or architecture fix before claiming a win.

---

## Generation samples

Prompts via `scripts/chat.py --max-len 50`, temperature 0.9.

### wiki_baseline (word)
- **Prompt:** `the king said`
- **Output:** `" , said was said said said said said said said said said said said said said said said said said said said said said the said said said said said said said said said said said said said said said said said said said said said said said said`

### wiki_dynamic (word + dynamic neurons)
- **Prompt:** `the king said`
- **Output:** `the king , the king king king king king king king king king king king king king king king king king king king king king king king king king king king king king king king king king king king king king king king king king king king king king`
- **Note:** Keeps “the king” anchor better than baseline; strong repetition.

### shakespeare_char (char)
- **Prompt:** `First Citizen:`
- **Output:** `, titttitttt tttttttttita 'eettim'trtitt 'efito't fitttt, 'ttitot tte ll 't, 't`
- **Note:** Val loss 2.09 is good for 65-char vocab; sampling still noisy (needs lower temperature or greedy decode).

Full log: `results/sprint/generation_samples.md`

---

## Neuron analysis (wiki_dynamic)

- Script: `scripts/exp3_neuron_analysis.py`
- Report: `results/sprint/wiki_dynamic_analysis/REPORT.md`
- 20 gate/neuron tensors, 128 neurons auto-detected
- Top gate neuron ~1.5% of norm mass — still near-uniform; limited specialization at this scale

---

## Disk layout

```
results/sprint/
  wiki_baseline/best_model.pt
  shakespeare_char/best_model.pt
  wiki_dynamic/best_model.pt
  wiki_dynamic_analysis/     # plots + REPORT.md
  checkpoints_archive/       # intermediate epochs (if archived)
```

Re-run sprint: `python scripts/run_sprint.py`

Chat: `python scripts/chat.py --checkpoint results/sprint/wiki_baseline/best_model.pt --prompt "your text"`

---

## Recommended next steps (Phase 5)

1. **Repetition penalty** in `chat.py` generation (reduce “said said said” loops)
2. **Wiki dynamic 30 epochs** (match baseline training budget) — ~9h on GTX 1650; or cloud GPU
3. **Test-split perplexity** script on WikiText-2 held-out test
4. **Fix `DynamicNeuronLayer` param growth** before 512-neuron configs
5. **Overlapping chunks** (`stride=1` or `seq_len/2`) for more training signal per token

---

## Tests

`pytest tests/ -q` → **19 passed** (includes full-sequence loss tests)
