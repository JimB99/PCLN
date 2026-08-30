# Improvement session — 2026-08-30

Agent-driven ~6h session: code improvements, disk cleanup, GPU training camp, evals.

**Wall time (GPU camp):** 10:51–14:07 (~3h 16m) | **Budget:** 6h (not fully used)

---

## Executive summary

| Outcome | Detail |
|---------|--------|
| **Best WikiText model** | `results/sprint/wiki_stride64/best_model.pt` — val **4.0334**, test ppl **53.0** |
| **Previous best** | `wiki_baseline` — val **4.1192**, test ppl **58.2** |
| **Disk freed** | ~17 GB → `results/` now ~198 MB |
| **Tests** | 21 passing (was 19; +chunk stride tests) |
| **Chat quality** | Decoding fixes help vs sprint; still WikiText-style repetition, not dialogue |

**Main win:** overlapping training chunks (`--chunk-stride 64`) beat the non-overlap baseline on validation and test perplexity.

**Main lesson:** More epochs on `wiki_baseline` (23→50) did **not** improve val loss (overfitting). Stride + fresh training was the right lever.

---

## Disk cleanup (removed)

| Removed | Approx. size | Why |
|---------|--------------|-----|
| `results/shakespeare_fallback_20260829/` | 3.6 GB | Invalid Shakespeare fallback benchmarks |
| `results/sprint/checkpoints_archive/` | 11 GB | Intermediate epoch checkpoints |
| `results/exp3_dynamic_neurons/`, `exp4_all_features/` | 1.7 GB | Old micro-benchmarks |
| `results/exp1_baseline/`, `exp2_char_level/` | ~100 MB | Pre-fix micro runs |
| `results/sprint/wiki_dynamic/` | 473 MB | Did not beat baseline at full scale |
| Docs | — | `CURSOR_AUTOMATION.md`, `SPRINT_MORNING_BRIEF.md`, `start_training_camp.bat` |

**Kept:** `wiki_baseline`, `wiki_stride64`, `shakespeare_char` best checkpoints only.

---

## Code changes

| File | Change |
|------|--------|
| `src/data/__init__.py` | `_build_sequence_chunks()`, `--chunk-stride` on all text/char loaders |
| `scripts/train_full.py` | `--chunk-stride`, `--save-epoch-checkpoints` (default off), resume stores `best_val_loss` |
| `scripts/chat.py` | `--no-repeat-ngram-size` (default 3) |
| `scripts/eval_perplexity.py` | Val/test NLL and perplexity from checkpoint vocab |
| `scripts/generate_samples.py` | Batch qualitative samples to markdown |
| `scripts/run_training_camp.py` | Queue: baseline → stride64 → shakespeare (no dynamic neurons) |
| `tests/test_full_sequence_loss.py` | +2 tests for chunk stride |
| `README.md`, `docs/benchmark_summary.md` | Updated entry points |

---

## Training results

### wiki_baseline (resume 23→50 epochs)

| Metric | Before session | After session |
|--------|----------------|---------------|
| Best val loss | 4.1192 (ep ~22) | **4.1192** (unchanged) |
| Val perplexity | 61.50 | 61.51 |
| Test perplexity | 58.19 | (unchanged checkpoint) |
| Epochs run | 30 | 50 total |

Train loss fell (3.39→3.33) but val rose → classic overfitting. Extra epochs were not useful.

### wiki_stride64 (new, stride=64, 40 epochs)

| Metric | Value |
|--------|-------|
| Best val loss | **4.0334** (epoch 17) |
| Val perplexity | **56.44** |
| Test NLL | 3.9703 |
| Test perplexity | **53.00** |
| Train sequences/epoch | ~32k (2× baseline) |
| Wall time | ~138 min |

### shakespeare_char (resume →50 epochs)

| Metric | Value |
|--------|-------|
| Best val loss | **2.0879** (tiny improvement from 2.0880) |
| Wall time | ~8 min |

---

## Generation (rep_penalty=1.35, ngram=3, temp=0.7)

See `results/sprint/generation_samples.md`.

- **wiki_stride64** on `the king said`: still heavy “said” loops but anchors “the king” better than sprint baseline.
- **wiki_baseline** samples in this run look worse than morning sprint — best checkpoint weights are from an earlier epoch; val-optimal ≠ best decode.
- **Recommendation for chat:** use `wiki_stride64` + `--repetition-penalty 1.35 --no-repeat-ngram-size 3` (try `4` for word models).

```bash
python scripts/chat.py \
  --checkpoint results/sprint/wiki_stride64/best_model.pt \
  --repetition-penalty 1.35 --no-repeat-ngram-size 3 --temperature 0.7
```

---

## Commands used

```bash
# Camp (this session)
python scripts/run_training_camp.py --max-hours 6 --resume-all

# Eval
python scripts/eval_perplexity.py --checkpoint results/sprint/wiki_stride64/best_model.pt --split test

# Samples
python scripts/generate_samples.py \
  --checkpoint results/sprint/wiki_baseline/best_model.pt \
  --checkpoint results/sprint/wiki_stride64/best_model.pt
```

Log: `results/sprint/camp_20260830_105047.log`, console: `results/sprint/improvement_camp_console.log`

---

## Next steps (not done this session)

1. Early stopping when val plateaus (would have stopped `wiki_baseline` around epoch 30)
2. Lower LR fine-tune from `wiki_stride64` best for a few epochs
3. Word-level no-repeat ngram size 4–5 or banned-token list for high-frequency words
4. Re-run Phase 4 micro-benchmarks on GPU with new data pipeline (valid comparison)

---

## Session timeline

| Time | Action |
|------|--------|
| ~10:49 | Cleanup ~17 GB stale checkpoints and obsolete docs |
| ~10:50 | Implement stride, eval, chat n-gram, checkpoint policy |
| 10:51 | Start 6h training camp |
| 11:40 | `wiki_baseline` finished (50 ep, val flat) |
| 13:58 | `wiki_stride64` finished (40 ep, **new best val**) |
| 14:07 | `shakespeare_char` finished, camp complete |
| 14:13 | Perplexity + generation samples |
