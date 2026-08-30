# Benchmark Summary

Fill this file after running the corrected benchmark suite on a GPU machine.

## Run metadata

| Field | Value |
|-------|-------|
| Date | 2026-08-30 |
| Machine | GTX 1650 (4 GB VRAM), low-VRAM mode |
| PyTorch | 2.10.0+cu130 |
| Dataset | WikiText-2 (`data/wikitext2/`, HF parquet download) |
| Train samples | 1000 (default) |
| Epochs | 8 |

## Sprint results (2026-08-30)

Full-scale training with full-sequence loss — see **[docs/SPRINT_RESULTS.md](SPRINT_RESULTS.md)**.

| Sprint run | Best val loss | Notes |
|------------|---------------|-------|
| wiki_baseline | 4.1192 | full WikiText-2, 30ep, vocab 10k |
| wiki_dynamic | 4.2523 | dynamic neurons 128, 15ep |
| shakespeare_char | 2.0880 | char-level Tiny Shakespeare |

Phase 4 table below used 1k samples and last-token loss only.
|------------|---------------|-------|
| exp1_baseline | 4.1315 | word-level WikiText-2 |
| exp2_char_level | 4.9394 | char-level WikiText-2 |
| exp3_dynamic_neurons | 3.6437 | dynamic neurons (128 neurons, low-VRAM) |
| exp4_all_features | 5.0518 | char + dynamic (128) + MoE (stacked blocks) |

## Best overall

- Winner: exp3_dynamic_neurons (best val loss 3.6437)
- vs baseline improvement: 11.81% (vs exp1 4.1315)

## Generation samples

Prompt: word-level → `"the king said"`, char-level → `"The king"` (`--max-len 40`)

| Experiment | Sample output |
|------------|---------------|
| exp1_baseline | `. and the ) , . and , , , and =` |
| exp2_char_level | `rl s trone s arr tacs ongn` / `se mtaf toa` |
| exp3_dynamic_neurons | `the the by of . and to . in to as . the = the . . the , In . . the` |
| exp4_all_features | `iatheresken caerrd oes eeinnsehesheeehe` |

Full log: `results/generation_samples.md`. Quality is weak (8 epochs on 1000 WikiText-2 samples); expect better text with more data/training.

## Notes

- GTX 1650 auto-enabled `--low-vram`: exp3/exp4 used `num_neurons=128`, `top_k_neurons=32` (512-neuron configs exceed 4 GB VRAM and caused a system crash).
- WikiText-2 download URLs updated Aug 2026 (HuggingFace parquet shards).
- Intermediate checkpoints archived to `results/checkpoints_archive/` (epochs 0–6); each experiment folder keeps `best_model.pt` only.
- Neuron analysis: 20 gate/neuron tensors analyzed; plots in `results/exp3_analysis/` (see `REPORT.md`). Gate norms are near-uniform (top neuron ~1.6% of mass) — limited specialization at this scale.

## Prior results

Feb 2026 benchmarks are invalid due to char-level and exp4 configuration bugs. Do not use pre-fix numbers.

## Phase 5 sprint (2026-08-30)

See **[SPRINT_RESULTS.md](SPRINT_RESULTS.md)** for overnight WikiText-2 runs with full-sequence loss.

| Sprint run | Best val loss | Notes |
|------------|---------------|-------|
| wiki_baseline | 4.1192 | 30ep, vocab 10k, full WikiText-2 |
| shakespeare_char | 2.0880 | 40ep char-level Tiny Shakespeare |

**Improvement session** (same day): see **[IMPROVEMENT_SESSION.md](IMPROVEMENT_SESSION.md)**.
