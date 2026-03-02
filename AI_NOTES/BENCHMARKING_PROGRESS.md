# Phase 4 Benchmarking — Summary (Completed)

**Last Updated**: February 24, 2026 — experiments completed

All four experiments finished (8 epochs each). A concise results summary and recommendations are in `results/BENCHMARK_REPORT.md`.

Quick summary:

- `exp1_baseline` — best val loss: **3.4879** (word-level baseline)
- `exp2_char_level` — best val loss: **3.3893** (best overall)
- `exp3_dynamic_neurons` — best val loss: **3.4511** (dynamic neurons; large model)
- `exp4_all_features` — best val loss: **3.4933** (char + dynamic + MoE)

Performance note:
- Small models (exp1/exp2): ~3s per epoch.  Large models (exp3/exp4): ~5 minutes per epoch (higher parameter count ~137M).

Next step: run `scripts/analyze_benchmarks.py` (or read `results/BENCHMARK_REPORT.md`) for the human-readable comparison and generation samples.

Quick commands:
```bash
# View the generated report
code results/BENCHMARK_REPORT.md

# Re-run analysis (if you update logs)
python scripts/analyze_benchmarks.py --results-dir results
```

See `NEXT_TASKS.md` for curated next actions (neuron analysis, MoE tuning, checkpoint pruning).
