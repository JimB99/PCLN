# Benchmark Summary

Fill this file after running the corrected benchmark suite on a GPU machine.

## Run metadata

| Field | Value |
|-------|-------|
| Date | |
| Machine | |
| PyTorch | |
| Train samples | 1000 (default) |
| Epochs | 8 |

## Results

| Experiment | Best val loss | Notes |
|------------|---------------|-------|
| exp1_baseline | | word-level WikiText-2 |
| exp2_char_level | | char-level WikiText-2 |
| exp3_dynamic_neurons | | dynamic neurons, word-level |
| exp4_all_features | | char + dynamic + MoE (stacked blocks) |

## Best overall

- Winner:
- vs baseline improvement:

## Generation samples

(Add 1–2 lines per experiment from `results/exp*/training.log` or manual chat)

## Prior results

Feb 2026 benchmarks are invalid due to char-level and exp4 configuration bugs. Do not use pre-fix numbers.
