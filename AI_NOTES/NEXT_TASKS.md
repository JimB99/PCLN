# Next Steps After Benchmarking

All experiments are complete. This file lists immediate actionable items to finish Phase 4.

Immediate actions

- Review the comparative report: `results/BENCHMARK_REPORT.md` (contains final losses and notes)
- Inspect generation samples in each experiment folder (`results/exp*/generation_samples.txt` if present)
- Compute final metrics (perplexity) on a held-out test split if desired

Analysis tasks

- Neuron specialization study (exp3): visualize neuron activation maps and load-balance loss trends
- MoE tuning (exp4): evaluate routing weights, load-balance auxiliary loss, and experiment with top-k experts or scaling factor

Checkpoint management

- Keep: `results/exp*/best_model.pt` for each experiment
- Optional: keep last checkpoint `checkpoint_epoch7.pt` for quick resume
- Archive or delete intermediate `checkpoint_epoch*.pt` files to save space (see recommended commands below)

Recommended commands (safe pruning)
```bash
# Create an archive directory
mkdir -p results/checkpoints_archive

# Move intermediate checkpoints to archive, keep best_model.pt and final checkpoint
for d in results/exp*; do
	mkdir -p results/checkpoints_archive/$(basename "$d")
	mv "$d"/checkpoint_epoch[0-6].pt results/checkpoints_archive/$(basename "$d")/ 2>/dev/null || true
done
```

Follow-up (Phase 5 prep)

- Scale experiments (d_model=512 / 1024) and run on larger dataset
- Prepare figures and methods for publication

If you want, I can:
- Run the neuron-specialization analysis for exp3 and produce plots
- Prune checkpoints now and archive them under `results/checkpoints_archive/`
- Run a held-out perplexity evaluation across the 4 models
