# Reproducing benchmarks and full training

Checklist for NVIDIA GPU machines. CPU-only hosts should use [SETUP.md](SETUP.md) smoke tests only; the full benchmark suite can take 10–24+ hours on CPU.

## 1. Environment

```bash
cd PCLN
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Install details: [SETUP.md](SETUP.md).

## 2. Quick sanity check (~1 min)

```bash
pytest tests/ -q
python scripts/train_full.py --dataset dummy --epochs 1 --batch-size 8
```

## 3. Full benchmark suite (~1–2 hours on GTX 1650 class)

Prior Feb 2026 results are **invalid** (char-level and exp4 bugs fixed Aug 2026). Rerun after syncing:

```bash
python scripts/run_benchmarks.py --experiments all
python scripts/analyze_benchmarks.py --results-dir results
python scripts/exp3_neuron_analysis.py --checkpoint results/exp3_dynamic_neurons/best_model.pt
```

**4 GB VRAM:** `run_benchmarks.py` auto-enables low-VRAM mode (128 neurons for exp3/exp4). Override with `--low-vram` / `--no-low-vram`.

WikiText-2 downloads to `data/wikitext2/` on first run (HuggingFace parquet shards).

| ID | Config |
|----|--------|
| exp1_baseline | Word-level WikiText-2 (causal encoder) |
| exp2_char_level | Char-level WikiText-2 |
| exp3_dynamic_neurons | Factorized dynamic neurons, word-level |
| exp4_all_features | Char + dynamic neurons + MoE (stacked blocks) |
| exp5_temporal_pcn | Temporal + hierarchical PCN |

```bash
python scripts/run_benchmarks.py --experiments exp5_temporal_pcn
```

Report: `results/BENCHMARK_REPORT.md`

## 4. Record results

Copy key metrics into [EXPERIMENTS.md](EXPERIMENTS.md).

## 5. Checkpoint cleanup (optional)

```bash
mkdir -p results/checkpoints_archive
for d in results/exp*; do
  mkdir -p results/checkpoints_archive/$(basename "$d")
  mv "$d"/checkpoint_epoch[0-6].pt results/checkpoints_archive/$(basename "$d")/ 2>/dev/null || true
done
```

Keep `best_model.pt` in each experiment folder.

## 6. After benchmarks

- Review neuron specialization (exp3 analysis plots)
- Review MoE load-balance in exp4 logs
- Optional: held-out perplexity via `scripts/eval_perplexity.py`
- Scale experiments: larger `d_model`, transformer baseline, publication figures
