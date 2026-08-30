# GPU Laptop — Next Steps

Use this checklist after syncing the fixed PCLN repo to your NVIDIA GPU machine.

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

See [SETUP.md](SETUP.md) for details.

## 2. Quick sanity check (~1 min)

```bash
pytest tests/ -q
python scripts/train_full.py --dataset dummy --epochs 1 --batch-size 8
```

## 3. Full benchmark suite (~1–2 hours on GTX 1650 class)

Prior Feb 2026 results are **invalid** (char-level and exp4 bugs fixed Aug 2026). Rerun everything:

```bash
python scripts/run_benchmarks.py --experiments all
python scripts/analyze_benchmarks.py --results-dir results
python scripts/exp3_neuron_analysis.py --checkpoint results/exp3_dynamic_neurons/best_model.pt
```

On **GTX 1650 (4 GB VRAM)** the script auto-enables low-VRAM mode: exp3/exp4 use 128 neurons instead of 512 (512-neuron models are ~1.5 GB per checkpoint and can crash the machine). Force with `--low-vram` or disable with `--no-low-vram`.

WikiText-2 is downloaded from HuggingFace parquet shards into `data/wikitext2/` on first run.

Experiments (8 epochs each, 1000 train samples):

| ID | Config |
|----|--------|
| exp1_baseline | word-level WikiText-2 |
| exp2_char_level | char-level WikiText-2 |
| exp3_dynamic_neurons | dynamic neurons, word-level |
| exp4_all_features | char + dynamic neurons + MoE (stacked blocks) |

Report output: `results/BENCHMARK_REPORT.md`

## 4. Record results

Copy key numbers into [benchmark_summary.md](benchmark_summary.md).

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

- Review neuron specialization (exp3 plots from analysis script)
- Review MoE load-balance in exp4 logs
- Optional: held-out perplexity on WikiText-2 test split
- Phase 5 (later): scale `d_model` to 512/1024, transformer baseline, publication figures

## Do not run on CPU-only laptop

The full 4×8-epoch suite on CPU can take 10–24+ hours. Use CPU machine for dev and smoke tests only.
