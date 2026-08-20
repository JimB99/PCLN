# PCLN Setup

## Requirements

- Python 3.10 or 3.11
- 4 GB+ RAM (34 GB recommended for larger models on CPU)
- NVIDIA GPU with CUDA 12.1 optional (recommended for full benchmarks)

## Create environment

```bash
cd PCLN
python -m venv .venv
source .venv/Scripts/activate   # Git Bash / Windows
# .venv\Scripts\Activate.ps1    # PowerShell
```

## Install PyTorch

**This laptop (CPU only, Intel Iris Xe):**

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**GPU laptop (NVIDIA, CUDA 12.1):**

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

See [requirements-gpu.txt](../requirements-gpu.txt) for the GPU one-liner reference.

## Smoke test (CPU, ~2–5 minutes)

```bash
pytest tests/ -q
python scripts/check_docs.py
python scripts/train_full.py --dataset dummy --epochs 1 --batch-size 8
```

## Full benchmark suite (GPU laptop, ~1–2 hours)

See **[GPU_NEXT_STEPS.md](GPU_NEXT_STEPS.md)** for the full checklist.

```bash
python scripts/run_benchmarks.py --experiments all
python scripts/analyze_benchmarks.py --results-dir results
```

Do not run the full 4×8-epoch suite on CPU-only hardware; exp3/exp4 can take many hours per experiment.
