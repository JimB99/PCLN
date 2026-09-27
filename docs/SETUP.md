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

**CPU-only install:**

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**CUDA 12.1 install:**

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

## Full benchmark suite

See **[REPRODUCING.md](REPRODUCING.md)**. Do not run the full 4×8-epoch suite on CPU-only hardware.

## Training recipes

```bash
# WikiText-2 word-level
python scripts/train_full.py --dataset wikitext2 --epochs 8

# Overlapping chunks (more sequences per epoch)
python scripts/train_full.py --dataset wikitext2 --chunk-stride 64 --seq-len 128 --epochs 8

# Multi-hour GPU queue (resume + samples)
python scripts/run_training_camp.py --max-hours 6 --resume-all

# WikiText-2 character-level
python scripts/train_full.py --dataset wikitext2 --use-char-level --epochs 8

# Char + dynamic neurons + MoE (stacked PCN blocks)
python scripts/train_full.py --dataset wikitext2 --use-char-level \
  --use-dynamic-neurons --use-sparse-moe --epochs 8

# Dynamic neurons only
python scripts/train_full.py --use-dynamic-neurons --num-neurons 512 --top-k-neurons 64
```

WikiText-2 loading uses an automatic fallback chain (Salesforce HF CDN → S3 zip → Tiny Shakespeare) for `--dataset wikitext2`.

## Chat and evaluation

Checkpoints must include `vocab_mappings` (not the dummy smoke run).

```bash
python scripts/chat.py --checkpoint results/sprint/wiki_stride64/best_model.pt \
  --repetition-penalty 1.35 --no-repeat-ngram-size 3

python scripts/eval_perplexity.py --checkpoint results/sprint/wiki_stride64/best_model.pt --split test
```
