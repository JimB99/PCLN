# PCLN - Predictive Coding Language Model

A hybrid **Predictive Coding + Transformer + Memory + Sparse MoE** language model for text generation.

Biologically-inspired architecture combining:
- **Predictive Coding**: Iterative latent state refinement via prediction error minimization
- **Episodic + Semantic Memory**: Explicit long-term memory retrieval
- **Sparse Modular Experts**: Mixture-of-Experts for dynamic token routing
- **Dynamic Neurons**: Gated sparse activation (optional)
- **Character-Level Tokenization**: Optional per-character vocab (no `<unk>`)

---

## Quick Start

See **[docs/SETUP.md](docs/SETUP.md)** for CPU vs GPU install and smoke tests.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/train_full.py --dataset dummy --epochs 1 --batch-size 8
```

### Chat with a trained model

```bash
python scripts/chat.py --checkpoint ./checkpoints/best_model.pt
```

Checkpoints must be trained with real text data (not dummy) and include vocab mappings.

### Train examples

```bash
# WikiText-2 word-level
python scripts/train_full.py --dataset wikitext2 --epochs 8

# WikiText-2 character-level
python scripts/train_full.py --dataset wikitext2 --use-char-level --epochs 8

# All Phase 4 features (char + dynamic neurons + MoE)
python scripts/train_full.py --dataset wikitext2 --use-char-level \
  --use-dynamic-neurons --use-sparse-moe --epochs 8
```

---

## Phase 4 Features

### Dynamic neurons

Learns which neurons to activate per token (top-k of a larger pool). Reduces FFN compute; adds a load-balance auxiliary loss.

```bash
python scripts/train_full.py --use-dynamic-neurons --num-neurons 512 --top-k-neurons 64
```

### Character-level tokenization

~65-char vocab for Shakespeare; no `<unk>` tokens. Works with `--dataset wikitext2` or `--data-file`.

```bash
python scripts/train_full.py --dataset wikitext2 --use-char-level
```

### WikiText-2 loading

Automatic fallback chain: Salesforce HF CDN → S3 zip → Tiny Shakespeare. No extra config needed for `--dataset wikitext2`.

### Combined flags

`--use-dynamic-neurons` and `--use-sparse-moe` stack as **sequential PCN blocks** (dynamic first, then MoE).

### Known limitations

- Dynamic neurons still compute all neuron outputs then gather top-k (sparse routing, not sparse FLOPs).
- Sparse MoE uses Python loops over sequence length (research-scale only).

---

## Architecture

```
Input Tokens → Encoder → PCN Blocks → Memory → Decoder → Output Tokens
```

---

## Repository Structure

```
src/model/           # Architecture
src/data/            # Data loading
scripts/             # train_full.py, chat.py, run_benchmarks.py
docs/                # SETUP.md, GPU_NEXT_STEPS.md, benchmark_summary.md
.cursor/rules/       # Cursor agent rules
```

---

## System Requirements

- **Python**: 3.10 or 3.11
- **GPU**: Optional; recommended for full benchmarks (GTX 1650 class ~1–2 h for suite)
- **CPU-only**: Fine for tests and smoke training; full suite not recommended

---

## Benchmarks

**On GPU laptop:** follow **[docs/GPU_NEXT_STEPS.md](docs/GPU_NEXT_STEPS.md)** after syncing.

Prior Feb 2026 results are invalid. Record new results in [docs/benchmark_summary.md](docs/benchmark_summary.md).

---

## Project Status

| Phase | Status |
|-------|--------|
| 1–3 | Complete (PCN, memory, MoE) |
| 4 | Code complete; benchmarks pending GPU rerun |
| 5 | Scaling and publication — after valid benchmarks |

---

## License

MIT License — see LICENSE file.
