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

The Quick Start dummy run (`checkpoints/best_model.pt`) is **not** for text chat — it has no vocabulary.

```bash
python scripts/chat.py
```

Defaults to `results/sprint/wiki_stride64/best_model.pt` when present. Or specify explicitly:

```bash
python scripts/chat.py --checkpoint results/sprint/wiki_stride64/best_model.pt \
  --repetition-penalty 1.35 --no-repeat-ngram-size 3
```

```bash
python scripts/eval_perplexity.py --checkpoint results/sprint/wiki_baseline/best_model.pt --split test
```

Checkpoints must be trained with real text data (not dummy) and include vocab mappings.

### Train examples

```bash
# WikiText-2 word-level
python scripts/train_full.py --dataset wikitext2 --epochs 8

# Overlapping chunks (2x training signal per token)
python scripts/train_full.py --dataset wikitext2 --chunk-stride 64 --seq-len 128 --epochs 8

# Multi-hour GPU camp (resume + chat samples)
python scripts/run_training_camp.py --max-hours 6 --resume-all

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

`--use-dynamic-neurons` and `--use-sparse-moe` stack as **sequential PCN blocks** (dynamic first, then MoE). Combine with `--use-temporal-pcn` / `--use-hierarchical-pcn` the same way.

### Known limitations

- Dynamic neurons: output path is top-k / rank-1; the gate still scores all neurons.
- Sparse MoE loops over the expert count (small), not sequence length.
- Old sprint checkpoints were trained **non-causal**; `chat.py` / `eval_perplexity.py` keep that behavior unless the checkpoint args say otherwise.

---

## Architecture

```
tokens → causal encoder → PCN refinement → memory → decoder → next-token logits
```

New training is a **causal** language model. Prior WikiText numbers used a bidirectional encoder with full-sequence loss, so they are not honest LM scores — retrain before comparing perplexity.

**Inventions now in the code (opt-in flags):**
- **Temporal PCN** (`--use-temporal-pcn`): predict the next latent from the previous one (not self-reconstruction)
- **Hierarchical PCN** (`--use-hierarchical-pcn`): slow causal pooled state predicts local means
- **Adaptive K** (`--adaptive-k`): extra refinement only while prediction error stays large
- **Surprise-gated memory**: write episodic slots when PCN error exceeds `--surprise-store-threshold`
- **Factorized dynamic neurons**: rank-1 units, sparse output FLOPs
- **Vectorized MoE**: loop over experts, not sequence length

Recommended first retrain:

```bash
python scripts/train_full.py --dataset wikitext2 --chunk-stride 64 --seq-len 128 \
  --epochs 30 --use-temporal-pcn --use-hierarchical-pcn --adaptive-k --patience 6
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

See **[docs/PCLN_METHOD.md](docs/PCLN_METHOD.md)** for architecture vs standard LLMs, memory, and online learning.

**On GPU laptop:** follow **[docs/GPU_NEXT_STEPS.md](docs/GPU_NEXT_STEPS.md)** after syncing.

Prior Feb 2026 results are invalid. Record new results in [docs/benchmark_summary.md](docs/benchmark_summary.md).

---

## Project Status

| Phase | Status |
|-------|--------|
| 1–3 | Complete (PCN, memory, MoE) |
| 4 | Code complete; prior WikiText scores need a **causal** retrain (see `docs/SPRINT_RESULTS.md`) |
| 5 | Causal + temporal/hierarchical PCN in code; GPU retrain pending |

---

## License

MIT License — see LICENSE file.
