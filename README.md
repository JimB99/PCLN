# PCLN — Predictive Coding Language Model

Hybrid **predictive coding + transformer encoder + explicit memory + optional sparse MoE** for next-token language modeling at research scale.

**Status:** Active research prototype (~2–40M parameters, WikiText-2 scale). Checkpoints are not committed (`results/` is gitignored). Reported perplexities below were measured with a **non-causal encoder** unless retrained with current defaults; treat them as internal comparisons, not published LM benchmarks. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) and [docs/PCLN_METHOD.md](docs/PCLN_METHOD.md).

---

## Motivation

Standard decoder-only transformers map tokens to logits in a fixed stack of layers. PCLN adds **iterative latent refinement** (predictive-coding blocks), **retrieved episodic and semantic memory**, and optional **sparse modular experts** so internal state is updated by prediction error—not only by depth. The goal is to study whether explicit belief update and memory help at small scale before scaling data and parameters.

---

## Architecture

```
tokens → causal encoder → PCN refinement → memory → decoder → next-token logits
```

| Component | Role |
|-----------|------|
| Causal transformer encoder | Token representations (causal on new training runs) |
| PCN blocks | Minimize prediction error over K refinement steps |
| Episodic + semantic memory | Retrieve and fuse stored latents before decoding |
| Optional dynamic neurons / MoE | Top-k sparse transforms (Shazeer et al.–style routing at block level) |

Technical report: **[docs/PCLN_METHOD.md](docs/PCLN_METHOD.md)**.

### Implemented variants (opt-in flags)

| Flag | Idea | Reference |
|------|------|-----------|
| `--use-temporal-pcn` | Causal next-latent prediction (generative PCN) | Rao & Ballard (1999); temporal predictive coding |
| `--use-hierarchical-pcn` | Slow pooled state predicts local means | Rao–Ballard hierarchical PCN |
| `--adaptive-k` | Stop refinement when error is small | Adaptive inference depth |
| `--surprise-store-threshold` | Write episodic memory on high PCN error | Surprise-gated memory |
| `--use-dynamic-neurons` | Top-k gated units (factorized rank-1) | Conditional computation |
| `--use-sparse-moe` | Route tokens to top-k expert PCN blocks | Shazeer et al., Mixture-of-Experts |
| `--use-char-level` | Character vocabulary (no `<unk>`) | — |

Flags compose as **sequential PCN blocks** (e.g. dynamic neurons then MoE).

---

## Known limitations

- Dynamic neurons: sparse output path; gate still scores the full pool.
- Sparse MoE: loop over expert count, not sequence length.
- Legacy sprint checkpoints may be **non-causal**; `chat.py` / `eval_perplexity.py` respect checkpoint args.
- Pre-causal WikiText perplexities are **not** comparable to standard causal LM reports until retrain (see Experiments).

---

## Usage

Install and smoke test: **[docs/SETUP.md](docs/SETUP.md)**. Full GPU benchmarks: **[docs/REPRODUCING.md](docs/REPRODUCING.md)**.

| Task | Command |
|------|---------|
| Smoke train | `python scripts/train_full.py --dataset dummy --epochs 1 --batch-size 8` |
| WikiText-2 word LM | `python scripts/train_full.py --dataset wikitext2 --epochs 8` |
| Overlapping chunks | `python scripts/train_full.py --dataset wikitext2 --chunk-stride 64 --seq-len 128 --epochs 8` |
| Recommended retrain | `python scripts/train_full.py --dataset wikitext2 --chunk-stride 64 --seq-len 128 --epochs 30 --use-temporal-pcn --use-hierarchical-pcn --adaptive-k --patience 6` |
| Chat (local checkpoint) | `python scripts/chat.py --checkpoint results/sprint/wiki_stride64/best_model.pt --repetition-penalty 1.35 --no-repeat-ngram-size 3` |
| Test perplexity | `python scripts/eval_perplexity.py --checkpoint <path> --split test` |

Dummy training produces a checkpoint **without vocabulary**—not usable for chat. Real checkpoints must include `vocab_mappings`.

More train flags (char-level, MoE, training camp): see **Training recipes** in [docs/SETUP.md](docs/SETUP.md).

---

## Evaluation

Recorded runs and comparability notes: **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**.

---

## Repository structure

```
src/model/           # Architecture
src/data/            # Data loading
scripts/             # train_full.py, chat.py, run_benchmarks.py
docs/                # SETUP.md, REPRODUCING.md, PCLN_METHOD.md, EXPERIMENTS.md
.cursor/rules/       # Coding and experiment conventions (Cursor)
```

---

## System requirements

- **Python:** 3.10 or 3.11
- **GPU:** Optional; recommended for full benchmark suite (~1–2 h on GTX 1650 class)
- **CPU:** Tests and smoke training; full benchmark suite not recommended on CPU

---

## Roadmap

- Causal retrain on WikiText-2 with temporal/hierarchical PCN; refresh perplexity table
- Pretrained encoder init (small GPT-2–compatible stack)
- Instruction-style fine-tuning; text-backed episodic memory
- Continual learning (replay / EWC) for `--learn-on-chat`

---

## License

MIT — see [LICENSE](LICENSE).
