# PCLN — Predictive Coding Language Model

Research-scale language model for a single question: **at ~4.2–4.5M parameters on WikiText-2, does a predictive-coding refinement block beat a matched causal transformer encoder + decoder?**

This is an architectural testbed, not a production LLM. Checkpoints are gitignored under `results/`.

Quoteable numbers: **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)** (causal ablation only). Method: **[docs/PCLN_METHOD.md](docs/PCLN_METHOD.md)**. Older sprint and 1k-sample runs are archived there as **not for citation**.

---

## Architecture

```
tokens → causal transformer encoder → optional PCN block → decoder → next-token logits
```

The locked ablation turns memory off so PCN is not confounded with a fusion MLP. Two PCN variants exist:

- **Self-PCN** (default `PCNBlock`): reconstruct each latent from itself (a learned denoiser, not Rao–Ballard sensory prediction).
- **Temporal PCN** (`--use-temporal-pcn`): predict `z_t` from `z_{t-1}` with precision-weighted updates.

`scripts/train_full.py` is the language-model trainer. `scripts/train.py` is a numpy/PyTorch smoke path and does **not** train PCLN.

---

## Five-minute path

```bash
cd PCLN
python -m venv .venv
source .venv/Scripts/activate          # Windows Git Bash
# source .venv/bin/activate            # Linux/macOS
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest tests/ -q
python scripts/check_docs.py
python scripts/train_full.py --dataset dummy --epochs 1 --batch-size 8 --num-pcn-blocks 0 --no-use-memory
```

Dummy checkpoints have no text vocabulary and cannot chat. GPU install: [docs/SETUP.md](docs/SETUP.md). Locked WikiText-2 command: [docs/REPRODUCING.md](docs/REPRODUCING.md).

---

## Causal ablation (the CV result)

Three matched runs, seed 42, full WikiText-2, custom 10k word vocab, causal encoder, train stride 64, **non-overlapping** val/test (stride 128), memory off, GTX 1650 4 GB:

| ID | Difference from transformer |
|----|-----------------------------|
| `wiki_tf` | `--num-pcn-blocks 0` |
| `wiki_pcn` | `--num-pcn-blocks 1` (self-PCN) |
| `wiki_temporal` | `--num-pcn-blocks 1 --use-temporal-pcn` |

```bash
python scripts/run_causal_ablation.py --max-hours 6.5
```

On a GTX 1650 (4 GB), seed 42, custom 10k word vocab, causal encoder, memory off, non-overlapping test:

| Run | Params | Test PPL |
|-----|--------|----------|
| Transformer (`wiki_tf`) | 4.15M | 92.12 |
| Self-PCN (`wiki_pcn`) | 4.41M | 96.73 |
| Temporal PCN (`wiki_temporal`) | 4.55M | **77.55** |

Self-PCN did not beat the transformer. Temporal PCN did on this protocol (15.8% relative test-PPL drop, with 9.5% more parameters). Full table, protocol, and caveats: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). These numbers are **not** comparable to published WikiText-2 literature perplexities.

---

## Known limitations

- Self-PCN is latent denoising, not cortical predictive coding of observations.
- Custom 10k word vocabulary; do not compare to AWD-LSTM / Transformer-XL tables.
- Optional memory, MoE, and dynamic neurons exist in code but are **out of** the locked ablation.
- Dynamic-neuron gates still score the full pool; sparse MoE loops over experts.
- WikiText-2 download no longer falls back to Tiny Shakespeare.

---

## Usage

| Task | Command |
|------|---------|
| Dummy smoke | `python scripts/train_full.py --dataset dummy --epochs 1 --batch-size 8 --num-pcn-blocks 0 --no-use-memory` |
| Locked ablation | `python scripts/run_causal_ablation.py --max-hours 6.5` |
| Test perplexity | `python scripts/eval_perplexity.py --checkpoint results/causal_ablation/wiki_temporal/best_model.pt --split test` |
| Chat (local ckpt) | `python scripts/chat.py --checkpoint results/causal_ablation/wiki_temporal/best_model.pt --repetition-penalty 1.35 --no-repeat-ngram-size 3` |

Eval defaults to non-overlapping chunks. Do not pass `--use-train-stride` for published PPL.

---

## Repository structure

```
src/model/           # Architecture
src/data/            # WikiText-2 and codecs
scripts/train_full.py
scripts/run_causal_ablation.py
scripts/eval_perplexity.py
scripts/chat.py
docs/                # SETUP, REPRODUCING, METHOD, EXPERIMENTS
.cursor/rules/       # Agent conventions
```

---

## System requirements

- **Python:** 3.10 or 3.11
- **GPU:** GTX 1650 class (4 GB) for the locked ablation (~5–6 h for three 28-epoch runs)
- **CPU:** tests and dummy smoke only

---

## License

MIT — see [LICENSE](LICENSE).
