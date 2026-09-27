# PCLN: method overview

Technical report for researchers evaluating this repository.

## Abstract

PCLN (Predictive Coding Language Model) is a research-scale language model that combines a **causal transformer encoder**, **predictive-coding refinement blocks**, **episodic and semantic memory**, and optional **sparse experts**. Training uses next-token cross-entropy plus auxiliary PCN and load-balance losses. At current scale (~2–40M parameters, WikiText-2), PCLN does not match production LLM quality; the contribution is an explicit architectural testbed for belief refinement and memory. Recorded metrics and comparability caveats are in [EXPERIMENTS.md](EXPERIMENTS.md).

## Background

Decoder-only transformers (Vaswani et al., 2017; Radford et al., GPT series) compute hidden states in a fixed-depth stack. **Predictive coding** (Rao & Ballard, 1999) models cortex as hierarchical prediction-error minimization. PCLN inserts PCN-style updates between encoding and decoding so latents are iteratively corrected before logits are produced.

## Method

### Forward path

```
tokens → Transformer encoder → PCN refinement → Memory → Decoder → next-token logits
```

| Aspect | Typical dense LLM | PCLN |
|--------|-------------------|------|
| Core compute | Stacked transformer layers | Encoder + **PCN refinement** |
| Latent state | Layer hidden states | Beliefs updated by **prediction error** |
| Inference depth | Fixed layer count | Encoder once + **K** PCN steps (optional adaptive K) |
| Memory | Context window / external RAG | **Episodic** ring buffer + **semantic** slots |
| Sparsity | Dense or MoE FFN | Optional **dynamic neurons** + **sparse MoE** PCN blocks |
| Training loss | Next-token CE | CE + PCN error + MoE balance |

### Predictive coding blocks

**Legacy self-PCN** (default block without temporal flags): reconstruct each position from itself (denoising); not a forward-time generative model.

**Temporal PCN** (`--use-temporal-pcn`): causal prediction of the next latent from the previous state; precision-weighted error update—appropriate for language modeling.

**Hierarchical PCN** (`--use-hierarchical-pcn`): slower causal pooled pathway predicts local means (Rao–Ballard-style hierarchy).

New training runs use a **causal encoder** so full-sequence next-token loss is a standard LM objective.

### Memory

- **Episodic:** similarity-based retrieval over stored latents; optional writes during chat (`store_memory=True`).
- **Semantic:** fixed learnable slots with soft attention.

The decoder fuses retrieved memory with refined latents before the output projection.

### Optional sparsity

- **Dynamic neurons:** top-k gated transforms per token (factorized rank-1 implementation).
- **Sparse MoE:** route tokens to top-k expert PCN blocks (Shazeer et al., 2017).

These modules target modularity and compute efficiency; they have **not** consistently beaten the dense baseline at full WikiText scale in recorded runs ([EXPERIMENTS.md](EXPERIMENTS.md)).

## Experimental setup

- **Data:** WikiText-2 (Merity et al., 2016), Tiny Shakespeare for char-level demos; optional custom `--data-file`.
- **Scale:** ~2–40M parameters; GTX 1650 class GPU for benchmark suite; low-VRAM mode for 4 GB cards.
- **Evaluation:** validation loss, test perplexity via `eval_perplexity.py`, qualitative generation via `chat.py` with repetition controls.
- **Reproducibility:** `pytest tests/`, smoke `train_full.py --dataset dummy`, full suite in [REPRODUCING.md](REPRODUCING.md).

## Results

See **[EXPERIMENTS.md](EXPERIMENTS.md)**. Summary:

- Best sprint validation loss on full WikiText-2: **wiki_stride64** (val 4.0334, test perplexity 53.0 with overlapping chunks)—still pre-causal-retrain encoder unless re-run with current defaults.
- Dynamic neurons won a **short** Phase 4 micro-benchmark but not the full-data sprint at matched training budget.

## Limitations

PCLN at this scale does not compete with billion-parameter models on data volume, compute, or instruction tuning. Sprint generations exhibit repetition; decoding mitigations (repetition penalty, n-gram blocking) are required for interactive use. Pre-causal checkpoint metrics must not be quoted as standard LM perplexity without retraining.

## Online learning and memory at chat time

| Mode | Mechanism | Status |
|------|-----------|--------|
| Session memory | Episodic store without weight updates | Implemented |
| Online fine-tuning | `--learn-on-chat` gradient steps | Experimental; forgetting risk |
| Continual learning | Stable long-term adaptation | Not implemented |

## Transfer from pretrained LMs

| Component | Transfer |
|-----------|----------|
| Token embeddings + encoder | Possible from small GPT-2–compatible checkpoints (match `d_model`) |
| PCN, memory | Architecture-specific; train from scratch |
| Decoder | Partial if vocabulary aligns |

End-to-end GPT weights cannot be loaded: the forward path includes PCN and memory not present in GPT.

## Recommended configurations

| Use case | Recommendation |
|----------|----------------|
| Best recorded WikiText val / test ppl | **wiki_stride64** checkpoint path + causal retrain for publication |
| Dynamic neurons ablation | 128 neurons on 4 GB GPU; match epoch budget to baseline |
| Char-level demo | Tiny Shakespeare, 40+ epochs |
| Interactive decode | `wiki_stride64` + `--repetition-penalty 1.35 --no-repeat-ngram-size 3` |

## Related work

- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- Shazeer, N., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *ICLR*.
- Merity, S., et al. (2016). Pointer sentinel mixture models. *ICLR* (WikiText-2).
- Millidge, B., Tschantz, A., & Buckley, C. L. Predictive coding approximates backprop along arbitrary computation graphs (survey line for PCN and deep learning).

## Future work

Completed in the 2026-09 audit pass: causal encoder, value retrieval, train/chat tokenizer alignment, temporal/hierarchical PCN, factorized neurons, vectorized MoE, nucleus sampling, early stopping.

Open:

1. Instruction fine-tuning
2. Text-backed episodic memory (token summaries, not only latents)
3. Pretrained encoder initialization
4. **Causal retrain** and refreshed perplexity table
5. Continual learning (replay / EWC) for `--learn-on-chat`

## Chat (local checkpoint)

```bash
python scripts/chat.py \
  --checkpoint results/sprint/wiki_stride64/best_model.pt \
  --repetition-penalty 1.35 --no-repeat-ngram-size 3
```

Optional session memory and `--learn-on-chat` as described above.
