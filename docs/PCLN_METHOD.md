# PCLN vs standard LLMs — method overview

For researchers and users evaluating this project.

## What PCLN is

**PCLN** (Predictive Coding Language Model) is a **text generator** built as a pipeline:

```
tokens → Transformer encoder → PCN refinement → Memory → Decoder → next-token logits
```

It is trained like a language model (predict the next token), but the **internal state** is refined by predictive-coding blocks and optional memory—not only a single transformer pass.

## Compared to a “normal” LLM (GPT-style)

| Aspect | Typical LLM (GPT) | PCLN |
|--------|-------------------|------|
| Core compute | Stacked transformer layers only | Transformer **encoder** + **PCN refinement** |
| Latent state | Hidden states per layer | Beliefs refined by minimizing **prediction error** |
| Iteration at inference | Fixed depth (layers) | Encoder once + **K refinement steps** in PCN blocks |
| Memory | Context window only (or external RAG) | Built-in **episodic** (KV store) + **semantic** (learned slots) |
| Sparsity | Dense FFN / MoE at scale | Optional **dynamic neurons** + **sparse MoE** PCN blocks |
| Training objective | Next-token CE on all positions | Same (after sprint fix); plus **PCN error** + MoE balance losses |
| Scale today | Billions of params, web-scale data | Research scale (~2–40M params, WikiText-2) |

### Predictive coding (the main idea)

**Legacy self-PCN** (still the default block if no temporal flags):

1. Starts from encoder output `z`
2. Predicts `z` from itself with a small network `f(z)`
3. Updates: `z ← z - α · (z - f(z))` for **K steps**

That is a denoiser. It does not predict the future.

**Temporal PCN** (`--use-temporal-pcn`) — the language-appropriate generative model:

1. Shift latents causally: position `t` sees `z_{t-1}` (learned start vector at `t=0`)
2. Predict `ẑ_t = f(z_{t-1})`
3. Error `e_t = z_t - ẑ_t`
4. Precision-weighted update `z ← z - α · Π(e) · e`

**Hierarchical PCN** (`--use-hierarchical-pcn`) adds a slower causal moving-average pathway (Rao–Ballard style): the pooled state predicts the next local mean and injects that error into tokens.

The encoder is **causal** on new training runs so full-sequence next-token loss is a real language-model objective.

### Memory (episodic + semantic)

- **Episodic:** ring buffer of latent vectors; attention retrieval by similarity. Can **store** turns during chat (session memory).
- **Semantic:** fixed learnable slots; soft attention over slots during training.

The decoder **fuses** retrieved memory with latent states before predicting tokens.

### Dynamic neurons & MoE (optional)

- **Dynamic neurons:** gate which “neuron” transforms apply per token (top-k of a pool).
- **Sparse MoE:** multiple expert PCN refinements; route tokens to top-k experts.

These are research modules for **modularity and efficiency**, not yet proven to beat dense baselines at full WikiText scale in our runs.

## Can PCLN beat normal LLMs?

**Honestly, not at today’s scale.**

Modern LLMs win on:

- **Data** (trillions of tokens vs our millions)
- **Parameters** (billions vs millions)
- **Engineering** (decades of tooling, distillation, instruction tuning)

PCLN’s bet is **different**, not bigger:

- Explicit belief refinement (interpretable errors, iterative inference)
- Native episodic memory without stuffing the context window
- Sparse / modular structure for specialized computation

To **compete on chat quality** with GPT-class models you would need comparable scale **or** a niche where PCN + memory helps (long sessions, continual adaptation, small domain).

Our sprint models already produce **word-like** continuations on WikiText but are not coherent chatbots yet—they need more data, training, and decoding fixes (repetition penalty, etc.).

## Learning while chatting

Three levels:

| Level | What | In PCLN today | Likelihood of good results |
|-------|------|---------------|----------------------------|
| **Session memory** | Remember prior turns without weight updates | Episodic store during chat (`store_memory=True`) | **High** — implemented |
| **Online fine-tuning** | Gradient steps on each turn | `--learn-on-chat` in `chat.py` | **Medium** — works, risk of forgetting / instability |
| **True continual learning** | Stable long-term improvement | Not implemented | **Hard** — needs replay, EWC, or PCN-specific consolidation |

ChatGPT does **not** learn from your messages by default; PCLN can optionally do small online updates—that is a deliberate design choice, not magic.

## Can you start from a pretrained model?

**Partially, with adaptation—not plug-and-play.**

| Component | Pretrain transfer |
|-----------|-------------------|
| Token embeddings + transformer encoder | **Yes** — init from small GPT-2 / similar (match `d_model`, layers) |
| PCN blocks | **No** — unique weights; train from scratch or small init |
| Memory modules | **No** — architecture-specific |
| Decoder head | **Partial** — if vocab matches; else retrain embedding + decoder |

Practical path:

1. Load **encoder** weights from a small HF model (same hidden size).
2. Random-init PCN + memory + decoder (or train decoder on your vocab).
3. Fine-tune on your corpus.

You cannot drop a GPT checkpoint into PCLN end-to-end because the **forward path differs** (PCN refinement + memory are not in GPT).

## Best configuration so far (our experiments)

| Use case | Recommendation |
|----------|----------------|
| WikiText quality | **wiki_baseline**, full data, full-sequence loss, 30+ epochs |
| Dynamic neurons | 128 neurons on 4GB GPU; needs **matched epoch budget** vs baseline |
| Fun coherent text fast | **Tiny Shakespeare char**, 40+ epochs |
| Chat | `wiki_stride64` + repetition penalty + n-gram blocking |

## Further improvements (roadmap)

Done in the 2026-09-02 audit pass: causal encoder, value retrieval, no train-time memory writes, temporal/hierarchical PCN, factorized neurons, vectorized MoE, nucleus sampling, train/chat tokenizer alignment, early stopping.

Still open:

1. Instruction-style fine-tuning (prompt/response pairs)
2. Text-backed episodic memory (store token summaries, not only latents)
3. Pretrained encoder init (small GPT-2 / similar) + small PCN stack
4. **Causal retrain** of WikiText so test perplexity is comparable to other LMs
5. True continual learning (replay / EWC) for `--learn-on-chat`

## Try chat (after sprint)

```bash
python scripts/chat.py \
  --checkpoint results/sprint/wiki_baseline/best_model.pt \
  --repetition-penalty 1.3

# With session memory (default) + optional online learning:
python scripts/chat.py --checkpoint results/sprint/wiki_baseline/best_model.pt --learn-on-chat
```

See also: `docs/SPRINT_RESULTS.md`, `docs/benchmark_summary.md`.
