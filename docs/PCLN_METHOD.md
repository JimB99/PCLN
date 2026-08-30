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

In each PCN block the model:

1. Starts from encoder output `z`
2. Predicts `z` from itself with a small network `f(z)`
3. Updates: `z ← z - α · (z - f(z))` for **K steps**
4. Trains auxiliary loss to shrink prediction **errors**

So the representation is **explicitly refined** toward self-consistency, not only passed through feedforward layers. That is the biological / neuroscience motivation: perception as inference, not one-shot feedforward.

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

1. **Overlapping training chunks** (stride 1) — more supervision per token
2. **Instruction-style fine-tuning** — prompt/response pairs, not raw LM only
3. **Fix dynamic neuron parameterization** — true per-neuron weights, not `Linear(in, num_neurons × out)`
4. **Retrieval-augmented episodic memory** — store text summaries, not only latent means
5. **Larger encoder** (pretrained) + keep PCN stack small
6. **Test perplexity** on WikiText-2 test for apples-to-apples LM comparison

## Try chat (after sprint)

```bash
python scripts/chat.py \
  --checkpoint results/sprint/wiki_baseline/best_model.pt \
  --repetition-penalty 1.3

# With session memory (default) + optional online learning:
python scripts/chat.py --checkpoint results/sprint/wiki_baseline/best_model.pt --learn-on-chat
```

See also: `docs/SPRINT_RESULTS.md`, `docs/benchmark_summary.md`.
