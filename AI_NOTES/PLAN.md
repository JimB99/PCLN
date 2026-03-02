# Predictive Coding Language Model (PCLM) - Development Plan

A hybrid **Predictive Coding + Transformer + Memory + Sparse Modular** language model for text-to-text tasks.  
This repository aims to explore a biologically inspired yet computationally scalable alternative to standard LLMs by integrating:

- Predictive coding (latent belief refinement + error minimization)
- Sparse modular connectivity (Mixture-of-Experts style)
- Explicit memory (episodic + semantic)
- Amortized inference (fast feedforward + few refinement steps)
- Attention as predictive routing (uncertainty reduction)

---

## Project Goals

### Core objectives
- Build a **text-to-text predictive coding language model**
- Reduce hallucinations via explicit prediction error
- Improve reasoning via iterative belief refinement
- Add explicit memory beyond synaptic weights
- Use sparse modular connectivity instead of fully dense layers
- Remain GPU-scalable and trainable with modern tooling

### Non-goals (for v1)
- Multimodal (image/audio) input
- Billion-parameter scale
- Full biological realism

---

## Conceptual Architecture

```
Tokens
  ↓
````markdown
# Predictive Coding Language Model (PCLM)

A hybrid **Predictive Coding + Transformer + Memory + Sparse Modular** language model for text-to-text tasks.  
This repository explores a biologically inspired yet scalable alternative to standard LLMs by integrating:

- Predictive coding (latent belief refinement + error minimization)
- Sparse modular connectivity (Mixture-of-Experts style)
- Explicit memory (episodic + semantic)
- Amortized inference (fast feedforward + few refinement steps)
- Attention as predictive routing (uncertainty reduction)

---

## 1. Project Goals

### Core objectives
- Build a **text-to-text predictive coding language model**
- Reduce hallucinations via explicit prediction error
- Improve reasoning via iterative belief refinement
- Add explicit memory beyond synaptic weights
- Use sparse modular connectivity instead of fully dense layers
- Remain GPU-scalable and trainable with modern tooling

### Non-goals (for v1)
- Multimodal (image/audio) input
- Billion-parameter scale
- Full biological realism

---

## 2. Conceptual Architecture

```
Tokens
↓
Sparse Transformer Encoder (amortized inference)
↓
Hierarchical Latent Belief States (z₁ … zₙ)
↓
Predictive Coding Refinement Loop (2–5 steps)
↓
Memory Retrieval (episodic + semantic)
↓
Token Decoder
```

Each layer contains:
- latent state z
- generative prediction ẑ
- error unit ε = z − ẑ

---

## 3. Core Components

### 3.1 Predictive Coding Block (PCN Layer)
Each block performs:
1. Feedforward initialization (amortized inference)
2. Iterative refinement (K steps)
3. Output refined latent state

### 3.2 Memory System
- **Episodic Memory**: Stores compressed latent belief states; KV retrieval via attention
- **Semantic Memory**: Slow-updated facts and abstractions

### 3.3 Sparse Modular Connectivity
- Mixture-of-Experts (MoE)
- Gated subnetworks
- Only a few modules active per token

### 3.4 Attention as Predictive Routing
Attention scores consider expected reduction in future prediction error (not just similarity)

---

## 4. Training Objectives

Total loss example:

```
L = λ1 * L_next_token
  + λ2 * L_pcn_error
  + λ3 * L_moe_balance
  + λ4 * L_memory_consistency
```

Where:
- `L_pcn_error`: ||z − ẑ||²
- `L_next_token`: cross-entropy
- `L_moe_balance`: load balance regularizer for MoE
- `L_memory_consistency`: correctness of retrieved memory

---

## 5. Development Roadmap & Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ COMPLETE | Minimal PCN + Transformer |
| Phase 2 | ✅ COMPLETE | Episodic + Semantic Memory |
| Phase 3 | ✅ COMPLETE | Sparse Modular MoE PCN |
| Phase 4 | 🔶 IN PROGRESS | Dynamic neurons, char-level, WikiText-2 |
| Phase 5 | ⏳ TODO | Scaling, benchmarking, publication |

---

## 6. Training Data Sources

Open text datasets for experiments:
- Wikipedia dumps (dumps.wikimedia.org)
- WikiText-2 / WikiText-103
- Tiny Shakespeare (fallback for quick tests)
- OpenWebText / Common Crawl / The Pile

Suggested dataset tooling: Hugging Face `datasets` library

---

## 7. Tooling & Dependencies

Recommended stack:
- Python 3.10+
- PyTorch
- NumPy, tqdm, matplotlib
- sentencepiece or tokenizers
- PyYAML

Install example:

```bash
pip install -r requirements.txt
```

---

## 8. Benchmarking

Compare against a small Transformer baseline. Metrics to collect:
- Perplexity
- Hallucination rate
- Reasoning/QA accuracy
- Long-context coherence
- Training time / tokens-per-second

Suggested experiments:
- Abalanced: arithmetic/logic tasks
- Long-context completion
- Memory retrieval accuracy

---

## 9. Stability & Safety Mechanisms

- Damping factor in refinement loop
- Gradient clipping
- Layer normalization on latent states
- Early stopping for oscillations
- Uncertainty threshold to produce "I don't know"

---

## 10. Research Questions

Key questions:
1. Can predictive coding improve reasoning in NLP?
2. Does explicit memory reduce hallucination?
3. Can sparse modular PCNs scale effectively?
4. How many refinement steps are optimal?

---

## 11. Getting Started

1. Clone repo
2. Create virtual environment
3. Download dataset (WikiText-2 or Tiny Shakespeare)
4. Run `scripts/train_full.py` (or `scripts/train.py` for older entrypoints)

Example (Windows PowerShell):

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python scripts/train_full.py --epochs 1 --use-char-level
```

---

## 12. References & Resources

- Hugging Face datasets: https://huggingface.co/datasets
- Wikipedia dumps: https://dumps.wikimedia.org/
- The Pile: https://pile.eleuther.ai/

---

**Project Vision:**
A language model that performs **belief inference**, detects its own errors, remembers explicitly, and learns with sparse modular structure—bridging neuroscience and modern NLP engineering.
````

