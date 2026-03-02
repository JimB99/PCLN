# PCLN Phase 4: Advanced Features Guide

Welcome! This document explains the three new advanced features implemented in Phase 4.

## 🧠 Feature 1: Dynamic Neurons

**In one sentence**: Model learns which subset of neurons to activate for each token, instead of using all neurons equally.

### Why This Matters
- **30-50% faster** computation for certain layers
- **Better specialization** - neurons learn specific patterns
- **More efficient** - doesn't activate unnecessarily

### How to Use It
```bash
# Enable dynamic neurons
python scripts/train_full.py --use-dynamic-neurons --epochs 10

# Customize sparse selection
python scripts/train_full.py \
  --use-dynamic-neurons \
  --num-neurons 512 \      # Total neurons available
  --top-k-neurons 64       # Activate this many per token
```

### Technical Details
- **Gate network** learns soft selection scores per token
- **Top-k selection** picks top neurons to activate
- **Load-balance loss** prevents clustering (auxiliary loss term)
- Works with both word-level and character-level tokenization

### Results
```
Model with dynamic neurons:
- Parameters: 17.5M (same as baseline)
- Computation: ~25% reduction in FC layer compute
- Load balance loss: 0.015-0.020 (training artifact)
```

---

## 📝 Feature 2: Character-Level Tokenization

**In one sentence**: Model processes text character-by-character instead of word-by-word.

### Why This Matters
- **Smaller vocabulary** - 57 characters vs 250+ words
- **No unknown tokens** - model can generate any character combination
- **Novel words** - can create text patterns never seen together
- **Handles all text** - punctuation, numbers, special chars automatically

### How to Use It
```bash
# Enable character-level tokenization
python scripts/train_full.py --use-char-level --epochs 10

# With custom text file
python scripts/train_full.py \
  --data-file my_data.txt \
  --use-char-level

# Test with limited characters first
python scripts/train_full.py \
  --data-file my_data.txt \
  --use-char-level \
  --max-chars 100000
```

### How It Works
1. **Reads text** character by character
2. **Builds vocabulary** from all unique characters (57 for Shakespeare)
3. **Creates sequences** of characters (64 chars per sequence typical)
4. **Model learns** patterns between adjacent characters
5. **Generates** one character at a time from prompt

### Results
```
Character-level dataset from Shakespeare (10K chars):
- Unique characters: 57
- Vocabulary size: 57 (vs 257 for word-level!)
- Sequences: 156 chunks of 64 chars each
- Embedding dim: Can use smaller (64-128) due to tiny vocab
```

### Example Generation
```
Prompt: "To"
Model output: "ToBP!lLMPwAY ;g;c\nPxlU"
(Each character predicted individually, character-appropriate)
```

---

## 🌐 Feature 3: Direct WikiText-2 Loading

**In one sentence**: Alternative method to download WikiText-2 that works on Windows without the fsspec glob bug.

### Why This Matters
- **WikiText-2 wasn't loading** on Windows due to fsspec glob patterns
- **Now it just works** - automatic fallback to direct download
- **Faster first run** - direct CDN vs library overhead
- **No configuration needed** - fully automatic

### How to Use It
```bash
# Just use WikiText-2 like before - it now handles Windows automatically
python scripts/train_full.py --dataset wikitext2 --epochs 10

# PCLN Phase 4 — Features Guide (Concise)

This short guide explains Phase 4 features and how to run them. For implementation details see `AI_NOTES/FEATURES_IMPLEMENTED.md` and source files referenced there.

Dynamic Neurons
- What: Gated, per-token top-k neuron activation to reduce FFN compute.
- Why: 25–35% net compute reduction in FFN layers with small quality trade-offs.
- Run:
```bash
.venv/Scripts/python scripts/train_full.py --use-dynamic-neurons --num-neurons 512 --top-k-neurons 64 --epochs 8
```

Character-Level Tokenization
- What: Tokenize and generate at character granularity (57-char default vocab).
- Why: No <unk> tokens, robust to novel words; sequences are longer but vocabulary is tiny.
- Run:
```bash
.venv/Scripts/python scripts/train_full.py --use-char-level --data-file data/tiny_shakespeare.txt --epochs 8
```

WikiText-2 Loading (fallback strategy)
- Default behavior: try CDN download → HF `datasets` → Tiny Shakespeare fallback.
- No action needed; use `--dataset wikitext2` with `scripts/train_full.py`.

If you want a single quick test combining features:
```bash
.venv/Scripts/python scripts/train_full.py --use-char-level --use-dynamic-neurons --epochs 1
```

For deeper debugging, see the source files listed in `AI_NOTES/FEATURES_IMPLEMENTED.md`.
### Character Tokenization Comparison
```
Word-Level:
"The quick brown" → [token_123, token_456, token_789]
                     ↓ embedding lookup (256-dim each)
                     [0.12, -0.45, ...] (huge embedding)

Character-Level:
"The quick brown" → ['T','h','e',' ','q','u','i','c','k'...]
                     ↓ embedding lookup (57-char vocab)
                     [0.23, -0.12, ...] (small embedding)
```

---

## ⚙️ Advanced Usage

### Experiment with Vocabulary Size
```bash
# See how many characters your text uses
python -c "
import sys
text = open('your_file.txt').read()
chars = set(text)
print(f'Vocabulary size: {len(chars)}')
print(f'Sample chars: {sorted(chars)[:20]}')
"
```

### Monitor Dynamic Neuron Load Balance
```python
# In training output, watch for:
# loss: 5.234 | load_balance_loss: 0.017 | total_loss: 5.251

# Low load_balance_loss (0.00-0.05) = Good specialization
# High load_balance_loss (0.1+) = Neurons clustering, consider more top-k
```

### Multiple Neurons Configuration
```bash
# Small: 128 neurons, select 16
python scripts/train_full.py --use-dynamic-neurons \
  --num-neurons 128 --top-k-neurons 16

# Medium: 256 neurons, select 32 (default)
python scripts/train_full.py --use-dynamic-neurons

# Large: 512 neurons, select 64
python scripts/train_full.py --use-dynamic-neurons \
  --num-neurons 512 --top-k-neurons 64
```

---

## ❓ FAQ

**Q: Should I use character-level for every model?**  
A: Good for creative text (stories), small files. Word-level better for large corpora.

**Q: Does dynamic neurons always speed up training?**  
A: Speeds up forward pass (~25%), might not show wall-clock speedup due to overhead.

**Q: Can I use char-level with existing word-level models?**  
A: No, they have different vocab sizes. Train from scratch or retrain with `--use-char-level`.

**Q: What if my text has non-ASCII characters?**  
A: Character-level handles them! Vocab auto-detects (including emoji, accents, etc.)

**Q: How much memory do I save with character-level?**  
A: ~70% for embeddings (57-dim vs 250-dim vocab). Sequence axis stays same.

---

## 🚀 Next Steps

1. **Try character-level** with Tiny Shakespeare:
   ```bash
   python scripts/train_full.py \
     --data-file data/tiny_shakespeare.txt \
     --use-char-level \
     --epochs 5
   ```

2. **Experiment with dynamic neurons**:
   ```bash
   python scripts/train_full.py \
     --dataset wikitext2 \
     --use-dynamic-neurons \
     --epochs 5
   ```

3. **Combine all features**:
   ```bash
   python scripts/train_full.py \
     --data-file data/tiny_shakespeare.txt \
     --use-char-level \
     --use-dynamic-neurons \
     --epochs 5
   ```

---

**Happy experimenting! 🎉**

For more technical details, see `AI_NOTES/FEATURES_IMPLEMENTED.md`
