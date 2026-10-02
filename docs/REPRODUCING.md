# Reproducing benchmarks and full training

CPU-only hosts: [SETUP.md](SETUP.md) smoke tests only.

## 1. Environment

```bash
cd PCLN
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

This machine's working install is PyTorch `2.10.0+cu130` on a GTX 1650; either CUDA wheel is fine if `torch.cuda.is_available()` is true.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Quick sanity check

```bash
pytest tests/ -q
python scripts/check_docs.py
python scripts/train_full.py --dataset dummy --epochs 1 --batch-size 8 --num-pcn-blocks 0 --no-use-memory
```

## 3. Locked causal ablation (quoteable)

This is the protocol to cite. Full WikiText-2, causal encoder, matched hyperparameters, memory off, non-overlapping val/test.

```bash
python scripts/run_causal_ablation.py --max-hours 6.5
```

Shared flags (also the `wiki_tf` command):

```bash
python scripts/train_full.py --dataset wikitext2 --seed 42 \
  --num-train-samples 0 --num-val-samples 0 \
  --seq-len 128 --chunk-stride 64 --eval-chunk-stride 128 \
  --batch-size 8 --d-model 256 --nhead 4 --num-encoder-layers 2 \
  --vocab-size 10000 --epochs 28 --patience 6 \
  --learning-rate 0.001 --weight-decay 1e-5 --grad-clip 1.0 \
  --dropout 0.1 --label-smoothing 0 --tie-embeddings --causal \
  --num-pcn-blocks 0 --no-use-memory \
  --checkpoint-dir results/causal_ablation/wiki_tf
```

- `wiki_pcn`: same with `--num-pcn-blocks 1`
- `wiki_temporal`: same as `wiki_pcn` plus `--use-temporal-pcn`

Do **not** `--resume` old `results/sprint/` checkpoints into this protocol.

After training:

```bash
python scripts/eval_perplexity.py --checkpoint results/causal_ablation/wiki_tf/best_model.pt --split test
```

Default eval stride is `seq_len` (non-overlapping). Copy metrics into [EXPERIMENTS.md](EXPERIMENTS.md).

WikiText-2 is downloaded to `data/wikitext2/` (gitignored). A failed download **raises**; it does not silently train on Tiny Shakespeare.

**4 GB VRAM:** batch size 8 is the locked setting. If you OOM, use batch size 4 for **all** runs, not a mix.

## 4. Optional 1k-sample smoke suite (not publishable)

`scripts/run_benchmarks.py` still runs 1000-train / 100-val sequences for 8 epochs. Use it only as a GPU smoke test. Do not cite it against the causal ablation or literature WikiText-2 tables.

```bash
python scripts/run_benchmarks.py --experiments exp1_baseline
```

## 5. Checkpoint cleanup (optional)

Keep `best_model.pt` in each experiment folder. Intermediate `checkpoint_epoch*.pt` files are not written unless `--save-epoch-checkpoints`.
