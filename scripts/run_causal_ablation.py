#!/usr/bin/env python3
"""Locked causal WikiText-2 ablation: transformer vs self-PCN vs temporal PCN.

Shared hyperparameters; only the refinement block differs. Memory is off.
Does not resume old sprint checkpoints.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
OUT = ROOT / "results" / "causal_ablation"

SHARED = [
    "--dataset", "wikitext2",
    "--seed", "42",
    "--num-train-samples", "0",
    "--num-val-samples", "0",
    "--seq-len", "128",
    "--chunk-stride", "64",
    "--eval-chunk-stride", "128",
    "--batch-size", "8",
    "--d-model", "256",
    "--nhead", "4",
    "--num-encoder-layers", "2",
    "--vocab-size", "10000",
    "--epochs", "28",
    "--patience", "6",
    "--learning-rate", "0.001",
    "--weight-decay", "1e-5",
    "--grad-clip", "1.0",
    "--dropout", "0.1",
    "--label-smoothing", "0",
    "--tie-embeddings",
    "--causal",
    "--no-use-memory",
    "--lambda-error", "0.1",
]


JOBS = [
    {
        "id": "wiki_tf",
        "extra": ["--num-pcn-blocks", "0"],
    },
    {
        "id": "wiki_pcn",
        "extra": ["--num-pcn-blocks", "1"],
    },
    {
        "id": "wiki_temporal",
        "extra": ["--num-pcn-blocks", "1", "--use-temporal-pcn"],
    },
]


def run(cmd: list[str]) -> int:
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def eval_split(checkpoint: Path, split: str, json_out: Path) -> dict:
    code = run(
        [
            PYTHON,
            "scripts/eval_perplexity.py",
            "--checkpoint",
            str(checkpoint),
            "--split",
            split,
            "--json-out",
            str(json_out),
        ]
    )
    if code != 0:
        raise RuntimeError(f"eval_perplexity failed for {checkpoint} {split}")
    return json.loads(json_out.read_text(encoding="utf-8"))


def append_summary(lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "SUMMARY.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Causal ablation summary\n\n"
    path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="PCLN causal WikiText-2 ablation")
    parser.add_argument("--max-hours", type=float, default=6.5)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument(
        "--jobs",
        default="wiki_tf,wiki_pcn,wiki_temporal",
        help="Comma-separated job ids",
    )
    args = parser.parse_args()

    wanted = [x.strip() for x in args.jobs.split(",") if x.strip()]
    jobs = [j for j in JOBS if j["id"] in wanted]
    if not jobs:
        print("No matching jobs", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.max_hours * 3600

    if not args.skip_pytest:
        code = run([PYTHON, "-m", "pytest", "tests/", "-q"])
        if code != 0:
            print("pytest failed; aborting ablation", file=sys.stderr)
            return code

    cuda = subprocess.run(
        [PYTHON, "-c", "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    cuda_out = (cuda.stdout or "").strip().splitlines()
    cuda_ok = bool(cuda_out) and cuda_out[0].strip() == "True"
    device_name = cuda_out[1] if len(cuda_out) > 1 else ""
    print(f"CUDA available: {cuda_ok} {device_name}", flush=True)
    if not cuda_ok:
        print("No CUDA; aborting GPU ablation", file=sys.stderr)
        return 1

    append_summary(
        [
            f"Started: {datetime.now().isoformat()}",
            f"Device: {device_name}",
            f"Max hours: {args.max_hours}",
            f"Jobs: {[j['id'] for j in jobs]}",
            "",
        ]
    )

    for job in jobs:
        if time.time() >= deadline:
            append_summary([f"- `{job['id']}`: skipped (time budget)"])
            print(f"Time budget reached — skipping {job['id']}", flush=True)
            break

        ckpt_dir = OUT / job["id"]
        cmd = [
            PYTHON,
            "scripts/train_full.py",
            *SHARED,
            *job["extra"],
            "--checkpoint-dir",
            str(ckpt_dir),
        ]
        print(f"\n=== {job['id']} ===", flush=True)
        start = time.time()
        code = run(cmd)
        elapsed = (time.time() - start) / 60
        best = ckpt_dir / "best_model.pt"
        line = f"- `{job['id']}`: exit {code}, {elapsed:.1f} min"
        if code != 0:
            append_summary([line, "  - FAILED"])
            if best.exists():
                print("Training failed but best_model.pt exists; evaluating it.", flush=True)
            else:
                return code

        if best.exists():
            val = eval_split(best, "validation", ckpt_dir / "metrics_val.json")
            test = eval_split(best, "test", ckpt_dir / "metrics_test.json")
            merged = {
                "id": job["id"],
                "elapsed_min": elapsed,
                "val": val,
                "test": test,
            }
            (ckpt_dir / "metrics.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
            append_summary(
                [
                    line,
                    f"  - val NLL {val['nll']:.4f} PPL {val['perplexity']:.2f}",
                    f"  - test NLL {test['nll']:.4f} PPL {test['perplexity']:.2f}",
                    f"  - params {test['parameters']} causal={test['causal']} stride={test['eval_stride']}",
                ]
            )
        else:
            append_summary([line, "  - no best_model.pt"])

        if code != 0:
            return code

    print("Causal ablation finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
