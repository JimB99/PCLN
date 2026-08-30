#!/usr/bin/env python3
"""Phase 5 quality sprint: scaled training + generation samples.

Runs sequentially (safe on 4 GB GPU). Logs to results/sprint/ and docs/SPRINT_RESULTS.md
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPRINT_DIR = ROOT / "results" / "sprint"
LOG = ROOT / "docs" / "SPRINT_RESULTS.md"
PYTHON = sys.executable


def run_step(name: str, cmd: list[str], log_lines: list[str]) -> int:
    print(f"\n{'='*60}\nSPRINT: {name}\n{'='*60}")
    log_lines.append(f"\n## {name}\n")
    log_lines.append(f"Started: {datetime.now().isoformat()}\n")
    log_lines.append(f"Command: `{' '.join(cmd)}`\n\n")
    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.time() - start
    log_lines.append(f"Exit code: {result.returncode}, elapsed: {elapsed/60:.1f} min\n")
    if result.stdout:
        log_lines.append("```\n" + result.stdout[-4000:] + "\n```\n")
    if result.stderr and result.returncode != 0:
        log_lines.append("stderr:\n```\n" + result.stderr[-2000:] + "\n```\n")
    return result.returncode


def chat_sample(checkpoint: Path, prompt: str, log_lines: list[str]) -> None:
    cmd = [
        PYTHON, "scripts/chat.py",
        "--checkpoint", str(checkpoint),
        "--prompt", prompt,
        "--max-len", "50",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (result.stdout or "").strip()
    log_lines.append(f"**Chat** `{checkpoint.parent.name}` prompt `{prompt}`:\n")
    log_lines.append(f"> {out.split('Generated:')[-1].strip() if 'Generated:' in out else out}\n")


def main():
    SPRINT_DIR.mkdir(parents=True, exist_ok=True)
    (SPRINT_DIR / "sprint_console.log").touch(exist_ok=True)
    log_lines = [
        "# PCLN Phase 5 Quality Sprint Results",
        "",
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Changes: full-sequence loss, shared train/val vocab, full WikiText-2.",
        "",
    ]

    steps = [
        (
            "wiki_baseline (30ep, full WikiText-2, vocab 10k)",
            [
                PYTHON, "scripts/train_full.py",
                "--dataset", "wikitext2",
                "--epochs", "30",
                "--batch-size", "8",
                "--d-model", "256",
                "--seq-len", "128",
                "--vocab-size", "10000",
                "--num-train-samples", "0",
                "--num-val-samples", "0",
                "--checkpoint-dir", str(SPRINT_DIR / "wiki_baseline"),
            ],
        ),
        (
            "shakespeare_char (40ep, char-level)",
            [
                PYTHON, "scripts/train_full.py",
                "--data-file", "data/tiny_shakespeare.txt",
                "--use-char-level",
                "--epochs", "40",
                "--batch-size", "8",
                "--d-model", "256",
                "--seq-len", "128",
                "--num-train-samples", "0",
                "--num-val-samples", "0",
                "--checkpoint-dir", str(SPRINT_DIR / "shakespeare_char"),
            ],
        ),
        (
            "wiki_dynamic (15ep, dynamic neurons 128)",
            [
                PYTHON, "scripts/train_full.py",
                "--dataset", "wikitext2",
                "--use-dynamic-neurons",
                "--num-neurons", "128",
                "--top-k-neurons", "32",
                "--epochs", "15",
                "--batch-size", "8",
                "--d-model", "256",
                "--seq-len", "128",
                "--vocab-size", "10000",
                "--num-train-samples", "0",
                "--num-val-samples", "0",
                "--checkpoint-dir", str(SPRINT_DIR / "wiki_dynamic"),
            ],
        ),
    ]

    for name, cmd in steps:
        code = run_step(name, cmd, log_lines)
        if code != 0:
            log_lines.append(f"**FAILED** step {name}\n")
            break

    log_lines.append("\n## Generation samples\n")
    samples = [
        (SPRINT_DIR / "wiki_baseline" / "best_model.pt", "the king said"),
        (SPRINT_DIR / "shakespeare_char" / "best_model.pt", "The king"),
        (SPRINT_DIR / "wiki_dynamic" / "best_model.pt", "the king said"),
    ]
    for ckpt, prompt in samples:
        if ckpt.exists():
            chat_sample(ckpt, prompt, log_lines)

    log_lines.append(f"\n---\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    LOG.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nSprint log written to {LOG}")


if __name__ == "__main__":
    main()
