#!/usr/bin/env python3
"""Multi-hour / multi-day GPU training camp for PCLN.

Runs sequential training jobs with --resume, time budget, chat evals, and status logs.
Designed for overnight or weekend runs on a local GPU (e.g. GTX 1650).

Examples:
  python scripts/run_training_camp.py --max-hours 8 --resume-all
  python scripts/run_training_camp.py --max-hours 48 --resume-all   # weekend
  python scripts/run_training_camp.py --jobs wiki_dynamic --max-hours 6
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / "results" / "sprint"
STATUS = ROOT / "docs" / "TRAINING_CAMP_STATUS.md"
PYTHON = sys.executable

# Shared WikiText-2 word-level settings (full data, full-sequence loss in train_full)
WIKI_BASE = [
    "--dataset", "wikitext2",
    "--batch-size", "8",
    "--d-model", "256",
    "--seq-len", "128",
    "--vocab-size", "10000",
    "--num-train-samples", "0",
    "--num-val-samples", "0",
]


@dataclass
class CampJob:
    id: str
    name: str
    cmd: list[str]
    chat_checkpoint: Path | None
    chat_prompt: str


# Shared WikiText-2 word-level settings (full data, full-sequence loss in train_full)
WIKI_BASE = [
    "--dataset", "wikitext2",
    "--batch-size", "8",
    "--d-model", "256",
    "--seq-len", "128",
    "--vocab-size", "10000",
    "--num-train-samples", "0",
    "--num-val-samples", "0",
]


def build_jobs(sprint: Path) -> list[CampJob]:
    return [
        CampJob(
            id="wiki_baseline",
            name="WikiText baseline resume -> 50 epochs",
            cmd=[
                PYTHON, "scripts/train_full.py",
                *WIKI_BASE,
                "--epochs", "50",
                "--checkpoint-dir", str(sprint / "wiki_baseline"),
                "--resume",
            ],
            chat_checkpoint=sprint / "wiki_baseline" / "best_model.pt",
            chat_prompt="the king said",
        ),
        CampJob(
            id="wiki_stride64",
            name="WikiText baseline stride=64 -> 40 epochs (overlap)",
            cmd=[
                PYTHON, "scripts/train_full.py",
                *WIKI_BASE,
                "--chunk-stride", "64",
                "--epochs", "40",
                "--checkpoint-dir", str(sprint / "wiki_stride64"),
            ],
            chat_checkpoint=sprint / "wiki_stride64" / "best_model.pt",
            chat_prompt="the king said",
        ),
        CampJob(
            id="shakespeare_char",
            name="Tiny Shakespeare char-level -> 50 epochs",
            cmd=[
                PYTHON, "scripts/train_full.py",
                "--data-file", "data/tiny_shakespeare.txt",
                "--use-char-level",
                "--epochs", "50",
                "--batch-size", "8",
                "--d-model", "256",
                "--seq-len", "128",
                "--num-train-samples", "0",
                "--num-val-samples", "0",
                "--checkpoint-dir", str(sprint / "shakespeare_char"),
                "--resume",
            ],
            chat_checkpoint=sprint / "shakespeare_char" / "best_model.pt",
            chat_prompt="First Citizen:",
        ),
    ]


def parse_best_val(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    matches = re.findall(r"Best val loss: ([\d.]+)", log_path.read_text(encoding="utf-8", errors="ignore"))
    return matches[-1] if matches else None


def run_chat(checkpoint: Path, prompt: str) -> str:
    cmd = [
        PYTHON, "scripts/chat.py",
        "--checkpoint", str(checkpoint),
        "--prompt", prompt,
        "--max-len", "45",
        "--repetition-penalty", "1.35",
        "--no-repeat-ngram-size", "3",
        "--temperature", "0.7",
    ]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (r.stdout or "").strip()
    if "Generated:" in out:
        return out.split("Generated:", 1)[1].strip()
    return out[-500:] if out else "(no output)"


def append_status(lines: list[str]) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Training camp status",
        "",
        f"Updated: {datetime.now().isoformat()}",
        "",
    ]
    STATUS.write_text("\n".join(header + lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="PCLN multi-hour GPU training camp")
    parser.add_argument(
        "--max-hours",
        type=float,
        default=8.0,
        help="Stop starting new jobs after this many hours (default 8)",
    )
    parser.add_argument(
        "--resume-all",
        action="store_true",
        help="Pass --resume to every job (recommended)",
    )
    parser.add_argument(
        "--jobs",
        type=str,
        default="all",
        help="Job ids comma-separated, or 'all' (wiki_dynamic,wiki_baseline,shakespeare_char)",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip initial pytest",
    )
    args = parser.parse_args()

    SPRINT.mkdir(parents=True, exist_ok=True)
    camp_log = SPRINT / f"camp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    status_lines: list[str] = []
    deadline = time.time() + args.max_hours * 3600
    all_jobs = build_jobs(SPRINT)

    if args.jobs.strip().lower() == "all":
        jobs = all_jobs
    else:
        ids = {x.strip() for x in args.jobs.split(",")}
        jobs = [j for j in all_jobs if j.id in ids]
        if not jobs:
            print("No matching jobs for:", args.jobs)
            return 1

    with camp_log.open("w", encoding="utf-8") as log:
        log.write(f"Training camp started {datetime.now().isoformat()}\n")
        log.write(f"Max hours: {args.max_hours}\n")
        log.write(f"Jobs: {[j.id for j in jobs]}\n\n")

        if not args.skip_pytest:
            log.write("=== pytest ===\n")
            r = subprocess.run([PYTHON, "-m", "pytest", "tests/", "-q"], cwd=ROOT)
            log.write(f"pytest exit {r.returncode}\n\n")
            if r.returncode != 0:
                status_lines.append("- **pytest FAILED** — camp aborted")
                append_status(status_lines)
                return r.returncode

        cuda_check = subprocess.run(
            [PYTHON, "-c", "import torch; print(torch.cuda.is_available())"],
            cwd=ROOT, capture_output=True, text=True,
        )
        cuda_ok = "True" in (cuda_check.stdout or "")
        log.write(f"CUDA available: {cuda_ok}\n\n")
        if not cuda_ok:
            log.write("No CUDA — training jobs will be slow or fail on CPU.\n")
            status_lines.append("- **WARNING:** CUDA not available")

        for job in jobs:
            if time.time() >= deadline:
                log.write(f"Time budget reached — skipping {job.id}\n")
                status_lines.append(f"- `{job.id}`: skipped (time budget)")
                break

            if not args.resume_all:
                job.cmd = [c for c in job.cmd if c != "--resume"]

            log.write(f"=== {job.name} ===\n")
            log.write(" ".join(job.cmd) + "\n")
            start = time.time()
            proc = subprocess.run(job.cmd, cwd=ROOT)
            elapsed_min = (time.time() - start) / 60
            log.write(f"Exit {proc.returncode}, {elapsed_min:.1f} min\n\n")

            val = parse_best_val(SPRINT / job.id / "training.log")
            line = f"- `{job.id}`: exit {proc.returncode}, {elapsed_min:.0f} min"
            if val:
                line += f", best val **{val}**"
            status_lines.append(line)

            if proc.returncode != 0:
                status_lines.append(f"  - **FAILED** — see {SPRINT / job.id / 'training.log'}")
                append_status(status_lines)
                log.write("Job failed — camp stopped.\n")
                return proc.returncode

            if job.chat_checkpoint and job.chat_checkpoint.exists():
                sample = run_chat(job.chat_checkpoint, job.chat_prompt)
                log.write(f"Chat sample: {sample[:300]}\n\n")
                status_lines.append(f"  - Chat: `{sample[:120]}...`" if len(sample) > 120 else f"  - Chat: `{sample}`")

        log.write(f"Camp finished {datetime.now().isoformat()}\n")

    status_lines.append(f"\nLog: `{camp_log.relative_to(ROOT)}`")
    append_status(status_lines)
    print(f"Camp complete. Status: {STATUS}")
    print(f"Full log: {camp_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
