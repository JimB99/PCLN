#!/usr/bin/env python3
"""Generate chat samples from one or more checkpoints for qualitative review."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

DEFAULT_PROMPTS = [
    "the king said",
    "in the year",
    "the united states",
    "he was born",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Batch chat samples for PCLN checkpoints")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint path (repeat for multiple)",
    )
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "sprint" / "generation_samples.md")
    parser.add_argument("--max-len", type=int, default=45)
    parser.add_argument("--repetition-penalty", type=float, default=1.35)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args(argv)

    prompts = args.prompt or DEFAULT_PROMPTS
    lines = [
        "# PCLN generation samples",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        f"Settings: temp={args.temperature}, rep_penalty={args.repetition_penalty}, "
        f"ngram={args.no_repeat_ngram_size}",
        "",
    ]

    for ckpt in args.checkpoint:
        ckpt_path = Path(ckpt)
        name = ckpt_path.parent.name
        lines.append(f"## {name}")
        lines.append(f"Checkpoint: `{ckpt_path}`")
        lines.append("")
        for prompt in prompts:
            cmd = [
                PYTHON, "scripts/chat.py",
                "--checkpoint", str(ckpt_path),
                "--prompt", prompt,
                "--max-len", str(args.max_len),
                "--repetition-penalty", str(args.repetition_penalty),
                "--no-repeat-ngram-size", str(args.no_repeat_ngram_size),
                "--temperature", str(args.temperature),
            ]
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            out = (r.stdout or "").strip()
            gen = out.split("Generated:", 1)[1].strip() if "Generated:" in out else out[-200:]
            lines.append(f"- **Prompt:** `{prompt}`")
            lines.append(f"  - **Output:** `{gen}`")
            lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
