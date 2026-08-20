#!/usr/bin/env python3
from pathlib import Path

files = [Path("README.md")]
issues = []

for f in files:
    if not f.exists():
        issues.append(f"{f} not found")
        continue
    s = f.read_text(encoding="utf-8")
    if "contentReference" in s or "oaicite" in s:
        issues.append(f"{f}: contains citation artifacts (contentReference/oaicite)")
    in_fence = False
    for i, line in enumerate(s.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if "pip install" in line and not in_fence:
            issues.append(f"{f}: un-fenced pip install on line {i}")
            break

if issues:
    print("Issues found:")
    for it in issues:
        print("-", it)
    raise SystemExit(1)
else:
    print("No obvious doc artifacts found in README.md")
