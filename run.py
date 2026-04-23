#!/usr/bin/env python3
from __future__ import annotations
import os
import re
import subprocess
import sys

INPUTS_DIR = os.path.join(os.path.dirname(__file__), "inputs")


def pretty(filename: str) -> str:
    name = filename.removesuffix(".txt")
    name = re.sub(r"^set\d+_", "", name)
    parts = name.split("_")
    task = parts[0].replace("-", " ").title()
    tags = " | ".join(p.replace("-", " ").upper() for p in parts[1:])
    return f"{task:<28} {tags}"


def main() -> None:
    files = sorted(f for f in os.listdir(INPUTS_DIR) if f.endswith(".txt"))
    if not files:
        print("No input sets found in inputs/")
        sys.exit(1)

    print("\n  ┌─────┬─────────────────────────────────────────────────────────┐")
    print(  "  │  #  │ Set                                                     │")
    print(  "  ├─────┼─────────────────────────────────────────────────────────┤")
    for i, f in enumerate(files, 1):
        print(f"  │ {i:>3} │ {pretty(f):<55} │")
    print(  "  └─────┴─────────────────────────────────────────────────────────┘")

    while True:
        raw = input("\n  Select set number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(files):
            chosen = files[int(raw) - 1]
            break
        print("  Invalid. Try again.")

    path = os.path.join(INPUTS_DIR, chosen)
    print(f"\n  Running: {chosen}\n  {'─' * 54}")

    with open(path) as f:
        result = subprocess.run([sys.executable, "main.py"], stdin=f)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
