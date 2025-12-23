#!/usr/bin/env python3
"""
Extract (step, loss, tok/s) points from validate_modern_arch_llama.py logs.
Writes a CSV suitable for plotting.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    log_path = Path(args.log)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Backward compatible:
    # - old: step=0000 loss=10.4906 tok/s=38808.6
    # - new: step=0000 loss=10.4906 bpc≈2.945 tok/s=38808.6 active_frac≈1.000
    pat = re.compile(
        r"^step=(\d+)\s+loss=([0-9.]+)"
        r"(?:\s+bpc≈([0-9.]+))?"
        r"\s+tok/s=([0-9.]+)"
        r"(?:\s+active_frac≈([0-9.]+))?"
    )

    rows: list[tuple[int, float, float, float, float]] = []
    for line in log_path.read_text().splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        step = int(m.group(1))
        loss = float(m.group(2))
        bpc = float(m.group(3)) if m.group(3) is not None else float("nan")
        toks = float(m.group(4))
        active_frac = float(m.group(5)) if m.group(5) is not None else float("nan")
        rows.append((step, loss, bpc, toks, active_frac))

    if not rows:
        raise SystemExit(f"no step lines found in: {log_path}")

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "bpc", "tok_per_s", "active_frac", "ppl_exp_loss"])
        for step, loss, bpc, toks, active_frac in rows:
            w.writerow([step, loss, bpc, toks, active_frac, math.exp(loss)])

    steps = [r[0] for r in rows]
    losses = [r[1] for r in rows]
    min_i = min(range(len(losses)), key=lambda i: losses[i])

    print("== extracted ==")
    print("log     =", log_path)
    print("out_csv =", out_csv)
    print("points  =", len(rows))
    print("first   =", rows[0])
    print("last    =", rows[-1])
    print("min_loss=", (steps[min_i], losses[min_i]))


if __name__ == "__main__":
    main()


