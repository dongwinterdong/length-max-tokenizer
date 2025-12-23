#!/usr/bin/env python3
"""
Plot one or more loss curves (CSV from extract_loss_curve.py) into a single figure.
Outputs PNG + PDF.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    loss: list[float] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            loss.append(float(row["loss"]))
    return steps, loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="Path to CSV (repeatable)")
    ap.add_argument("--label", action="append", required=True, help="Curve label (repeatable)")
    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--out_pdf", type=str, required=True)
    ap.add_argument("--title", type=str, default="loss vs step")
    ap.add_argument("--y_col", type=str, default="loss", choices=["loss", "bpc"], help="Which column to plot")
    args = ap.parse_args()

    if len(args.csv) != len(args.label):
        raise SystemExit("--csv and --label must have the same count")

    out_png = Path(args.out_png)
    out_pdf = Path(args.out_pdf)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8.5, 4.8), dpi=150)
    for csv_path, label in zip(args.csv, args.label):
        path = Path(csv_path)
        steps: list[int] = []
        ys: list[float] = []
        with path.open() as f:
            r = csv.DictReader(f)
            for row in r:
                steps.append(int(row["step"]))
                ys.append(float(row[args.y_col]))
        plt.plot(steps, ys, marker="o", markersize=3, linewidth=1.6, alpha=0.85, label=label)

    plt.title(args.title)
    plt.xlabel("step")
    plt.ylabel(args.y_col)
    plt.grid(True, linewidth=0.4, alpha=0.35)
    plt.legend(frameon=False, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    print("wrote", out_png)
    print("wrote", out_pdf)


if __name__ == "__main__":
    main()


