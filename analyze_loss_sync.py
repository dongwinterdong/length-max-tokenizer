#!/usr/bin/env python3
"""
Analyze why two tokenizer runs show similar loss fluctuations over steps.

Inputs:
- Two CSVs with at least: step, loss
  (e.g. loss_curve_steps10000.csv and loss_curve_bpe_steps10000.csv)

Outputs:
- A delta CSV: step, loss_a, loss_b, delta_loss, bpc_a, bpc_b, delta_bpc
- Plots: overlay (already exists), delta plots, and residual (detrended) plots.

Key idea:
- If both runs use the same RNG seed and the same batch sampling logic,
  they see (nearly) the same raw-text batches at each step.
  Batch difficulty then dominates short-term loss fluctuations, creating
  synchronized oscillations even when tokenization differs.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Curve:
    steps: list[int]
    loss: list[float]


def _read_curve(path: Path) -> Curve:
    with path.open() as f:
        r = csv.DictReader(f)
        steps: list[int] = []
        loss: list[float] = []
        for row in r:
            steps.append(int(row["step"]))
            loss.append(float(row["loss"]))
    if not steps:
        raise ValueError(f"no points in {path}")
    return Curve(steps=steps, loss=loss)


def _pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("length mismatch")
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    va = 0.0
    vb = 0.0
    cov = 0.0
    for x, y in zip(a, b):
        dx = x - ma
        dy = y - mb
        va += dx * dx
        vb += dy * dy
        cov += dx * dy
    den = math.sqrt(va * vb)
    return cov / den if den > 0 else float("nan")


def _moving_avg(x: list[float], radius: int) -> list[float]:
    out: list[float] = []
    n = len(x)
    for i in range(n):
        a = max(0, i - radius)
        b = min(n, i + radius + 1)
        out.append(sum(x[a:b]) / (b - a))
    return out


def _residual(x: list[float], radius: int) -> list[float]:
    ma = _moving_avg(x, radius)
    return [xi - mi for xi, mi in zip(x, ma)]


def _write_delta_csv(
    *,
    out_csv: Path,
    steps: list[int],
    loss_a: list[float],
    loss_b: list[float],
    tpc_a: float,
    tpc_b: float,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ln2 = math.log(2.0)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss_a", "loss_b", "delta_loss", "bpc_a", "bpc_b", "delta_bpc"])
        for s, la, lb in zip(steps, loss_a, loss_b):
            bpc_a = (la * tpc_a / ln2) if tpc_a > 0 else float("nan")
            bpc_b = (lb * tpc_b / ln2) if tpc_b > 0 else float("nan")
            w.writerow([s, la, lb, lb - la, bpc_a, bpc_b, bpc_b - bpc_a])


def _plot_xy(
    *,
    out_png: Path,
    out_pdf: Path,
    x: list[int],
    y: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 4.2), dpi=150)
    plt.plot(x, y, marker="o", markersize=3, linewidth=1.6, alpha=0.9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linewidth=0.4, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_pdf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_csv", type=str, required=True)
    ap.add_argument("--b_csv", type=str, required=True)
    ap.add_argument("--a_name", type=str, default="A")
    ap.add_argument("--b_name", type=str, default="B")
    ap.add_argument("--tpc_a", type=float, required=True, help="Tokens-per-char for A (estimate)")
    ap.add_argument("--tpc_b", type=float, required=True, help="Tokens-per-char for B (estimate)")
    ap.add_argument("--ma_radius", type=int, default=3, help="Moving-average radius for residual")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    a = _read_curve(Path(args.a_csv))
    b = _read_curve(Path(args.b_csv))

    if a.steps != b.steps:
        raise SystemExit("step grids differ; rerun extraction with the same print_every")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # correlations
    corr_loss = _pearson(a.loss, b.loss)
    da = [a.loss[i + 1] - a.loss[i] for i in range(len(a.loss) - 1)]
    db = [b.loss[i + 1] - b.loss[i] for i in range(len(b.loss) - 1)]
    corr_dloss = _pearson(da, db)
    ra = _residual(a.loss, int(args.ma_radius))
    rb = _residual(b.loss, int(args.ma_radius))
    corr_res = _pearson(ra, rb)

    # write summary
    summary = out_dir / "loss_sync_summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"a={args.a_name} csv={args.a_csv} tpc={args.tpc_a}",
                f"b={args.b_name} csv={args.b_csv} tpc={args.tpc_b}",
                f"points={len(a.steps)} step_range=[{a.steps[0]},{a.steps[-1]}]",
                f"pearson(loss)={corr_loss:.6f}",
                f"pearson(delta_loss)={corr_dloss:.6f}",
                f"pearson(residual_ma{args.ma_radius})={corr_res:.6f}",
                "",
                "Interpretation:",
                "- High correlation indicates shared batch-difficulty fluctuations (paired sampling).",
                "- Tokenizer typically shifts absolute loss scale; short-term oscillations remain synchronized.",
                "",
            ]
        )
        + "\n"
    )

    # delta CSV
    delta_csv = out_dir / "delta_loss_bpc.csv"
    _write_delta_csv(
        out_csv=delta_csv,
        steps=a.steps,
        loss_a=a.loss,
        loss_b=b.loss,
        tpc_a=float(args.tpc_a),
        tpc_b=float(args.tpc_b),
    )

    # delta plots (loss and bpc)
    # Read back delta CSV for convenience
    steps: list[int] = []
    delta_loss: list[float] = []
    delta_bpc: list[float] = []
    with delta_csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            delta_loss.append(float(row["delta_loss"]))
            delta_bpc.append(float(row["delta_bpc"]))

    _plot_xy(
        out_png=out_dir / "delta_loss.png",
        out_pdf=out_dir / "delta_loss.pdf",
        x=steps,
        y=delta_loss,
        title=f"Δloss = loss({args.b_name}) - loss({args.a_name})",
        xlabel="step",
        ylabel="Δloss (nats/token)",
    )
    _plot_xy(
        out_png=out_dir / "delta_bpc.png",
        out_pdf=out_dir / "delta_bpc.pdf",
        x=steps,
        y=delta_bpc,
        title=f"Δbpc proxy = bpc({args.b_name}) - bpc({args.a_name})  (bpc≈loss*TPC/ln2)",
        xlabel="step",
        ylabel="Δbpc (bits/char)",
    )

    print("== done ==")
    print("summary =", summary)
    print("delta_csv=", delta_csv)
    print("plots   =", out_dir / "delta_loss.png", ",", out_dir / "delta_bpc.png")


if __name__ == "__main__":
    main()


