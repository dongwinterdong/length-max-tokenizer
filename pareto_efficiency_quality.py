#!/usr/bin/env python3
"""
Efficiency–quality Pareto plots for tokenizers.

We want to highlight that Length-MAX can be advantageous even when bpc is higher:
- Efficiency: higher chars/token (lower TPC) -> more text per fixed token window and lower compute for the same text length.
- Quality: lower eval_bpc (bits/char proxy) -> better modeling per character.

This script reads our training logs (with '== eval ==' lines) and produces:
- A CSV table of: tokenizer, model_tag, best_eval_bpc, last_tok_per_s, chars_per_token, chars_per_s.
- A scatter plot: chars/token vs best_eval_bpc (lower-left is best; Length-MAX typically shifts right).
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_STEP_RE = re.compile(
    r"^step=(\d+)\s+loss=([0-9.]+)\s+bpc≈([0-9.]+)\s+tok/s=([0-9.]+)\s+active_frac≈([0-9.]+)"
)
_EVAL_RE = re.compile(r"^== eval ==.*\beval_bpc≈([0-9.]+)")


@dataclass
class Run:
    name: str
    tokenizer: str
    model_tag: str
    log: Path
    tpc: float


def _parse_last_step(log: Path) -> tuple[int, float, float]:
    # Returns: (step, tok_per_s, train_bpc)
    last = None
    for line in log.read_text().splitlines():
        m = _STEP_RE.match(line.strip())
        if not m:
            continue
        last = (int(m.group(1)), float(m.group(4)), float(m.group(3)))
    if last is None:
        raise ValueError(f"no step lines found in {log}")
    return last


def _parse_eval_bp_cs(log: Path) -> list[float]:
    out: list[float] = []
    for line in log.read_text().splitlines():
        m = _EVAL_RE.match(line.strip())
        if not m:
            continue
        out.append(float(m.group(1)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tpc_lenmax", type=float, default=0.176099, help="full-corpus TPC for Length-MAX")
    ap.add_argument("--tpc_bpe", type=float, default=0.211906, help="full-corpus TPC for BPE")
    ap.add_argument("--len_small_log", type=str, default="/home/arxiv_code/tokenizers_rust/run_llama_full_lenmax_wt103_full_ddp2_h512l8_sl512_bs64_steps10000_bf16_pack4000_evalbest_v023.log")
    ap.add_argument("--bpe_small_log", type=str, default="/home/arxiv_code/tokenizers_rust/run_llama_full_bpe_wt103_full_ddp2_h512l8_sl512_bs64_steps10000_bf16_pack4000_evalbest_v023.log")
    ap.add_argument("--len_big_log", type=str, default="/home/arxiv_code/tokenizers_rust/run_llama_big_lenmax_wt103_full_ddp2_h768l12_sl512_bs64_steps10000_bf16_pack4000_big_h768l12_tokfixed_pack4000_evalbest_steps10000_v024.log")
    ap.add_argument("--bpe_big_log", type=str, default="/home/arxiv_code/tokenizers_rust/run_llama_big_bpe_wt103_full_ddp2_h768l12_sl512_bs64_steps10000_bf16_pack4000_big_h768l12_tokfixed_pack4000_evalbest_steps10000_v024.log")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[Run] = [
        Run(
            name="lenmax_small",
            tokenizer="Length-MAX",
            model_tag="h512l8",
            log=Path(args.len_small_log),
            tpc=float(args.tpc_lenmax),
        ),
        Run(name="bpe_small", tokenizer="BPE", model_tag="h512l8", log=Path(args.bpe_small_log), tpc=float(args.tpc_bpe)),
        Run(
            name="lenmax_big",
            tokenizer="Length-MAX",
            model_tag="h768l12",
            log=Path(args.len_big_log),
            tpc=float(args.tpc_lenmax),
        ),
        Run(name="bpe_big", tokenizer="BPE", model_tag="h768l12", log=Path(args.bpe_big_log), tpc=float(args.tpc_bpe)),
    ]

    rows = []
    for r in runs:
        step, tok_s, train_bpc = _parse_last_step(r.log)
        eval_bpcs = _parse_eval_bp_cs(r.log)
        best_eval_bpc = min(eval_bpcs) if eval_bpcs else float("nan")
        last_eval_bpc = eval_bpcs[-1] if eval_bpcs else float("nan")
        chars_per_token = (1.0 / r.tpc) if r.tpc > 0 else float("nan")
        chars_per_s = (tok_s / r.tpc) if r.tpc > 0 else float("nan")
        rows.append(
            dict(
                name=r.name,
                tokenizer=r.tokenizer,
                model_tag=r.model_tag,
                log=str(r.log),
                last_step=step,
                last_tok_per_s=tok_s,
                last_train_bpc=train_bpc,
                best_eval_bpc=best_eval_bpc,
                last_eval_bpc=last_eval_bpc,
                tpc=r.tpc,
                chars_per_token=chars_per_token,
                chars_per_s=chars_per_s,
            )
        )

    out_csv = out_dir / "pareto_efficiency_quality.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "tokenizer",
                "model_tag",
                "log",
                "last_step",
                "last_tok_per_s",
                "last_train_bpc",
                "best_eval_bpc",
                "last_eval_bpc",
                "tpc",
                "chars_per_token",
                "chars_per_s",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # Plot 1: chars/token vs best_eval_bpc
    plt.figure(figsize=(7.6, 4.8), dpi=160)
    colors = {"Length-MAX": "#1f77b4", "BPE": "#d62728"}
    markers = {"h512l8": "o", "h768l12": "s"}
    for row in rows:
        tok = row["tokenizer"]
        tag = row["model_tag"]
        x = float(row["chars_per_token"])
        y = float(row["best_eval_bpc"])
        plt.scatter(x, y, c=colors.get(tok, "black"), marker=markers.get(tag, "o"), s=64, alpha=0.9)
        plt.text(x + 0.02, y + 0.005, f"{tok}-{tag}", fontsize=8, alpha=0.9)
    plt.title("Efficiency–quality tradeoff (lower eval_bpc is better; higher chars/token is better)")
    plt.xlabel("chars per token  (≈ 1/TPC, higher is better)")
    plt.ylabel("best eval_bpc  (lower is better)")
    plt.grid(True, linewidth=0.4, alpha=0.35)
    plt.tight_layout()
    out_png = out_dir / "pareto_chars_per_token_vs_eval_bpc.png"
    out_pdf = out_dir / "pareto_chars_per_token_vs_eval_bpc.pdf"
    plt.savefig(out_png)
    plt.savefig(out_pdf)

    # Plot 2: chars/s vs best_eval_bpc (using last tok/s * chars/token)
    plt.figure(figsize=(7.6, 4.8), dpi=160)
    for row in rows:
        tok = row["tokenizer"]
        tag = row["model_tag"]
        x = float(row["chars_per_s"])
        y = float(row["best_eval_bpc"])
        plt.scatter(x, y, c=colors.get(tok, "black"), marker=markers.get(tag, "o"), s=64, alpha=0.9)
        plt.text(x + 1000, y + 0.005, f"{tok}-{tag}", fontsize=8, alpha=0.9)
    plt.title("Throughput–quality tradeoff (approx chars/s from training tok/s)")
    plt.xlabel("approx chars/s (tok/s ÷ TPC, higher is better)")
    plt.ylabel("best eval_bpc (lower is better)")
    plt.grid(True, linewidth=0.4, alpha=0.35)
    plt.tight_layout()
    out_png2 = out_dir / "pareto_chars_per_s_vs_eval_bpc.png"
    out_pdf2 = out_dir / "pareto_chars_per_s_vs_eval_bpc.pdf"
    plt.savefig(out_png2)
    plt.savefig(out_pdf2)

    summary_txt = out_dir / "pareto_efficiency_quality_summary.txt"
    lines = ["== runs =="]
    for row in rows:
        lines.append(
            f"{row['name']}: best_eval_bpc={row['best_eval_bpc']:.3f} chars/token={row['chars_per_token']:.3f} "
            f"chars/s≈{row['chars_per_s']:.0f} (last tok/s={row['last_tok_per_s']:.0f})"
        )
    lines.append("")
    lines.append(f"csv={out_csv}")
    lines.append(f"plot1={out_png}")
    lines.append(f"plot2={out_png2}")
    summary_txt.write_text("\n".join(lines) + "\n")

    print("wrote", out_csv)
    print("wrote", out_png)
    print("wrote", out_png2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())







