#!/usr/bin/env python3
"""
Plot a 3-way comparison figure (BPE vs SuperBPE vs Length-MAX) on WT103-valid.

This is a lightweight plotting helper for rebuttal/paper figures.
Numbers are taken from eval_extra_metrics_lenmax_vs_superbpe.py with:
  device=cuda precision=bf16 seq_len=512 pack_chars=4000 pack_mode=contiguous
  force_eos=True batches=32 batch_size=16 warmup_batches=8

Outputs:
  - extra_metrics_wt103_valid_threeway_v047.png
  - extra_metrics_wt103_valid_threeway_v047.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Row:
    name: str
    eval_loss: float
    bpc: float
    ppl_char: float
    bpb: float
    eval_tpc: float
    model_tok_s: float
    model_char_s: float
    unk_rate: float


def main() -> None:
    # WT103-valid, warmup-corrected.
    rows = [
        Row(
            name="BPE",
            eval_loss=3.4549,
            bpc=1.0676,
            ppl_char=2.096,
            bpb=1.0657,
            eval_tpc=0.214193,
            model_tok_s=201_894,
            model_char_s=942_578,
            unk_rate=0.0,
        ),
        Row(
            name="SuperBPE",
            eval_loss=4.4750,
            bpc=1.1023,
            ppl_char=2.147,
            bpb=1.1003,
            eval_tpc=0.170733,
            model_tok_s=202_238,
            model_char_s=1_184_528,
            unk_rate=0.0,
        ),
        Row(
            name="Length-MAX v047",
            eval_loss=4.2918,
            bpc=1.0662,
            ppl_char=2.094,
            bpb=1.0643,
            eval_tpc=0.172205,
            model_tok_s=201_227,
            model_char_s=1_168_531,
            unk_rate=1.07e-4,
        ),
    ]

    # Local import so the script fails fast if matplotlib is missing.
    import matplotlib.pyplot as plt

    names = [r.name for r in rows]

    def _bar(ax, values, *, title: str, fmt: str, better: str) -> None:
        bars = ax.bar(names, values, color=["#4C78A8", "#F58518", "#54A24B"])
        ax.set_title(f"{title} ({better} better)")
        ax.grid(axis="y", alpha=0.25)
        # Value labels
        for b, v in zip(bars, values):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                b.get_height(),
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=0,
            )

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), constrained_layout=True)
    axes = axes.reshape(-1)

    _bar(axes[0], [r.bpc for r in rows], title="bpc", fmt="{:.4f}", better="↓")
    _bar(axes[1], [r.eval_tpc for r in rows], title="eval_tpc (tok/char)", fmt="{:.6f}", better="↓")
    _bar(axes[2], [r.model_tok_s for r in rows], title="model-only tok/s", fmt="{:,.0f}", better="↑")
    _bar(axes[3], [r.model_char_s for r in rows], title="model-only char/s", fmt="{:,.0f}", better="↑")

    fig.suptitle(
        "WT103-valid: BPE vs SuperBPE vs Length-MAX (model-only throughput)\n"
        "eval: seq_len=512, bf16, cuda, pack_chars=4000, batches=32, batch_size=16, warmup=8",
        fontsize=12,
    )

    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / "extra_metrics_wt103_valid_threeway_v047.png"
    out_pdf = out_dir / "extra_metrics_wt103_valid_threeway_v047.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print("wrote", out_png)
    print("wrote", out_pdf)

    # Also produce a single Pareto-style bubble plot:
    # x = chars/token (higher is better compression), y = bpc (lower is better quality),
    # bubble size = model-only char/s (higher is better throughput).
    import matplotlib.pyplot as plt

    xs = [1.0 / r.eval_tpc for r in rows]  # chars/token
    ys = [r.bpc for r in rows]
    ss = [max(1.0, r.model_char_s) for r in rows]

    # Normalize bubble sizes (area).
    s_min = min(ss)
    s_max = max(ss)
    sizes = []
    for s in ss:
        t = (s - s_min) / (s_max - s_min + 1e-9)
        sizes.append(250.0 + 750.0 * t)

    fig2, ax = plt.subplots(1, 1, figsize=(7.2, 5.2), constrained_layout=True)
    ax.grid(alpha=0.25)
    ax.set_title("WT103-valid Pareto view (compression vs quality vs model throughput)")
    ax.set_xlabel("chars/token (higher = more compression)")
    ax.set_ylabel("bpc (lower = better)")

    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for r, x, y, s, c in zip(rows, xs, ys, sizes, colors):
        ax.scatter([x], [y], s=s, color=c, alpha=0.75, edgecolor="black", linewidth=0.6, zorder=3)
        ax.text(x, y, f"  {r.name}", fontsize=10, va="center", ha="left")

    # Slight padding
    ax.set_xlim(min(xs) * 0.98, max(xs) * 1.02)
    ax.set_ylim(min(ys) - 0.01, max(ys) + 0.02)

    foot = (
        "bubble area ∝ model-only char/s; "
        "eval: seq_len=512 bf16 cuda, pack_chars=4000, batches=32, bs=16, warmup=8"
    )
    fig2.text(0.01, 0.01, foot, fontsize=9, alpha=0.8)

    out_png2 = out_dir / "extra_metrics_wt103_valid_pareto_threeway_v047.png"
    out_pdf2 = out_dir / "extra_metrics_wt103_valid_pareto_threeway_v047.pdf"
    fig2.savefig(out_png2, dpi=200)
    fig2.savefig(out_pdf2)
    print("wrote", out_png2)
    print("wrote", out_pdf2)


if __name__ == "__main__":
    main()


