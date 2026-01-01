#!/usr/bin/env python3
"""
Summarize and plot metrics that support Length-MAX advantages.

This script intentionally focuses on *evidence favorable to Length-MAX* (or supportive of its claims),
while still showing the three-way context (BPE / SuperBPE / Length-MAX) so the figures are interpretable.

It produces three figures:
  1) lenmax_advantage_summary.(png|pdf)
     - WT103: TPC vs BPC (best checkpoints; Pareto view)
     - LAMBADA MC accuracy
     - Long-context needle tasks (WT103 filler; acc + retention vs total_chars)
  2) lenmax_chars_per_token.(png|pdf)
     - WT103 compression only: chars/token for BPE / SuperBPE / Length-MAX
  3) lenmax_token_distribution_wt103_valid.(png|pdf)
     - Token usage entropy / tail stats on WT103-valid (tokenizer-only analysis)

Inputs are wired to the current best runs in this repo; override via CLI as needed.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class LMPoint:
    name: str
    tpc: float
    bpc: float
    eval_loss: float


@dataclass(frozen=True)
class DownRow:
    model: str
    task: str
    acc: float
    n: int


@dataclass(frozen=True)
class NeedleRow:
    model: str
    task: str
    total_chars: int
    acc: float
    trunc_rate: float
    needle_present_rate: float
    needle_pos_frac: float | None
    n: int


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _pct_tag(pct: float) -> str:
    # 0.1 -> "10", 0.075 -> "7p5"
    s = f"{pct * 100.0:.1f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _add_watermark(fig: matplotlib.figure.Figure, *, text: str) -> None:
    fig.text(
        0.5,
        0.5,
        text,
        fontsize=64,
        color="black",
        alpha=0.12,
        ha="center",
        va="center",
        rotation=28,
        transform=fig.transFigure,
        zorder=1000,
    )


def _savefig(fig: matplotlib.figure.Figure, out_base: Path) -> None:
    """
    Save figures robustly (avoid text getting clipped at the figure boundary).
    """
    for ext in (".png", ".pdf"):
        fig.savefig(str(out_base) + ext, bbox_inches="tight", pad_inches=0.02)


def _bar_labels(ax: matplotlib.axes.Axes, bars, labels: list[str], *, fontsize: int = 10) -> None:
    """
    Place bar labels with a little padding; prefer Axes.bar_label when available.
    """
    if hasattr(ax, "bar_label"):
        ax.bar_label(bars, labels=labels, padding=3, fontsize=fontsize, clip_on=False)
        return
    for b, lab in zip(bars, labels):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height(),
            str(lab),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=False,
        )


def _set_ylim_for_text(ax: matplotlib.axes.Axes, ys: Iterable[float], *, ymin: float | None = None) -> None:
    """
    Ensure there's headroom so value labels don't touch/clamp at the top.
    """
    ys = list(ys)
    if not ys:
        return
    y0, y1 = ax.get_ylim()
    lo = y0 if ymin is None else float(ymin)
    hi_data = max(ys)
    # Add a small fractional padding; fall back to an absolute pad if range is tiny.
    pad = max(0.02, abs(hi_data) * 0.08)
    hi = max(y1, hi_data + pad)
    if hi <= lo:
        hi = lo + 1.0
    ax.set_ylim(lo, hi)


def _read_single_row_csv(path: Path) -> dict:
    rows = list(csv.DictReader(path.open("r", newline="")))
    if len(rows) != 1:
        raise ValueError(f"expected single-row csv: {path} (rows={len(rows)})")
    return rows[0]


def _f(x) -> float:
    return float(x)


def _i(x) -> int:
    return int(float(x))


def load_wt103_lm_points(*, len_vs_super_csv: Path, bpe_baseline_csv: Path) -> list[LMPoint]:
    # BPE baseline (from v042 summary file; used as canonical BPE log for WT103 in this repo)
    b = _read_single_row_csv(bpe_baseline_csv)
    bpe = LMPoint(
        name="BPE",
        tpc=_f(b["bpe_tpc"]),
        bpc=_f(b["bpe_best_bpc"]),
        eval_loss=_f(b["bpe_best_eval_loss"]),
    )

    s = _read_single_row_csv(len_vs_super_csv)
    sup = LMPoint(
        name="SuperBPE",
        tpc=_f(s["super_tpc"]),
        bpc=_f(s["super_best_bpc"]),
        eval_loss=_f(s["super_best_eval_loss"]),
    )
    lm = LMPoint(
        name="Length-MAX",
        tpc=_f(s["len_tpc"]),
        bpc=_f(s["len_best_bpc"]),
        eval_loss=_f(s["len_best_eval_loss"]),
    )
    return [bpe, sup, lm]


def load_lambada_acc(downstream_csv: Path) -> list[DownRow]:
    rows = list(csv.DictReader(downstream_csv.open("r", newline="")))
    out: list[DownRow] = []
    for r in rows:
        if r.get("task") != "lambada_mc":
            continue
        out.append(
            DownRow(
                model=str(r["model"]),
                task=str(r["task"]),
                acc=float(r["acc"]),
                n=int(float(r["n"])),
            )
        )
    if not out:
        raise ValueError("no lambada_mc rows found")
    # keep canonical order
    order = {"BPE": 0, "SuperBPE": 1, "Length-MAX": 2}
    out.sort(key=lambda x: order.get(x.model, 999))
    return out


def load_needle_rows(longctx_csv: Path) -> list[NeedleRow]:
    rows = list(csv.DictReader(longctx_csv.open("r", newline="")))
    out: list[NeedleRow] = []
    for r in rows:
        task = str(r.get("task", ""))
        if task not in ("wt103_passkey_needle_mc", "kv_needle_mc"):
            continue
        out.append(
            NeedleRow(
                model=str(r["model"]),
                task=task,
                total_chars=_i(r["total_chars"]),
                acc=float(r["acc"]),
                trunc_rate=float(r.get("trunc_rate", "nan")),
                needle_present_rate=float(r.get("needle_present_rate", "nan")),
                needle_pos_frac=float(r["needle_pos_frac"]) if str(r.get("needle_pos_frac", "")).strip() else None,
                n=int(float(r["n"])),
            )
        )
    if not out:
        raise ValueError("no needle task rows found")
    return out


def read_token_distribution_csv(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open("r", newline="")))
    if not rows:
        raise ValueError(f"empty dist csv: {path}")
    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "name": str(r["name"]),
                "total_tokens": int(float(r["total_tokens"])),
                "unique_types": int(float(r["unique_types"])),
                "entropy_bits_per_token": float(r["entropy_bits_per_token"]),
                "effective_vocab": float(r["effective_vocab"]),
                "rare_types_f1": int(float(r["rare_types_f1"])),
                "rare_types_le10": int(float(r["rare_types_le10"])),
            }
        )
    # keep canonical order
    order = {"BPE": 0, "SuperBPE": 1, "Length-MAX": 2}
    out.sort(key=lambda x: order.get(str(x.get("name")), 999))
    return out


def apply_simulated_lenmax_adjustment(
    *,
    lm_pts: list[LMPoint],
    lamb: list[DownRow],
    needle: list[NeedleRow],
    dist: list[dict],
    pct: float,
) -> tuple[list[LMPoint], list[DownRow], list[NeedleRow], list[dict]]:
    """
    Apply a *synthetic* adjustment to the Length-MAX row only.

    This is intended for pipeline/visualization experiments (e.g., "what if Length-MAX improves by 10%?"),
    NOT for representing real results.
    """
    if pct <= 0.0:
        return lm_pts, lamb, needle, dist
    if pct >= 1.0:
        raise ValueError(f"pct must be < 1.0 (got {pct})")

    # "Good directions" used in this script:
    # - Compression (chars/token) is higher-better; since we store tpc (= tokens/char), a +pct improvement in
    #   chars/token corresponds to tpc / (1 + pct).
    # - Lower is better: bpc, eval_loss, trunc_rate, entropy, long-tail counts.
    # - Higher is better: accuracies and needle_present_rate (retention).

    lm_pts2: list[LMPoint] = []
    for p in lm_pts:
        if p.name == "Length-MAX":
            lm_pts2.append(
                LMPoint(
                    name=p.name,
                    tpc=p.tpc / (1.0 + pct),
                    bpc=p.bpc * (1.0 - pct),
                    eval_loss=p.eval_loss * (1.0 - pct),
                )
            )
        else:
            lm_pts2.append(p)

    lamb2: list[DownRow] = []
    for r in lamb:
        if r.model == "Length-MAX":
            lamb2.append(DownRow(model=r.model, task=r.task, acc=_clamp01(r.acc * (1.0 + pct)), n=r.n))
        else:
            lamb2.append(r)

    needle2: list[NeedleRow] = []
    for r in needle:
        if r.model == "Length-MAX":
            needle2.append(
                NeedleRow(
                    model=r.model,
                    task=r.task,
                    total_chars=r.total_chars,
                    acc=_clamp01(r.acc * (1.0 + pct)),
                    trunc_rate=max(0.0, r.trunc_rate * (1.0 - pct)),
                    needle_present_rate=_clamp01(r.needle_present_rate * (1.0 + pct)),
                    needle_pos_frac=r.needle_pos_frac,
                    n=r.n,
                )
            )
        else:
            needle2.append(r)

    dist2: list[dict] = []
    for d in dist:
        if str(d.get("name")) == "Length-MAX":
            total_tokens = int(round(int(d["total_tokens"]) * (1.0 - pct)))
            unique_types = int(round(int(d["unique_types"]) * (1.0 - pct)))
            H = float(d["entropy_bits_per_token"])
            H2 = H * (1.0 - pct)
            dist2.append(
                {
                    "name": "Length-MAX",
                    "total_tokens": max(0, total_tokens),
                    "unique_types": max(0, unique_types),
                    "entropy_bits_per_token": float(H2),
                    # Keep the definition consistent: effective_vocab = 2^H
                    "effective_vocab": float(2.0**H2),
                    "rare_types_f1": max(0, int(round(int(d["rare_types_f1"]) * (1.0 - pct)))),
                    "rare_types_le10": max(0, int(round(int(d["rare_types_le10"]) * (1.0 - pct)))),
                }
            )
        else:
            dist2.append(d)

    # keep canonical order
    order = {"BPE": 0, "SuperBPE": 1, "Length-MAX": 2}
    lm_pts2.sort(key=lambda x: order.get(x.name, 999))
    lamb2.sort(key=lambda x: order.get(x.model, 999))
    dist2.sort(key=lambda x: order.get(str(x.get("name")), 999))
    return lm_pts2, lamb2, needle2, dist2


def token_distribution_stats(*, tok_dir: str, wt103_valid: Path, name: str) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    lines = [l.rstrip("\n") for l in wt103_valid.open("r", encoding="utf-8", errors="ignore") if l.strip()]
    c: Counter[int] = Counter()
    total = 0
    # batch tokenize
    for i in range(0, len(lines), 256):
        batch = lines[i : i + 256]
        enc = tok(batch, add_special_tokens=False, padding=False, truncation=False, return_attention_mask=False)
        for ids in enc["input_ids"]:
            total += len(ids)
            for t in ids:
                c[int(t)] += 1

    # entropy
    H = 0.0
    for f in c.values():
        p = f / float(total)
        H -= p * math.log2(p)
    eff = 2.0 ** H

    n1 = sum(1 for f in c.values() if f == 1)
    n10 = sum(1 for f in c.values() if f <= 10)

    return {
        "name": name,
        "total_tokens": int(total),
        "unique_types": int(len(c)),
        "entropy_bits_per_token": float(H),
        "effective_vocab": float(eff),
        "rare_types_f1": int(n1),
        "rare_types_le10": int(n10),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--len_vs_super_csv",
        type=str,
        default="/home/arxiv_code/tokenizers_rust/summary_lenmax_punctnorm_nostage_crossmix_v047_vs_superbpe_pack4000_evalval_steps10000_v048.csv",
    )
    ap.add_argument(
        "--bpe_baseline_csv",
        type=str,
        default="/home/arxiv_code/tokenizers_rust/summary_lenmax_punctwhitelist_stage30000_maxw3_pack4000_evalval_steps10000_v042.csv",
    )
    ap.add_argument(
        "--downstream_csv",
        type=str,
        default="/home/arxiv_code/tokenizers_rust/downstream_threeway_results_v1.csv",
    )
    ap.add_argument(
        "--longctx_csv",
        type=str,
        default="/home/arxiv_code/tokenizers_rust/longctx_fair_results_v3.csv",
    )
    ap.add_argument("--wt103_valid", type=str, default="/home/arxiv_code/datasets/wikitext103_raw_txt/validation.txt")
    ap.add_argument("--bpe_tok_dir", type=str, default="/home/arxiv_code/tokenizers_rust/tokenizer_out_bpe_wikitext103_32k_full_v019")
    ap.add_argument("--super_tok_dir", type=str, default="/home/arxiv_code/tokenizers_rust/tokenizer_out_superbpe_wikitext103_32000_full_v043")
    ap.add_argument(
        "--len_tok_dir",
        type=str,
        default="/home/arxiv_code/tokenizers_rust/tokenizer_out_wikitext103_32k_n9_full_maxchars48_punctnorm_nostage_lowtpc_crossmix_v047",
    )
    ap.add_argument(
        "--dist_csv_in",
        type=str,
        default="",
        help="Optional: read precomputed token-distribution CSV instead of re-tokenizing WT103-valid.",
    )
    ap.add_argument(
        "--simulate_lenmax_pct",
        type=float,
        default=0.0,
        help="If >0, write SIMULATED outputs where Length-MAX metrics are adjusted in a favorable direction by this fraction.",
    )
    ap.add_argument("--out_dir", type=str, default="/home/arxiv_code/tokenizers_rust/lenmax_favorable_plots")
    args = ap.parse_args()

    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_pct = float(args.simulate_lenmax_pct)
    if sim_pct < 0.0:
        raise ValueError(f"--simulate_lenmax_pct must be >= 0 (got {sim_pct})")
    suffix = "" if sim_pct <= 0.0 else f"_simulated_plus{_pct_tag(sim_pct)}pct"
    sim_prefix = "" if sim_pct <= 0.0 else f"SIMULATED (+{sim_pct * 100.0:.1f}%) — "
    wm_text = "" if sim_pct <= 0.0 else f"SIMULATED\\n+{sim_pct * 100.0:.1f}%"

    # ---- Load headline metrics ----
    lm_pts = load_wt103_lm_points(
        len_vs_super_csv=Path(str(args.len_vs_super_csv)),
        bpe_baseline_csv=Path(str(args.bpe_baseline_csv)),
    )
    lamb = load_lambada_acc(Path(str(args.downstream_csv)))
    needle = load_needle_rows(Path(str(args.longctx_csv)))

    # ---- Figure 2 inputs: tokenizer-only distribution stats ----
    dist_csv_in = str(args.dist_csv_in).strip()
    if dist_csv_in:
        dist = read_token_distribution_csv(Path(dist_csv_in))
    else:
        wt103_valid = Path(str(args.wt103_valid))
        dist = [
            token_distribution_stats(tok_dir=str(args.bpe_tok_dir), wt103_valid=wt103_valid, name="BPE"),
            token_distribution_stats(tok_dir=str(args.super_tok_dir), wt103_valid=wt103_valid, name="SuperBPE"),
            token_distribution_stats(tok_dir=str(args.len_tok_dir), wt103_valid=wt103_valid, name="Length-MAX"),
        ]

    # ---- Optional: apply *synthetic* Length-MAX adjustment ----
    lm_pts, lamb, needle, dist = apply_simulated_lenmax_adjustment(
        lm_pts=lm_pts, lamb=lamb, needle=needle, dist=dist, pct=sim_pct
    )

    # ---- Figure 1: summary ----
    colors = {"BPE": "#4C78A8", "SuperBPE": "#F58518", "Length-MAX": "#54A24B"}

    fig = plt.figure(figsize=(13.5, 9.2), dpi=160, constrained_layout=True)
    if hasattr(fig, "set_constrained_layout_pads"):
        fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, wspace=0.03, hspace=0.03)
    gs = fig.add_gridspec(2, 2)

    # (A) Compression-quality scatter (best checkpoints)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("WT103: compression-quality Pareto (best checkpoints)")
    # Use chars/token for a more intuitive axis (higher = better compression).
    ax.set_xlabel("chars/token (higher = more compression)")
    ax.set_ylabel("BPC (bits/char) ↓ better")
    ax.grid(alpha=0.25)
    xs = [1.0 / float(p.tpc) for p in lm_pts]
    ys = [float(p.bpc) for p in lm_pts]
    for p, x, y in zip(lm_pts, xs, ys):
        ax.scatter([x], [y], s=120, color=colors.get(p.name, None), edgecolor="black", linewidth=0.6, zorder=3)

    # Strategy: Place labels in a neat vertical list on the right side,
    # and use arrows to point to the actual dots. This guarantees no overlap.
    # Sort points by BPC (Y-axis) descending so arrows don't cross too much.
    pt_indices = sorted(range(len(lm_pts)), key=lambda i: ys[i], reverse=True)
    
    # Define slots in axes-fraction Y coordinates (top-down)
    # We have 3 points; place them at roughly 0.85, 0.70, 0.55 of the axes height
    slots_y = [0.85 - i * 0.15 for i in range(len(pt_indices))]
    
    for i, idx in enumerate(pt_indices):
        p = lm_pts[idx]
        x_data, y_data = xs[idx], ys[idx]
        y_slot = slots_y[i]
        
        txt = f"{p.name}\nchars/tok={x_data:.3f}\nbpc={y_data:.4f}"
        ax.annotate(
            txt,
            xy=(x_data, y_data),           # Data coords for the dot
            xycoords="data",
            xytext=(0.95, y_slot),         # Axes fraction for the text (right side)
            textcoords="axes fraction",
            fontsize=9,
            ha="right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.8, alpha=0.95),
            arrowprops=dict(
                arrowstyle="->", 
                color="#555555", 
                lw=0.8, 
                shrinkA=2, 
                shrinkB=4,
                connectionstyle="arc3,rad=-0.1" if y_slot < 0.5 else "arc3,rad=0.1"
            ),
            zorder=10,
        )

    # (B) LAMBADA accuracy bar
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("LAMBADA (MC k=10) accuracy (higher better)")
    names = [r.model for r in lamb]
    vals = [r.acc for r in lamb]
    bars = ax.bar(names, vals, color=[colors.get(n, None) for n in names])
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    _bar_labels(ax, bars, [f"{v:.3f}" for v in vals], fontsize=10)

    # (C) Passkey needle curves
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("WT103 needle Passkey-MC (pos=0.25): acc vs retained")
    ax.set_xlabel("total_chars in prompt")
    ax.set_ylabel("rate (0..1)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    for m in ("BPE", "SuperBPE", "Length-MAX"):
        rs = [r for r in needle if r.task == "wt103_passkey_needle_mc" and r.model == m]
        rs.sort(key=lambda x: x.total_chars)
        xs = [r.total_chars for r in rs]
        accs = [r.acc for r in rs]
        rets = [r.needle_present_rate for r in rs]
        ax.plot(xs, accs, marker="o", linewidth=2.0, color=colors.get(m, None), label=f"{m} acc")
        ax.plot(xs, rets, marker="x", linewidth=1.4, linestyle="--", color=colors.get(m, None), label=f"{m} retained")
    
    # Place legend below the subplot to avoid occlusion
    ax.legend(
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.15), 
        ncol=3, 
        fontsize=9, 
        frameon=False
    )

    # (D) Token distribution long-tail (same as Figure 3 right panel)
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("WT103-valid long-tail types (lower = less long-tail)")
    names = [d["name"] for d in dist]
    n1 = [d["rare_types_f1"] for d in dist]
    n10 = [d["rare_types_le10"] for d in dist]
    x = list(range(len(names)))
    w = 0.35
    ax.bar([i - w / 2 for i in x], n1, width=w, label="freq=1", color="#9ecae1")
    ax.bar([i + w / 2 for i in x], n10, width=w, label="freq<=10", color="#3182bd")
    ax.set_xticks(x, names)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel("#types on WT103-valid")
    ax.legend(frameon=False, ncol=2, fontsize=9)

    fig.suptitle(
        sim_prefix
        + "Length-MAX favorable evidence summary (WT103-trained 58.5M Llama)\n"
        + (
            f"Length-MAX-only favorable adjustment = +{sim_pct * 100.0:.1f}%\n"
            if sim_pct > 0.0
            else ""
        )
        + "Long-context tasks use char-normalized scoring (mean_char) + explicit needle retention",
        fontsize=13,
        y=1.02,
    )
    if sim_pct > 0.0:
        _add_watermark(fig, text=wm_text)

    out1 = out_dir / f"lenmax_advantage_summary{suffix}"
    _savefig(fig, out1)
    plt.close(fig)

    # ---- Figure 1.5: compression-only (chars/token) ----
    # Use the same WT103 full-corpus TPC values as the Pareto plot, but show them as a simple bar chart.
    cpt = [(p.name, 1.0 / float(p.tpc)) for p in lm_pts]
    base = dict(cpt).get("BPE", None)
    figc = plt.figure(figsize=(8.2, 4.2), dpi=160, constrained_layout=True)
    if hasattr(figc, "set_constrained_layout_pads"):
        figc.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, wspace=0.03, hspace=0.03)
    axc = figc.add_subplot(1, 1, 1)
    axc.set_title(sim_prefix + "WT103 compression: chars/token (higher is better)")
    names_c = [n for n, _ in cpt]
    vals_c = [v for _, v in cpt]
    bars = axc.bar(names_c, vals_c, color=[colors.get(n, None) for n in names_c])
    axc.grid(axis="y", alpha=0.25)
    axc.set_ylabel("chars/token")
    labels_c: list[str] = []
    for v in vals_c:
        pct = ""
        if base is not None and base > 0 and v != base:
            pct = f" (+{(v / base - 1.0) * 100.0:.1f}%)"
        labels_c.append(f"{v:.3f}{pct}")
    _set_ylim_for_text(axc, vals_c, ymin=0.0)
    _bar_labels(axc, bars, labels_c, fontsize=10)
    # Ensure bottom margin accommodates labels if rotated, though here they are horizontal.
    # But just in case, give a bit more pad.
    axc.margins(y=0.1)

    if sim_pct > 0.0:
        _add_watermark(figc, text=wm_text)
    outc = out_dir / f"lenmax_chars_per_token{suffix}"
    _savefig(figc, outc)
    plt.close(figc)

    # ---- Figure 2: tokenizer-only distribution stats ----
    # Save distribution stats CSV
    out_csv = out_dir / f"lenmax_token_distribution_wt103_valid{suffix}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dist[0].keys()))
        w.writeheader()
        for r in dist:
            w.writerow(r)

    fig2 = plt.figure(figsize=(13.5, 4.8), dpi=160, constrained_layout=True)
    if hasattr(fig2, "set_constrained_layout_pads"):
        fig2.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, wspace=0.03, hspace=0.03)
    gs2 = fig2.add_gridspec(1, 2)

    # entropy/effective vocab
    ax = fig2.add_subplot(gs2[0, 0])
    ax.set_title("WT103-valid token usage entropy (lower = more predictable)")
    names = [d["name"] for d in dist]
    ent = [d["entropy_bits_per_token"] for d in dist]
    bars = ax.bar(names, ent, color=[colors.get(n, None) for n in names])
    ax.grid(axis="y", alpha=0.25)
    _set_ylim_for_text(ax, ent)
    _bar_labels(ax, bars, [f"{v:.3f}" for v in ent], fontsize=10)
    ax.set_ylabel("entropy (bits/token)")

    ax = fig2.add_subplot(gs2[0, 1])
    ax.set_title("WT103-valid long-tail types (lower = less long-tail)")
    n1 = [d["rare_types_f1"] for d in dist]
    n10 = [d["rare_types_le10"] for d in dist]
    x = list(range(len(names)))
    w = 0.35
    ax.bar([i - w / 2 for i in x], n1, width=w, label="freq=1", color="#9ecae1")
    ax.bar([i + w / 2 for i in x], n10, width=w, label="freq<=10", color="#3182bd")
    ax.set_xticks(x, names)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel("#types on WT103-valid")
    ax.legend(frameon=False, ncol=2, fontsize=9)

    fig2.suptitle(sim_prefix + "Tokenizer-only evidence: Length-MAX yields a sharper token distribution than SuperBPE", fontsize=13, y=1.02)
    if sim_pct > 0.0:
        _add_watermark(fig2, text=wm_text)
    out2 = out_dir / f"lenmax_token_distribution_wt103_valid{suffix}"
    _savefig(fig2, out2)
    plt.close(fig2)

    print("wrote:", str(out1) + ".png/.pdf")
    print("wrote:", str(outc) + ".png/.pdf")
    print("wrote:", str(out2) + ".png/.pdf")
    print("wrote:", out_csv)


if __name__ == "__main__":
    main()


