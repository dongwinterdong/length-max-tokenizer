#!/usr/bin/env python3
"""
Plot downstream + long-context results produced by eval_downstream_threeway.py.

Inputs:
  --csv /path/to/downstream_threeway_results.csv

Outputs (in same dir as CSV unless --out_dir is provided):
  - downstream_acc_bar.png/.pdf
  - longmc_curves.png/.pdf
  - passkey_curves.png/.pdf
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Row:
    model: str
    task: str
    split: str
    n: int
    acc: float
    trunc_rate: float
    avg_prompt_tokens_full: float
    avg_prompt_tokens_used: float
    avg_prompt_chars: float
    needle_present_rate: float
    needle_pos_frac: float | None
    context_chars: int | None
    cont_chars: int | None
    total_chars: int | None
    num_choices: int | None


def _to_int(x: str) -> int | None:
    x = (x or "").strip()
    if not x:
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _read_rows(csv_path: Path) -> list[Row]:
    rows: list[Row] = []
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(
                Row(
                    model=str(d.get("model", "")),
                    task=str(d.get("task", "")),
                    split=str(d.get("split", "")),
                    n=int(float(d.get("n", 0) or 0)),
                    acc=_to_float(d.get("acc", "")),
                    trunc_rate=_to_float(d.get("trunc_rate", "")),
                    avg_prompt_tokens_full=_to_float(d.get("avg_prompt_tokens_full", "")),
                    avg_prompt_tokens_used=_to_float(d.get("avg_prompt_tokens_used", "")),
                    avg_prompt_chars=_to_float(d.get("avg_prompt_chars", "")),
                    needle_present_rate=_to_float(d.get("needle_present_rate", "")),
                    needle_pos_frac=_to_float(d.get("needle_pos_frac", "")) if str(d.get("needle_pos_frac", "")).strip() else None,
                    context_chars=_to_int(d.get("context_chars", "")),
                    cont_chars=_to_int(d.get("cont_chars", "")),
                    total_chars=_to_int(d.get("total_chars", "")),
                    num_choices=_to_int(d.get("num_choices", "")),
                )
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="")
    args = ap.parse_args()

    csv_path = Path(str(args.csv))
    out_dir = Path(str(args.out_dir)) if str(args.out_dir).strip() else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(csv_path)
    models = sorted({r.model for r in rows if r.model})

    # ---- 1) Standard downstream accuracy bar plot ----
    std_tasks = ["lambada_mc", "hellaswag", "piqa", "winogrande"]
    std_rows = [r for r in rows if r.task in std_tasks]

    # Build (task -> model -> acc)
    acc_map: dict[str, dict[str, float]] = {t: {} for t in std_tasks}
    for r in std_rows:
        acc_map[r.task][r.model] = r.acc

    plt.figure(figsize=(9.5, 4.6), dpi=160)
    x0 = list(range(len(std_tasks)))
    width = 0.22
    colors = {"BPE": "#4C78A8", "SuperBPE": "#F58518", "Length-MAX": "#54A24B"}
    for mi, m in enumerate(models):
        xs = [x + (mi - (len(models) - 1) / 2.0) * width for x in x0]
        ys = [acc_map[t].get(m, float("nan")) for t in std_tasks]
        plt.bar(xs, ys, width=width, label=m, color=colors.get(m, None))
        for x, y in zip(xs, ys):
            if y == y:  # not nan
                plt.text(x, y + 0.005, f"{y:.3f}", ha="center", va="bottom", fontsize=8)

    plt.xticks(x0, std_tasks)
    plt.ylim(0.0, 1.0)
    plt.ylabel("accuracy")
    plt.title("Downstream accuracy (WT103-trained 58.5M Llama)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False, ncol=3, loc="upper right")
    plt.tight_layout()
    out1 = out_dir / "downstream_acc_bar"
    plt.savefig(str(out1) + ".png")
    plt.savefig(str(out1) + ".pdf")

    # ---- 2) WT103 longMC curves ----
    longmc = [r for r in rows if r.task == "wt103_longmc" and r.context_chars is not None]
    if longmc:
        by_model: dict[str, list[Row]] = defaultdict(list)
        for r in longmc:
            by_model[r.model].append(r)
        for m in by_model:
            by_model[m].sort(key=lambda x: int(x.context_chars or 0))

        fig, ax1 = plt.subplots(1, 1, figsize=(9.5, 4.8), dpi=160)
        ax2 = ax1.twinx()
        for m, rs in by_model.items():
            xs = [float(r.context_chars or 0) for r in rs]
            ys = [float(r.acc) for r in rs]
            ts = [float(r.trunc_rate) for r in rs]
            ax1.plot(xs, ys, marker="o", linewidth=2.0, label=f"{m} acc", color=colors.get(m, None))
            ax2.plot(xs, ts, marker="x", linewidth=1.5, linestyle="--", label=f"{m} trunc", color=colors.get(m, None))
        ax1.set_xlabel("context_chars")
        ax1.set_ylabel("accuracy (higher better)")
        ax2.set_ylabel("trunc_rate (higher worse)")
        ax1.set_title("WT103-longMC: accuracy and prompt truncation vs context length")
        ax1.grid(alpha=0.25)

        # Merge legends
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, frameon=False, ncol=3, loc="lower left")
        fig.tight_layout()
        out2 = out_dir / "longmc_curves"
        fig.savefig(str(out2) + ".png")
        fig.savefig(str(out2) + ".pdf")

    # ---- 3) passkey curves ----
    pk = [r for r in rows if r.task == "passkey_mc" and r.total_chars is not None]
    if pk:
        by_model: dict[str, list[Row]] = defaultdict(list)
        for r in pk:
            by_model[r.model].append(r)
        for m in by_model:
            by_model[m].sort(key=lambda x: int(x.total_chars or 0))

        fig, ax1 = plt.subplots(1, 1, figsize=(9.5, 4.8), dpi=160)
        ax2 = ax1.twinx()
        for m, rs in by_model.items():
            xs = [float(r.total_chars or 0) for r in rs]
            ys = [float(r.acc) for r in rs]
            ts = [float(r.trunc_rate) for r in rs]
            ax1.plot(xs, ys, marker="o", linewidth=2.0, label=f"{m} acc", color=colors.get(m, None))
            ax2.plot(xs, ts, marker="x", linewidth=1.5, linestyle="--", label=f"{m} trunc", color=colors.get(m, None))
        ax1.set_xlabel("total_chars")
        ax1.set_ylabel("accuracy (higher better)")
        ax2.set_ylabel("trunc_rate (higher worse)")
        ax1.set_title("Passkey-MC: accuracy and prompt truncation vs total length")
        ax1.grid(alpha=0.25)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, frameon=False, ncol=3, loc="upper right")
        fig.tight_layout()
        out3 = out_dir / "passkey_curves"
        fig.savefig(str(out3) + ".png")
        fig.savefig(str(out3) + ".pdf")

    # ---- 4) WT103-filler long-context-sensitive tasks ----
    def _plot_len_sweep(task_name: str, *, x_key: str, title: str, out_stem: str) -> None:
        sel = [r for r in rows if r.task == task_name]
        # Select rows with a length field.
        if x_key == "total_chars":
            sel = [r for r in sel if r.total_chars is not None]
            xget = lambda rr: int(rr.total_chars or 0)
            xlabel = "total_chars"
        else:
            return
        if not sel:
            return

        by_model: dict[str, list[Row]] = defaultdict(list)
        for rr in sel:
            by_model[rr.model].append(rr)
        for m in by_model:
            by_model[m].sort(key=lambda x: xget(x))

        fig, ax1 = plt.subplots(1, 1, figsize=(9.5, 4.8), dpi=160)
        ax2 = ax1.twinx()
        for m, rs in by_model.items():
            xs = [float(xget(rr)) for rr in rs]
            ys = [float(rr.acc) for rr in rs]
            ts = [float(rr.trunc_rate) for rr in rs]
            ax1.plot(xs, ys, marker="o", linewidth=2.0, label=f"{m} acc", color=colors.get(m, None))
            ax2.plot(xs, ts, marker="x", linewidth=1.5, linestyle="--", label=f"{m} trunc", color=colors.get(m, None))
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("accuracy (higher better)")
        ax2.set_ylabel("trunc_rate (higher worse)")
        ax1.set_title(title)
        ax1.grid(alpha=0.25)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, frameon=False, ncol=3, loc="upper right")
        fig.tight_layout()
        outp = out_dir / out_stem
        fig.savefig(str(outp) + ".png")
        fig.savefig(str(outp) + ".pdf")

    _plot_len_sweep(
        "wt103_passkey_mc",
        x_key="total_chars",
        title="WT103-filler Passkey-MC: accuracy and truncation vs total length",
        out_stem="wt103_passkey_mc_curves",
    )
    _plot_len_sweep(
        "kv_retrieval_mc",
        x_key="total_chars",
        title="WT103-filler KV-Retrieval-MC: accuracy and truncation vs total length",
        out_stem="kv_retrieval_mc_curves",
    )

    # ---- 5) Needle-in-haystack style curves (acc + retention) ----
    def _plot_len_sweep_with_retention(*, task_name: str, out_stem: str, title: str) -> None:
        sel = [r for r in rows if r.task == task_name and r.total_chars is not None]
        if not sel:
            return
        pos_vals = sorted({r.needle_pos_frac for r in sel if r.needle_pos_frac is not None})
        pos_note = f" (needle_pos={pos_vals[0]:.2f})" if len(pos_vals) == 1 else ""

        by_model: dict[str, list[Row]] = defaultdict(list)
        for rr in sel:
            by_model[rr.model].append(rr)
        for m in by_model:
            by_model[m].sort(key=lambda x: int(x.total_chars or 0))

        fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.8), dpi=160)
        for m, rs in by_model.items():
            xs = [float(rr.total_chars or 0) for rr in rs]
            accs = [float(rr.acc) for rr in rs]
            rets = [float(rr.needle_present_rate) for rr in rs]
            ax.plot(xs, accs, marker="o", linewidth=2.0, label=f"{m} acc", color=colors.get(m, None))
            ax.plot(xs, rets, marker="x", linewidth=1.5, linestyle="--", label=f"{m} retained", color=colors.get(m, None))

        ax.set_xlabel("total_chars")
        ax.set_ylabel("rate (0..1)")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title + pos_note)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=3, loc="upper right")
        fig.tight_layout()
        outp = out_dir / out_stem
        fig.savefig(str(outp) + ".png")
        fig.savefig(str(outp) + ".pdf")

    _plot_len_sweep_with_retention(
        task_name="wt103_passkey_needle_mc",
        out_stem="wt103_passkey_needle_mc_curves",
        title="WT103 needle Passkey-MC: accuracy vs retention",
    )
    _plot_len_sweep_with_retention(
        task_name="kv_needle_mc",
        out_stem="kv_needle_mc_curves",
        title="WT103 needle KV-Retrieval-MC: accuracy vs retention",
    )

    print("wrote plots to", out_dir)


if __name__ == "__main__":
    main()


