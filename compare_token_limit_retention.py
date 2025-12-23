#!/usr/bin/env python3
"""
Token-limit retention experiment: how many ORIGINAL characters fit into a fixed token window?

Why:
- A core advantage of Length-MAX is lower tokens-per-character (TPC), which implies that for
  the same context length in tokens (e.g. 512), you can fit more characters (and thus more raw text).
- This matters for long-context truncation and KV-cache/attention cost.

What we measure:
- For N packed long samples (contiguous lines from corpus), and for each token budget L:
  - Encode the full sample into token ids (no special tokens).
  - Truncate to the first L tokens.
  - Decode those L tokens back to text.
  - Compute:
    - decoded_chars: len(decoded_text)
    - lcp_chars: longest-common-prefix length between original sample and decoded text.
      (This is robust if decoding normalizes some whitespace.)

Outputs:
- CSV summary (mean/std) for each tokenizer and budget.
- A plot: covered characters vs token budget.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


def _read_lines(path: Path) -> list[str]:
    # Keep lines as-is (except newline), including empty lines.
    out: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            out.append(line.rstrip("\n"))
    if not out:
        raise ValueError(f"empty corpus: {path}")
    return out


def _pack_contiguous(lines: list[str], start: int, target_chars: int) -> str:
    if target_chars <= 0:
        return lines[start % len(lines)]
    parts: list[str] = []
    n = 0
    i = start % len(lines)
    # guard against extremely short corpora
    while n < target_chars and len(parts) < 100_000:
        s = lines[i]
        parts.append(s)
        n += len(s) + 1  # + newline
        i += 1
        if i >= len(lines):
            i = 0
    return "\n".join(parts)


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    # Python-level loop is OK for our sizes.
    while i < n and a[i] == b[i]:
        i += 1
    return i


_WS_RE = re.compile(r"\s+")


def _norm_ws(s: str) -> str:
    """
    Normalize whitespace to make decode-based comparisons more robust across tokenizers.
    Some tokenizers' decode implementations may collapse runs of whitespace/newlines.
    """
    return _WS_RE.sub(" ", s).strip()


@dataclass
class Stats:
    values: list[float]

    def mean(self) -> float:
        return sum(self.values) / max(1, len(self.values))

    def std(self) -> float:
        if len(self.values) <= 1:
            return 0.0
        m = self.mean()
        v = sum((x - m) ** 2 for x in self.values) / (len(self.values) - 1)
        return math.sqrt(v)


def _summarize(values: list[int]) -> tuple[float, float]:
    s = Stats(values=[float(x) for x in values])
    return s.mean(), s.std()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, required=True)
    ap.add_argument("--len_tok_dir", type=str, required=True)
    ap.add_argument("--bpe_tok_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--sample_chars", type=int, default=20000)
    ap.add_argument(
        "--budgets",
        type=str,
        default="128,256,512,1024",
        help="comma-separated token budgets, e.g. 128,256,512,1024",
    )
    args = ap.parse_args()

    corpus = Path(args.corpus)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    budgets = [int(x.strip()) for x in str(args.budgets).split(",") if x.strip()]
    if not budgets:
        raise SystemExit("--budgets is empty")

    print("[load] reading corpus ...")
    lines = _read_lines(corpus)
    print(f"[load] lines={len(lines)}")

    print("[load] loading tokenizers ...")
    len_tok = AutoTokenizer.from_pretrained(args.len_tok_dir, trust_remote_code=True)
    bpe_tok = AutoTokenizer.from_pretrained(args.bpe_tok_dir, use_fast=True)

    rng = random.Random(int(args.seed))
    starts = [rng.randrange(len(lines)) for _ in range(int(args.num_samples))]
    samples = [_pack_contiguous(lines, s, int(args.sample_chars)) for s in starts]

    def measure(tok, name: str):
        # Per budget metrics:
        lcp_by_budget: dict[int, list[int]] = {b: [] for b in budgets}
        norm_lcp_by_budget: dict[int, list[int]] = {b: [] for b in budgets}
        dec_by_budget: dict[int, list[int]] = {b: [] for b in budgets}
        tpc_list: list[float] = []
        for text in samples:
            # No special tokens: measure raw text coverage per token window.
            ids = tok.encode(text, add_special_tokens=False)
            if len(text) > 0:
                tpc_list.append(len(ids) / len(text))
            for b in budgets:
                prefix = ids[:b]
                dec = tok.decode(prefix, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                dec_by_budget[b].append(len(dec))
                # Raw LCP (may be sensitive to decode whitespace normalization).
                lcp_by_budget[b].append(_common_prefix_len(text, dec))
                # Whitespace-normalized LCP (more robust).
                nt = _norm_ws(text)
                nd = _norm_ws(dec)
                norm_lcp_by_budget[b].append(_common_prefix_len(nt, nd))

        tpc_mean = sum(tpc_list) / max(1, len(tpc_list))
        print(f"[{name}] approx_tpc_on_samples={tpc_mean:.6f} tok/char")
        return lcp_by_budget, norm_lcp_by_budget, dec_by_budget, tpc_mean

    print("[measure] Length-MAX ...")
    len_lcp, len_norm_lcp, len_dec, len_tpc = measure(len_tok, "Length-MAX")
    print("[measure] BPE ...")
    bpe_lcp, bpe_norm_lcp, bpe_dec, bpe_tpc = measure(bpe_tok, "BPE")

    # Write summary CSV
    out_csv = out_dir / "token_limit_retention_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "budget_tokens",
                "lenmax_decoded_mean",
                "lenmax_decoded_std",
                "bpe_decoded_mean",
                "bpe_decoded_std",
                "decoded_ratio_lenmax_over_bpe",
                "lenmax_normlcp_mean",
                "bpe_normlcp_mean",
                "normlcp_ratio_lenmax_over_bpe",
                "lenmax_chars_per_token_decoded",
                "bpe_chars_per_token_decoded",
            ]
        )
        for b in budgets:
            dec_len_m, dec_len_s = _summarize(len_dec[b])
            dec_bpe_m, dec_bpe_s = _summarize(bpe_dec[b])
            dec_ratio = (dec_len_m / dec_bpe_m) if dec_bpe_m > 0 else float("nan")
            nlcp_len_m, _ = _summarize(len_norm_lcp[b])
            nlcp_bpe_m, _ = _summarize(bpe_norm_lcp[b])
            nlcp_ratio = (nlcp_len_m / nlcp_bpe_m) if nlcp_bpe_m > 0 else float("nan")
            w.writerow(
                [
                    b,
                    f"{dec_len_m:.2f}",
                    f"{dec_len_s:.2f}",
                    f"{dec_bpe_m:.2f}",
                    f"{dec_bpe_s:.2f}",
                    f"{dec_ratio:.4f}",
                    f"{nlcp_len_m:.2f}",
                    f"{nlcp_bpe_m:.2f}",
                    f"{nlcp_ratio:.4f}",
                    f"{(dec_len_m / b):.4f}",
                    f"{(dec_bpe_m / b):.4f}",
                ]
            )

    # Plot: chars covered vs token budget (using decoded length)
    xs = budgets
    ys_len = [Stats([float(x) for x in len_dec[b]]).mean() for b in budgets]
    ys_bpe = [Stats([float(x) for x in bpe_dec[b]]).mean() for b in budgets]

    plt.figure(figsize=(8.5, 4.6), dpi=150)
    plt.plot(xs, ys_len, marker="o", linewidth=1.8, label="Length-MAX (decoded chars)")
    plt.plot(xs, ys_bpe, marker="o", linewidth=1.8, label="BPE (decoded chars)")
    plt.title("Token-limit retention: covered original chars vs token budget")
    plt.xlabel("token budget (tokens)")
    plt.ylabel("decoded chars (mean over samples)")
    plt.grid(True, linewidth=0.4, alpha=0.35)
    plt.legend(frameon=False, fontsize=9, loc="best")
    plt.tight_layout()
    out_png = out_dir / "token_limit_retention.png"
    out_pdf = out_dir / "token_limit_retention.pdf"
    plt.savefig(out_png)
    plt.savefig(out_pdf)

    # Plot: ratio
    ratios = [ys_len[i] / ys_bpe[i] for i in range(len(xs))]
    plt.figure(figsize=(8.5, 3.8), dpi=150)
    plt.plot(xs, ratios, marker="o", linewidth=1.8)
    plt.axhline(1.0, color="black", linewidth=1, alpha=0.5)
    plt.title("Retention ratio: Length-MAX / BPE (higher is better)")
    plt.xlabel("token budget (tokens)")
    plt.ylabel("ratio of covered chars")
    plt.grid(True, linewidth=0.4, alpha=0.35)
    plt.tight_layout()
    out_png2 = out_dir / "token_limit_retention_ratio.png"
    out_pdf2 = out_dir / "token_limit_retention_ratio.pdf"
    plt.savefig(out_png2)
    plt.savefig(out_pdf2)

    # Write a tiny text summary for convenience
    summary_txt = out_dir / "token_limit_retention_summary.txt"
    summary_txt.write_text(
        "\n".join(
            [
                f"corpus={corpus}",
                f"samples={len(samples)} sample_chars={int(args.sample_chars)} seed={int(args.seed)}",
                f"budgets={budgets}",
                f"approx_tpc_lenmax_samples={len_tpc:.6f}",
                f"approx_tpc_bpe_samples={bpe_tpc:.6f}",
                f"csv={out_csv}",
                f"plot_chars={out_png}",
                f"plot_ratio={out_png2}",
                "",
            ]
        )
        + "\n"
    )

    print("[done] wrote:", out_csv)
    print("[done] wrote:", out_png)
    print("[done] wrote:", out_png2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


