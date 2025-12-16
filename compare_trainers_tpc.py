#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare "local trainer" vs "HF rust_trainer" on multiple corpora:

For each corpus:
  1) Train token_table.json with local `target/release/length_tokenizer`
  2) Train token_table.json with HF `hf_tokenizer_out/rust_trainer/target/release/length_tokenizer`
  3) Evaluate TPC on the same corpus with `target/release/tpc_corpus`

Print a compact table showing which trainer yields lower tokenized TPC (and token count).

Notes:
- Training may be non-deterministic due to parallelism / hash ordering; use --repeat >1 to see variance.
- This script assumes corpora are plain text (.txt), one sentence per line.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TpcResult:
    lines: int
    total_chars: int
    baseline_tokens: int
    tpc_base: float
    tokenized_tokens: int
    tpc_tokenized: float


def _run(cmd: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), file=sys.stderr)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _parse_tpc(out: str) -> TpcResult:
    # matches tpc_corpus.rs output
    m_lines = re.search(r"^lines=(\d+)$", out, re.M)
    m_chars = re.search(r"^total_chars=(\d+)$", out, re.M)
    m_base = re.search(r"^baseline_tokens=(\d+)\s+tpc_base=([0-9.]+)", out, re.M)
    m_tok = re.search(r"^tokenized_tokens=(\d+)\s+tpc_tokenized=([0-9.]+)", out, re.M)
    if not (m_lines and m_chars and m_base and m_tok):
        raise ValueError(f"failed to parse tpc output:\n{out}")
    return TpcResult(
        lines=int(m_lines.group(1)),
        total_chars=int(m_chars.group(1)),
        baseline_tokens=int(m_base.group(1)),
        tpc_base=float(m_base.group(2)),
        tokenized_tokens=int(m_tok.group(1)),
        tpc_tokenized=float(m_tok.group(2)),
    )


def _count_merges(token_table: Path) -> int:
    with token_table.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return len(obj.get("merges", []))


def main() -> int:
    repo = Path(__file__).resolve().parent
    local_bin = repo / "target" / "release" / "length_tokenizer"
    hf_bin = repo / "hf_tokenizer_out" / "rust_trainer" / "target" / "release" / "length_tokenizer"
    tpc_bin = repo / "target" / "release" / "tpc_corpus"

    ap = argparse.ArgumentParser(description="Compare trainer TPC across corpora (local vs hf rust_trainer)")
    ap.add_argument("--num-merges", type=int, default=200)
    ap.add_argument("--aim-token-num", type=int, default=20000)
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--multi-process", action="store_true")
    ap.add_argument("--max-docs", type=int, default=0, help="Limit docs read during training (0=unlimited)")
    ap.add_argument("--eval-lines", type=int, default=0, help="Limit lines during TPC eval (0=unlimited)")
    ap.add_argument("--repeat", type=int, default=1, help="Repeat per corpus to observe nondeterminism")
    ap.add_argument(
        "corpora",
        nargs="*",
        default=[
            "corpus_smoke.txt",
            "corpus_small.txt",
            "corpus_medium.txt",
            "corpus_inspect_out/head.txt",
            "corpus_inspect_out/tail.txt",
            "corpus_inspect_out/random.txt",
            "corpus_inspect_out/first_400.txt",
        ],
        help="List of corpus .txt paths (relative to repo) to test",
    )
    args = ap.parse_args()

    if not local_bin.exists():
        raise FileNotFoundError(f"local trainer binary not found: {local_bin} (run cargo build --release)")
    if not hf_bin.exists():
        raise FileNotFoundError(f"hf trainer binary not found: {hf_bin} (run cargo build --release in hf_tokenizer_out/rust_trainer)")
    if not tpc_bin.exists():
        raise FileNotFoundError(f"tpc_corpus binary not found: {tpc_bin} (run cargo build --release)")

    corpora = [Path(p) for p in args.corpora]
    corpora = [p if p.is_absolute() else (repo / p) for p in corpora]
    for p in corpora:
        if not p.exists():
            raise FileNotFoundError(f"corpus not found: {p}")

    print(
        "corpus\trep\tlocal_merges\thf_merges\tlocal_tpc\thf_tpc\tlocal_tokens\thf_tokens\twinner",
        flush=True,
    )

    for corpus in corpora:
        for r in range(args.repeat):
            with tempfile.TemporaryDirectory(prefix="lt_tpc_cmp_") as td:
                td_path = Path(td)
                local_table = td_path / "local.json"
                hf_table = td_path / "hf.json"

                # Train local
                cmd_local = [
                    str(local_bin),
                    "--corpus",
                    str(corpus),
                    "--output",
                    str(local_table),
                    "--num-merges",
                    str(args.num_merges),
                    "--aim-token-num",
                    str(args.aim_token_num),
                    "--n-max",
                    str(args.n_max),
                    "--num-workers",
                    str(args.num_workers),
                ]
                if args.multi_process:
                    cmd_local.append("--multi-process")
                if args.max_docs:
                    cmd_local += ["--max-docs", str(args.max_docs)]
                _run(cmd_local, cwd=repo)

                # Train HF
                cmd_hf = [
                    str(hf_bin),
                    "--corpus",
                    str(corpus),
                    "--output",
                    str(hf_table),
                    "--num-merges",
                    str(args.num_merges),
                    "--aim-token-num",
                    str(args.aim_token_num),
                    "--n-max",
                    str(args.n_max),
                    "--num-workers",
                    str(args.num_workers),
                ]
                if args.multi_process:
                    cmd_hf.append("--multi-process")
                if args.max_docs:
                    cmd_hf += ["--max-docs", str(args.max_docs)]
                _run(cmd_hf, cwd=repo / "hf_tokenizer_out" / "rust_trainer")

                local_merges = _count_merges(local_table)
                hf_merges = _count_merges(hf_table)

                max_lines = args.eval_lines if args.eval_lines > 0 else 10**18
                # Evaluate TPC with local `tpc_corpus` (DP tokenize)
                t_local = _run([str(tpc_bin), str(local_table), str(corpus), str(max_lines)], cwd=repo)
                t_hf = _run([str(tpc_bin), str(hf_table), str(corpus), str(max_lines)], cwd=repo)
                res_local = _parse_tpc(t_local.stdout)
                res_hf = _parse_tpc(t_hf.stdout)

                winner = "equal"
                if res_local.tokenized_tokens < res_hf.tokenized_tokens:
                    winner = "local"
                elif res_hf.tokenized_tokens < res_local.tokenized_tokens:
                    winner = "hf"

                print(
                    f"{corpus.name}\t{r+1}\t{local_merges}\t{hf_merges}\t"
                    f"{res_local.tpc_tokenized:.6f}\t{res_hf.tpc_tokenized:.6f}\t"
                    f"{res_local.tokenized_tokens}\t{res_hf.tokenized_tokens}\t{winner}",
                    flush=True,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



