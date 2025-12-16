#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train vocab with the Rust trainer (length_tokenizer), then tokenize text with the Python wheel.

Why this script:
- Training is implemented in Rust CLI (fast, consistent with your trainer).
- Tokenization in Python uses the Rust wheel `length-tokenizer-rs` (DpTokenizer) when available.
- If the wheel isn't installed, it falls back to the HF remote-code Python DP implementation
  (no transformers dependency required; a minimal stub is injected).

Outputs:
- token_table.json (trainer output)
- vocab.json + HF config files (via export_hf_tokenizer)
- optional token ids file for a text corpus

Examples:
1) Train on a txt corpus and tokenize the same file:
   python3 train_vocab_and_tokenize.py \
     --train-corpus ./corpus_small.txt \
     --out ./out_small \
     --num-merges 5000 --aim-token-num 20000 \
     --tokenize-file ./corpus_small.txt --out-ids ./out_small/ids.txt

2) Train on FineWeb-Edu parquet (directory) with sampling:
   python3 train_vocab_and_tokenize.py \
     --train-corpus /data/fineweb-edu/sample/10BT \
     --corpus-format parquet --text-column text --max-docs 2000000 \
     --out ./out_fw \
     --num-merges 50000 --aim-token-num 20000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


def _run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd is not None else None)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_release_bins(repo: Path) -> None:
    # We need length_tokenizer + export_hf_tokenizer.
    bin_train = repo / "target" / "release" / "length_tokenizer"
    bin_export = repo / "target" / "release" / "export_hf_tokenizer"
    if bin_train.exists() and bin_export.exists():
        return
    _run(["cargo", "build", "--release"], cwd=repo)


def _install_transformers_stub() -> None:
    """Allow importing hf_tokenizer_out/tokenization_length_tokenizer.py without transformers installed."""
    import types

    m = types.ModuleType("transformers")

    class PreTrainedTokenizer:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def tokenize(self, text: str):
            return self._tokenize(text)  # type: ignore[attr-defined]

    m.PreTrainedTokenizer = PreTrainedTokenizer  # type: ignore[attr-defined]
    sys.modules["transformers"] = m


class _TokenizerAdapter:
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]


class _WheelTokenizer(_TokenizerAdapter):
    def __init__(self, vocab_json: Path, unk_token: str) -> None:
        from length_tokenizer_rs import DpTokenizer  # type: ignore

        self._tok = DpTokenizer(str(vocab_json), unk_token)

    def encode(self, text: str) -> List[int]:
        return [int(x) for x in self._tok.encode(text)]

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [[int(x) for x in row] for row in self._tok.encode_batch(texts)]


class _RemoteCodeTokenizer(_TokenizerAdapter):
    def __init__(self, vocab_json: Path, disable_rust: bool) -> None:
        # Force remote-code to stay in Python DP mode (no wheel assumed).
        if disable_rust:
            os.environ["LENGTH_TOKENIZER_DISABLE_RUST"] = "1"
        else:
            os.environ.pop("LENGTH_TOKENIZER_DISABLE_RUST", None)

        # Ensure transformers import doesn't fail.
        try:
            import transformers  # noqa: F401
        except Exception:
            _install_transformers_stub()

        mod_path = _repo_root() / "hf_tokenizer_out" / "tokenization_length_tokenizer.py"
        spec = importlib.util.spec_from_file_location("tokenization_length_tokenizer", str(mod_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import remote-code module: {mod_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[call-arg]

        self._tok = mod.LengthTokenizer(str(vocab_json))  # type: ignore[attr-defined]
        # token -> id mapping
        self._vocab = dict(getattr(self._tok, "vocab"))
        self._unk_id = int(getattr(self._tok, "_unk_id", self._vocab.get("<unk>", 0)))

    def encode(self, text: str) -> List[int]:
        toks = self._tok.tokenize(text)  # type: ignore[attr-defined]
        return [int(self._vocab.get(t, self._unk_id)) for t in toks]


def _iter_lines(path: Path, max_lines: int) -> Iterable[str]:
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.rstrip("\n").rstrip("\r")
            if not s.strip():
                continue
            yield s
            n += 1
            if max_lines > 0 and n >= max_lines:
                break


def main() -> int:
    repo = _repo_root()
    ap = argparse.ArgumentParser(description="Train vocab (Rust) and tokenize (Python wheel) for LengthTokenizer")

    # Training
    ap.add_argument("--train-corpus", required=True, help="Training corpus path (txt or parquet file/dir)")
    ap.add_argument("--out", required=True, help="Output directory (token_table.json + vocab.json will be created)")
    ap.add_argument("--corpus-format", choices=["auto", "txt", "parquet"], default="auto")
    ap.add_argument("--text-column", default="text", help="Parquet text column name (FineWeb-Edu default: text)")
    ap.add_argument("--max-docs", type=int, default=0, help="Limit docs read during training (0=unlimited)")
    ap.add_argument("--parquet-batch-size", type=int, default=8192)
    ap.add_argument("--parquet-recursive", action="store_true")

    ap.add_argument("--num-merges", type=int, default=50000)
    ap.add_argument("--aim-token-num", type=int, default=20000)
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--multi-process", action="store_true")

    # Tokenization
    ap.add_argument("--tokenize-text", default="", help="If set, tokenize this text and print ids")
    ap.add_argument("--tokenize-file", default="", help="If set, tokenize this txt file (one sentence per line)")
    ap.add_argument("--out-ids", default="", help="Output ids file path (each line: space-separated ids)")
    ap.add_argument("--max-lines", type=int, default=0, help="Max lines to tokenize from tokenize-file (0=all)")
    ap.add_argument("--batch-size", type=int, default=256, help="Batch size for encode_batch()")

    ap.add_argument("--unk-token", default="<unk>")
    ap.add_argument("--force-python", action="store_true", help="Force Python DP (ignore wheel even if installed)")

    args = ap.parse_args()

    train_corpus = Path(args.train_corpus).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    token_table = out_dir / "token_table.json"
    vocab_json = out_dir / "vocab.json"

    _ensure_release_bins(repo)
    bin_train = repo / "target" / "release" / "length_tokenizer"
    bin_export = repo / "target" / "release" / "export_hf_tokenizer"

    # 1) Train token_table.json
    cmd_train = [
        str(bin_train),
        "--corpus",
        str(train_corpus),
        "--output",
        str(token_table),
        "--num-merges",
        str(args.num_merges),
        "--aim-token-num",
        str(args.aim_token_num),
        "--n-max",
        str(args.n_max),
        "--num-workers",
        str(args.num_workers),
        "--corpus-format",
        str(args.corpus_format),
        "--text-column",
        str(args.text_column),
        "--parquet-batch-size",
        str(args.parquet_batch_size),
    ]
    if args.parquet_recursive:
        cmd_train.append("--parquet-recursive")
    if args.multi_process:
        cmd_train.append("--multi-process")
    if args.max_docs:
        cmd_train.extend(["--max-docs", str(args.max_docs)])

    _run(cmd_train, cwd=repo)

    # 2) Export vocab.json (and HF config files) into out_dir
    _run([str(bin_export), str(token_table), str(out_dir)], cwd=repo)
    if not vocab_json.exists():
        raise RuntimeError(f"export did not create vocab.json at {vocab_json}")

    # 3) Load tokenizer for Python encoding
    tok: _TokenizerAdapter
    if args.force_python:
        tok = _RemoteCodeTokenizer(vocab_json=vocab_json, disable_rust=True)
        print("[tokenize] using python DP (forced)", file=sys.stderr)
    else:
        try:
            tok = _WheelTokenizer(vocab_json=vocab_json, unk_token=args.unk_token)
            print("[tokenize] using Rust wheel (DpTokenizer)", file=sys.stderr)
        except Exception as e:
            print(f"[tokenize] wheel not available ({e}); fallback to python DP", file=sys.stderr)
            tok = _RemoteCodeTokenizer(vocab_json=vocab_json, disable_rust=True)

    # 4) Tokenize
    if args.tokenize_text:
        ids = tok.encode(args.tokenize_text)
        print("ids:", ids)

    if args.tokenize_file:
        in_path = Path(args.tokenize_file).resolve()
        if in_path.suffix.lower() != ".txt":
            raise ValueError("tokenize-file must be a .txt file (one sentence per line)")

        out_ids = Path(args.out_ids).resolve() if args.out_ids else (out_dir / "ids.txt")
        out_ids.parent.mkdir(parents=True, exist_ok=True)

        total_chars = 0
        total_tokens = 0
        lines = 0

        buf: List[str] = []
        with out_ids.open("w", encoding="utf-8", newline="\n") as w:
            for s in _iter_lines(in_path, args.max_lines):
                buf.append(s)
                if len(buf) >= max(1, args.batch_size):
                    batch = tok.encode_batch(buf)
                    for text, ids in zip(buf, batch):
                        w.write(" ".join(str(i) for i in ids))
                        w.write("\n")
                        total_chars += len(text)
                        total_tokens += len(ids)
                        lines += 1
                    buf.clear()

            if buf:
                batch = tok.encode_batch(buf)
                for text, ids in zip(buf, batch):
                    w.write(" ".join(str(i) for i in ids))
                    w.write("\n")
                    total_chars += len(text)
                    total_tokens += len(ids)
                    lines += 1

        tpc = (total_tokens / total_chars) if total_chars else 0.0
        print(
            f"[tokenize] done lines={lines} total_chars={total_chars} total_tokens={total_tokens} tpc={tpc:.6f} ids_out={out_ids}",
            file=sys.stderr,
        )

    print(f"done: token_table={token_table} vocab={vocab_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




