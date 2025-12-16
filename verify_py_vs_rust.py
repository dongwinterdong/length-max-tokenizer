#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end consistency check:

1) Train vocab (token_table.json) with local Rust CLI and (optionally) HF rust_trainer
2) Export HF vocab.json from both token tables (should match)
3) Tokenize a corpus with:
   - Rust reference: `tokenize_table_hash` (loads token_table.json and tokenizes with DP)
   - Python reference: HF remote-code tokenizer `hf_tokenizer_out/tokenization_length_tokenizer.py`
     (will use Rust wheel if importable; otherwise Python DP fallback)
   Compare deterministic FNV64 hash + total token count.

This script is designed to run without extra Python deps (no datasets/pyarrow).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), file=sys.stderr)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parent


def _install_transformers_stub() -> None:
    """
    Allow importing hf_tokenizer_out/tokenization_length_tokenizer.py on machines without transformers installed.

    We only need a minimal PreTrainedTokenizer that forwards tokenize() -> _tokenize().
    """
    import types

    m = types.ModuleType("transformers")

    class PreTrainedTokenizer:  # noqa: D401
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def tokenize(self, text: str):
            return self._tokenize(text)  # type: ignore[attr-defined]

    m.PreTrainedTokenizer = PreTrainedTokenizer  # type: ignore[attr-defined]
    sys.modules["transformers"] = m


def _fnv1a64_update(h: int, data: bytes) -> int:
    FNV_OFFSET = 14695981039346656037
    FNV_PRIME = 1099511628211
    if h == 0:
        h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def _python_tokenize_hash(
    tokenizer_module_path: Path,
    vocab_json: Path,
    corpus_txt: Path,
    max_lines: int,
    force_disable_rust: bool,
    wheel_path: Optional[Path],
) -> Dict[str, Any]:
    # Make wheel importable even without pip: add .whl to sys.path (zipimport works)
    if wheel_path is not None:
        sys.path.insert(0, str(wheel_path))

    if force_disable_rust:
        os.environ["LENGTH_TOKENIZER_DISABLE_RUST"] = "1"
    else:
        os.environ.pop("LENGTH_TOKENIZER_DISABLE_RUST", None)

    import importlib.util

    # If transformers isn't installed, inject a minimal stub so remote-code can import.
    try:
        import transformers  # noqa: F401
    except Exception:
        _install_transformers_stub()

    spec = importlib.util.spec_from_file_location("lt_remote_code", str(tokenizer_module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import tokenizer module: {tokenizer_module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    tok = mod.LengthTokenizer(str(vocab_json))  # type: ignore[attr-defined]
    rust_enabled = getattr(tok, "_rust", None) is not None

    h = 0
    total_tokens = 0
    lines = 0
    with corpus_txt.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            lines += 1
            toks = tok.tokenize(line.rstrip("\n"))
            total_tokens += len(toks)
            for t in toks:
                h = _fnv1a64_update(h, t.encode("utf-8"))
                h = _fnv1a64_update(h, b"\0")
            h = _fnv1a64_update(h, b"\n")
            if max_lines > 0 and lines >= max_lines:
                break

    return {
        "lines": lines,
        "total_tokens": total_tokens,
        "fnv64": f"0x{h:016x}",
        "rust_enabled": rust_enabled,
    }


def _ensure_release_built(repo: Path) -> None:
    bin1 = repo / "target" / "release" / "length_tokenizer"
    bin2 = repo / "target" / "release" / "export_hf_tokenizer"
    bin3 = repo / "target" / "release" / "tokenize_table_hash"
    if bin1.exists() and bin2.exists() and bin3.exists():
        return
    _run(["cargo", "build", "--release"], cwd=repo)


def _ensure_hf_trainer_built(repo: Path) -> Path:
    rust_trainer = repo / "hf_tokenizer_out" / "rust_trainer"
    bin_path = rust_trainer / "target" / "release" / "length_tokenizer"
    if bin_path.exists():
        return bin_path
    _run(["cargo", "build", "--release"], cwd=rust_trainer)
    if not bin_path.exists():
        raise RuntimeError(f"hf rust_trainer binary not found after build: {bin_path}")
    return bin_path


def _export_vocab(repo: Path, token_table: Path, out_dir: Path) -> Tuple[int, Path]:
    export_bin = repo / "target" / "release" / "export_hf_tokenizer"
    cp = _run([str(export_bin), str(token_table), str(out_dir)], cwd=repo)
    # export prints merges count to stderr: "[export] merges=..."
    m = re.search(r"\[export\] merges=(\d+)", cp.stderr)
    merges = int(m.group(1)) if m else -1
    vocab = out_dir / "vocab.json"
    if not vocab.exists():
        raise RuntimeError(f"export did not produce vocab.json: {vocab}")
    return merges, vocab


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    repo = _find_repo_root()

    ap = argparse.ArgumentParser(description="Verify Python wheel/remote-code vs local Rust (train + tokenize) consistency")
    ap.add_argument("--corpus", default=str(repo / "corpus_smoke.txt"), help="Training corpus (txt or parquet path)")
    ap.add_argument("--tokenize-corpus", default="", help="Corpus for tokenization consistency (txt). Defaults to --corpus if it is txt.")
    ap.add_argument("--max-lines", type=int, default=1000, help="Max lines to tokenize for hashing (0=unlimited)")
    ap.add_argument("--num-merges", type=int, default=50)
    ap.add_argument("--aim-token-num", type=int, default=20000)
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--multi-process", action="store_true")
    ap.add_argument("--max-docs", type=int, default=0, help="Limit docs read during training (0=unlimited)")
    ap.add_argument("--wheel", default="", help="Optional path to a built .whl to add to sys.path for import")
    ap.add_argument("--disable-rust", action="store_true", help="Force Python remote-code to use Python DP fallback (no wheel)")
    ap.add_argument("--skip-hf-trainer", action="store_true", help="Skip comparing against hf_tokenizer_out/rust_trainer")
    args = ap.parse_args()

    corpus = Path(args.corpus).resolve()
    if not corpus.exists():
        raise FileNotFoundError(f"corpus not found: {corpus}")

    tokenize_corpus = Path(args.tokenize_corpus).resolve() if args.tokenize_corpus else corpus
    if tokenize_corpus.suffix.lower() != ".txt":
        raise ValueError(
            "tokenize-corpus must be a .txt file (Python side doesn't read parquet). "
            "Provide a small txt sample for tokenization check."
        )

    wheel_path = Path(args.wheel).resolve() if args.wheel else None
    if wheel_path is not None and not wheel_path.exists():
        raise FileNotFoundError(f"wheel not found: {wheel_path}")

    _ensure_release_built(repo)
    local_bin = repo / "target" / "release" / "length_tokenizer"
    tokhash_bin = repo / "target" / "release" / "tokenize_table_hash"

    hf_bin: Optional[Path] = None
    if not args.skip_hf_trainer:
        hf_bin = _ensure_hf_trainer_built(repo)

    with tempfile.TemporaryDirectory(prefix="lt_verify_") as td:
        td_path = Path(td)

        local_table = td_path / "token_table_local.json"
        hf_table = td_path / "token_table_hf.json"

        # 1) Train with local Rust CLI
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
            cmd_local.extend(["--max-docs", str(args.max_docs)])
        _run(cmd_local, cwd=repo)

        # 2) Train with HF rust_trainer (optional)
        if hf_bin is not None:
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
                cmd_hf.extend(["--max-docs", str(args.max_docs)])
            _run(cmd_hf, cwd=repo / "hf_tokenizer_out" / "rust_trainer")

        # 3) Export vocab.json from both (compare)
        local_out = td_path / "hf_local"
        hf_out = td_path / "hf_hf"
        local_out.mkdir(parents=True, exist_ok=True)
        hf_out.mkdir(parents=True, exist_ok=True)

        local_merges, local_vocab = _export_vocab(repo, local_table, local_out)
        local_vocab_obj = _load_json(local_vocab)

        if hf_bin is not None:
            hf_merges, hf_vocab = _export_vocab(repo, hf_table, hf_out)
            hf_vocab_obj = _load_json(hf_vocab)
            if local_vocab_obj != hf_vocab_obj:
                raise SystemExit(
                    f"ERROR: vocab.json mismatch between local vs hf rust_trainer\\n"
                    f"local_merges={local_merges} hf_merges={hf_merges}\\n"
                    f"local_vocab_size={len(local_vocab_obj)} hf_vocab_size={len(hf_vocab_obj)}\\n"
                )
            print(
                f"OK: train vocab match (local vs hf rust_trainer), merges={local_merges}, vocab={len(local_vocab_obj)}",
                file=sys.stderr,
            )

        # 4) Tokenize hash (Rust reference)
        rust_out = _run(
            [
                str(tokhash_bin),
                "--table",
                str(local_table),
                "--corpus",
                str(tokenize_corpus),
                "--max-lines",
                str(args.max_lines),
                "--n-max",
                str(args.n_max),
            ],
            cwd=repo,
        )
        rust_stats = json.loads(rust_out.stdout.strip())

        # 5) Tokenize hash (Python remote-code)
        tokenizer_module = repo / "hf_tokenizer_out" / "tokenization_length_tokenizer.py"
        py_stats = _python_tokenize_hash(
            tokenizer_module_path=tokenizer_module,
            vocab_json=local_vocab,
            corpus_txt=tokenize_corpus,
            max_lines=args.max_lines,
            force_disable_rust=args.disable_rust,
            wheel_path=wheel_path,
        )

        if (
            int(py_stats["lines"]) != int(rust_stats["lines"])
            or int(py_stats["total_tokens"]) != int(rust_stats["total_tokens"])
            or str(py_stats["fnv64"]) != str(rust_stats["fnv64"])
        ):
            raise SystemExit(
                "ERROR: tokenization mismatch (python vs rust)\\n"
                f"rust={rust_stats}\\n"
                f"python={py_stats}\\n"
            )

        print(
            f"OK: tokenize match python vs rust "
            f"(rust_enabled={py_stats['rust_enabled']}) lines={py_stats['lines']} "
            f"total_tokens={py_stats['total_tokens']} fnv64={py_stats['fnv64']}"
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())


