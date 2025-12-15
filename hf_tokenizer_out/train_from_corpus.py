#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 HuggingFace tokenizer 仓库内“一键重训并导出”。

这个脚本会：
1) 使用仓库自带的 rust_trainer（你当前的训练方法）从指定语料训练出 token_table.json
2) 调用 rust_trainer 内置的 export_hf_tokenizer 导出 HuggingFace 可分发目录
3) 复制 tokenization_length_tokenizer.py（remote code）到输出目录

为什么这里**没有 `import` 你的库**？
- 你的训练实现目前是 **Rust crate**（`cargo` 项目），并没有被编译/安装成 Python 可 `import` 的包（例如 PyO3/maturin 生成的 wheel）。
- HuggingFace Hub 的 tokenizer repo 本质是“文件分发”，不会自动为用户安装你的 Rust crate。
- 当前环境也没有 `pip`/`tokenizers`/`transformers`，所以这里选择最稳妥的方式：**直接调用 cargo 编译运行**，
  既保证训练逻辑与 Rust 代码完全一致，也避免 Python 依赖问题。

要求：
- 本机已安装 Rust 工具链（cargo）

用法示例：
    python3 train_from_corpus.py \
      --corpus /path/to/new_corpus.txt \
      --out ./hf_tokenizer_new \
      --num-merges 50000 \
      --n-max 6 \
      --num-workers 0 \
      --multi-process
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd is not None else None)


def main() -> int:
    ap = argparse.ArgumentParser(description="重训 LengthTokenizer 并导出 HuggingFace tokenizer 包（remote code）")
    ap.add_argument("--corpus", required=True, help="语料文件（每行一句）")
    ap.add_argument("--out", required=True, help="输出目录（生成可上传到 HF Hub 的 tokenizer 目录）")
    ap.add_argument("--num-merges", type=int, default=50000)
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--aim-token-num", type=int, default=15000)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--multi-process", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    rust_dir = here / "rust_trainer"
    corpus = Path(args.corpus).resolve()
    out_dir = Path(args.out).resolve()

    if not corpus.exists():
        raise FileNotFoundError(f"corpus not found: {corpus}")
    if not rust_dir.exists():
        raise FileNotFoundError(f"rust_trainer not found: {rust_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 临时 token_table.json
    with tempfile.TemporaryDirectory(prefix="lt_hf_train_") as td:
        td_path = Path(td)
        token_table = td_path / "token_table.json"

        # 1) 训练
        cmd_train = [
            "cargo",
            "run",
            "--release",
            "--manifest-path",
            str(rust_dir / "Cargo.toml"),
            "--",
            "--corpus",
            str(corpus),
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
        ]
        if args.multi_process:
            cmd_train.append("--multi-process")

        _run(cmd_train)

        # 2) 导出 HF 目录
        cmd_export = [
            "cargo",
            "run",
            "--release",
            "--manifest-path",
            str(rust_dir / "Cargo.toml"),
            "--bin",
            "export_hf_tokenizer",
            "--",
            str(token_table),
            str(out_dir),
        ]
        _run(cmd_export)

        # 3) 复制 remote code tokenizer 实现
        shutil.copy2(here / "tokenization_length_tokenizer.py", out_dir / "tokenization_length_tokenizer.py")

    print(f"done: {out_dir}")
    print("你可以把这个目录直接上传到 HuggingFace Hub（需要 trust_remote_code=True）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


