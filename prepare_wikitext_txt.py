#!/usr/bin/env python3
"""
把本地的 WikiText parquet（来自 datasets/Salesforce/wikitext 仓库快照）导出成“每行一句/一段”的 txt，
用于训练 tokenizer / 训练小模型。

输入目录示例：
  /home/arxiv_code/datasets/wikitext_repo/wikitext-2-raw-v1

输出：
  out_dir/train.txt / validation.txt / test.txt（空行会过滤）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def _iter_text_from_parquet(p: Path):
    table = pq.read_table(p, columns=["text"])
    col = table.column(0)
    # pyarrow StringArray -> python list[str|None]
    for x in col.to_pylist():
        if x is None:
            continue
        s = str(x).strip()
        if not s:
            continue
        yield s


def _write_split(split_name: str, parquet_files: list[Path], out_path: Path, max_lines: int) -> int:
    n = 0
    with out_path.open("w", encoding="utf-8") as w:
        for pf in parquet_files:
            for s in _iter_text_from_parquet(pf):
                w.write(s.replace("\n", " ").strip() + "\n")
                n += 1
                if max_lines > 0 and n >= max_lines:
                    return n
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikitext_dir", type=str, required=True, help="wikitext-*-raw-v1 目录（包含 train/validation/test parquet）")
    ap.add_argument("--out_dir", type=str, required=True, help="输出目录")
    ap.add_argument("--max_train", type=int, default=0, help="最多导出多少行训练集（0=不限制）")
    ap.add_argument("--max_valid", type=int, default=0, help="最多导出多少行验证集（0=不限制）")
    ap.add_argument("--max_test", type=int, default=0, help="最多导出多少行测试集（0=不限制）")
    args = ap.parse_args()

    src = Path(args.wikitext_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_files = sorted(src.glob("train-*.parquet"))
    valid_files = sorted(src.glob("validation-*.parquet"))
    test_files = sorted(src.glob("test-*.parquet"))
    if not train_files or not valid_files or not test_files:
        raise SystemExit(f"missing parquet splits under {src}")

    n_train = _write_split("train", train_files, out / "train.txt", args.max_train)
    n_valid = _write_split("validation", valid_files, out / "validation.txt", args.max_valid)
    n_test = _write_split("test", test_files, out / "test.txt", args.max_test)

    print("== done ==")
    print("src =", str(src))
    print("out =", str(out))
    print("train_lines =", n_train)
    print("valid_lines =", n_valid)
    print("test_lines  =", n_test)


if __name__ == "__main__":
    main()


