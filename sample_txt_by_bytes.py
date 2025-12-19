#!/usr/bin/env python3
"""
从一个“每行一段”的大文本中，按固定随机种子采样出指定字节数（近似/严格不超过）的子集。

设计目标：
- 可复现：同 seed + 同输入文件 => 输出一致
- 接近随机：对每行生成确定性的 64-bit 随机数 u，按 u < p 初选（p 由 target_bytes/total_bytes 设定）
- 精确控体量：若初选总字节数 > target_bytes，则按 u 从大到小删除直到 <= target_bytes

注意：
- 这里的“字节数”按 UTF-8 编码计，包含每行末尾 '\\n'
- 会过滤空行（strip 后为空则跳过）
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def _u64_hash(seed: int, s: str) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(seed.to_bytes(8, "little", signed=False))
    h.update(s.encode("utf-8", errors="ignore"))
    return int.from_bytes(h.digest(), "little", signed=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", type=str, required=True)
    ap.add_argument("--out_file", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_bytes", type=int, default=100 * 1024 * 1024, help="默认 100MiB")
    args = ap.parse_args()

    in_path = Path(args.in_file)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = in_path.stat().st_size
    target = int(args.target_bytes)
    if target <= 0:
        raise SystemExit("target_bytes must be > 0")
    if total_bytes <= 0:
        raise SystemExit(f"empty input: {in_path}")

    # 初选概率（按文件总字节近似）
    p = min(1.0, target / float(total_bytes))
    thr = int(p * (2**64 - 1))

    # 先扫描一遍：按 u < thr 初选
    selected: list[tuple[int, int, str]] = []  # (u, nbytes, line)
    sel_bytes = 0
    seen = 0
    kept = 0

    with in_path.open("r", encoding="utf-8", errors="ignore") as r:
        for line in r:
            seen += 1
            s = line.strip()
            if not s:
                continue
            u = _u64_hash(args.seed, s)
            if u <= thr:
                b = len(s.encode("utf-8", errors="ignore")) + 1
                selected.append((u, b, s))
                sel_bytes += b
                kept += 1

    # 若超过目标体量，按 u 从大到小删（等价于保留最小 u 的样本，仍然可复现）
    if sel_bytes > target:
        selected.sort(key=lambda x: x[0], reverse=True)
        drop = 0
        while selected and sel_bytes > target:
            u, b, _ = selected.pop(0)
            sel_bytes -= b
            drop += 1
        # 输出时按 u 升序，保证稳定
        selected.sort(key=lambda x: x[0])
    else:
        drop = 0
        selected.sort(key=lambda x: x[0])

    with out_path.open("w", encoding="utf-8") as w:
        for _, _, s in selected:
            w.write(s + "\n")

    print("== sample_txt_by_bytes ==")
    print("in_file      =", str(in_path))
    print("out_file     =", str(out_path))
    print("seed         =", args.seed)
    print("target_bytes =", target)
    print("input_bytes  =", total_bytes)
    print("p_init       =", f"{p:.6f}")
    print("seen_lines   =", seen)
    print("kept_lines   =", kept)
    print("dropped      =", drop)
    print("out_bytes    =", out_path.stat().st_size)


if __name__ == "__main__":
    main()


