#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_corpus.py

用途：
- 从超大语料（逐行一句/一段）里“提取可读的小样本”，用于快速人工辨别语料类型/来源/质量。
- 不依赖第三方库（仅 Python 标准库），适合当前环境（无 pip）。

默认会输出：
- head（前 N 行）
- tail（后 N 行，使用反向块读取，不扫描全文件）
- positions（按字节偏移在不同位置各取一小段）
- random（按随机字节偏移抽样若干行，不扫描全文件）
- first_k（提取前 K 行，便于直接打开阅读）
- report.json / report.md（简单统计特征：多语言/URL/Email/脚本比例等）

注意：
- 这不是严格的“语言识别”，只是基于 Unicode 范围的粗略脚本统计，用来判断是否“混合语种/网页抓取”。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from typing import Iterable, List, Tuple, Dict, Optional


DEFAULT_PATH = "/home/arxiv_code/tokenizers_rust/corpus_py.txt"


def _safe_decode(b: bytes) -> str:
    # 用 replace 保证任何脏字节都能解码出来（会产生 �）
    return b.decode("utf-8", errors="replace")


def _strip_line(s: str) -> str:
    return s.rstrip("\r\n")


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            if u == "B":
                return f"{int(f)} {u}"
            return f"{f:.2f} {u}"
        f /= 1024
    return f"{n} B"


def fast_wc_lines(path: str) -> Optional[int]:
    """
    尝试用 wc -l 快速得到行数（不保证所有环境都有 wc）。
    """
    try:
        out = subprocess.check_output(["wc", "-l", path], text=True)
        # 输出格式："{lines} {path}"
        return int(out.strip().split()[0])
    except Exception:
        return None


def read_head(path: str, n: int) -> List[str]:
    out: List[str] = []
    with open(path, "rb") as f:
        for _ in range(n):
            b = f.readline()
            if not b:
                break
            out.append(_strip_line(_safe_decode(b)))
    return out


def read_tail(path: str, n: int, block_size: int = 1024 * 1024) -> List[str]:
    """
    类似 tail -n 的实现：从文件末尾反向读取块，直到收集到足够换行。
    """
    if n <= 0:
        return []

    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        pos = end
        buf = b""
        # splitlines() 对最后是否有换行都能处理
        while pos > 0 and buf.count(b"\n") <= n:
            read_size = min(block_size, pos)
            pos -= read_size
            f.seek(pos, os.SEEK_SET)
            chunk = f.read(read_size)
            buf = chunk + buf

        lines = buf.splitlines()
        tail = lines[-n:] if len(lines) >= n else lines
        return [_strip_line(_safe_decode(x)) for x in tail]


def read_at_offset(path: str, offset: int, max_lines: int) -> Tuple[int, List[str]]:
    """
    在指定字节偏移处定位到“下一行开头”，然后读取 max_lines 行。
    返回：(实际起始偏移, 行列表)
    """
    out: List[str] = []
    with open(path, "rb") as f:
        f.seek(max(0, offset), os.SEEK_SET)
        if offset > 0:
            _ = f.readline()  # 丢弃半行
        start_pos = f.tell()
        for _ in range(max_lines):
            b = f.readline()
            if not b:
                break
            out.append(_strip_line(_safe_decode(b)))
    return start_pos, out


def sample_lines_by_random_offsets(
    path: str,
    k: int,
    seed: int,
    max_tries: int = 10_000,
) -> List[str]:
    """
    通过随机字节偏移抽样行（不扫描全文件，速度快）。
    这不是严格均匀的“按行”抽样，但足够用于人工判断语料类型。
    """
    if k <= 0:
        return []

    size = os.path.getsize(path)
    rng = random.Random(seed)
    samples: List[str] = []
    tries = 0

    with open(path, "rb") as f:
        while len(samples) < k and tries < max_tries:
            tries += 1
            off = rng.randrange(0, max(1, size))
            f.seek(off, os.SEEK_SET)
            if off > 0:
                _ = f.readline()  # 丢弃半行
            b = f.readline()
            if not b:
                continue
            s = _strip_line(_safe_decode(b))
            if not s.strip():
                continue
            samples.append(s)

    return samples


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def script_counter(text: str) -> Counter:
    """
    非严格语言识别：按 Unicode 范围做脚本/字符集粗统计。
    """
    c = Counter()
    for ch in text:
        o = ord(ch)
        if o < 128:
            c["ascii"] += 1
        # CJK Unified Ideographs
        elif 0x4E00 <= o <= 0x9FFF:
            c["cjk"] += 1
        # Hiragana / Katakana
        elif 0x3040 <= o <= 0x30FF:
            c["jp_kana"] += 1
        # Hangul
        elif 0xAC00 <= o <= 0xD7AF:
            c["kr_hangul"] += 1
        # Cyrillic
        elif 0x0400 <= o <= 0x04FF:
            c["cyrillic"] += 1
        # Arabic
        elif 0x0600 <= o <= 0x06FF:
            c["arabic"] += 1
        # Devanagari
        elif 0x0900 <= o <= 0x097F:
            c["devanagari"] += 1
        else:
            c["other_non_ascii"] += 1
    return c


def line_features(line: str) -> Dict[str, object]:
    s = line
    has_url = ("http://" in s) or ("https://" in s) or ("www." in s)
    has_email = EMAIL_RE.search(s) is not None
    has_html_like = ("</" in s) or ("<br" in s.lower()) or ("<div" in s.lower())
    has_replacement_char = "\ufffd" in s  # 解码 replace 产生的 �

    sc = script_counter(s)
    total = sum(sc.values()) or 1
    return {
        "len_chars": len(s),
        "len_bytes_utf8": len(s.encode("utf-8", errors="replace")),
        "has_url": has_url,
        "has_email": has_email,
        "has_html_like": has_html_like,
        "has_replacement_char": has_replacement_char,
        "script_counts": dict(sc),
        "ascii_ratio": sc.get("ascii", 0) / total,
        "cjk_ratio": sc.get("cjk", 0) / total,
    }


def summarize_lines(lines: Iterable[str]) -> Dict[str, object]:
    lines = list(lines)
    feat_list = [line_features(x) for x in lines if x is not None]
    if not feat_list:
        return {}

    url_cnt = sum(1 for f in feat_list if f["has_url"])
    email_cnt = sum(1 for f in feat_list if f["has_email"])
    html_cnt = sum(1 for f in feat_list if f["has_html_like"])
    bad_cnt = sum(1 for f in feat_list if f["has_replacement_char"])

    lens = sorted(f["len_chars"] for f in feat_list)
    def percentile(p: float) -> int:
        if not lens:
            return 0
        idx = int(round((len(lens) - 1) * p))
        return int(lens[max(0, min(len(lens) - 1, idx))])

    # 汇总脚本分布
    scripts = Counter()
    for f in feat_list:
        scripts.update(f["script_counts"])
    total_chars = sum(scripts.values()) or 1
    script_ratio = {k: v / total_chars for k, v in scripts.most_common()}

    return {
        "n_lines": len(feat_list),
        "url_line_ratio": url_cnt / len(feat_list),
        "email_line_ratio": email_cnt / len(feat_list),
        "html_like_line_ratio": html_cnt / len(feat_list),
        "decode_replacement_line_ratio": bad_cnt / len(feat_list),
        "len_chars_p50": percentile(0.50),
        "len_chars_p90": percentile(0.90),
        "len_chars_p99": percentile(0.99),
        "script_ratio": script_ratio,
    }


def write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(x)
            f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="抽样/提取 corpus_py.txt，用于辨别语料类型与质量（纯标准库）")
    ap.add_argument("--path", default=DEFAULT_PATH, help="语料文件路径（默认 corpus_py.txt）")
    ap.add_argument("--out_dir", default="/home/arxiv_code/tokenizers_rust/corpus_inspect_out", help="输出目录")
    ap.add_argument("--head", type=int, default=30, help="输出前 N 行")
    ap.add_argument("--tail", type=int, default=30, help="输出后 N 行")
    ap.add_argument("--first_lines", type=int, default=2000, help="额外提取前 K 行到文件（0 表示不提取）")
    ap.add_argument("--random", type=int, default=200, help="随机偏移抽样行数（不扫描全文件）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument(
        "--positions",
        default="0,0.1,0.25,0.5,0.75,0.9,0.99",
        help="按文件字节比例取样的位置（逗号分隔，0~1）",
    )
    ap.add_argument("--pos_lines", type=int, default=10, help="每个位置提取多少行")

    args = ap.parse_args()

    path = args.path
    if not os.path.exists(path):
        print(f"找不到文件：{path}", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)

    size = os.path.getsize(path)
    wc_lines = fast_wc_lines(path)

    t0 = time.time()
    head_lines = read_head(path, args.head)
    tail_lines = read_tail(path, args.tail)
    rand_lines = sample_lines_by_random_offsets(path, args.random, args.seed)

    # positions 抽样
    pos_blocks = []
    for p_str in [x.strip() for x in args.positions.split(",") if x.strip()]:
        try:
            ratio = float(p_str)
        except ValueError:
            continue
        ratio = max(0.0, min(1.0, ratio))
        off = int(size * ratio)
        start_pos, lines = read_at_offset(path, off, args.pos_lines)
        pos_blocks.append({
            "ratio": ratio,
            "offset": off,
            "start_pos": start_pos,
            "lines": lines,
        })

    # first_lines 提取
    first_k = []
    if args.first_lines and args.first_lines > 0:
        first_k = read_head(path, args.first_lines)

    # 写输出文件
    write_lines(os.path.join(args.out_dir, "head.txt"), head_lines)
    write_lines(os.path.join(args.out_dir, "tail.txt"), tail_lines)
    write_lines(os.path.join(args.out_dir, "random.txt"), rand_lines)
    if first_k:
        write_lines(os.path.join(args.out_dir, f"first_{args.first_lines}.txt"), first_k)

    # positions.txt（带分隔符）
    pos_txt_path = os.path.join(args.out_dir, "positions.txt")
    with open(pos_txt_path, "w", encoding="utf-8") as f:
        for b in pos_blocks:
            f.write(f"=== ratio={b['ratio']:.2f} offset={b['offset']} start_pos={b['start_pos']} ===\n")
            for line in b["lines"]:
                f.write(line)
                f.write("\n")
            f.write("\n")

    # 统计特征（用样本集合：head + tail + random + positions）
    sample_for_stats: List[str] = []
    sample_for_stats.extend(head_lines)
    sample_for_stats.extend(tail_lines)
    sample_for_stats.extend(rand_lines)
    for b in pos_blocks:
        sample_for_stats.extend(b["lines"])

    summary = summarize_lines(sample_for_stats)

    report = {
        "path": path,
        "size_bytes": size,
        "size_human": _human_bytes(size),
        "wc_lines": wc_lines,
        "extract": {
            "head": args.head,
            "tail": args.tail,
            "random": args.random,
            "seed": args.seed,
            "positions": args.positions,
            "pos_lines": args.pos_lines,
            "first_lines": args.first_lines,
        },
        "sample_stats": summary,
        "elapsed_sec": round(time.time() - t0, 3),
    }

    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 简短结论：基于样本特征的启发式判断
    guess = []
    url_ratio = summary.get("url_line_ratio", 0.0) or 0.0
    script_ratio = summary.get("script_ratio", {}) or {}
    cjk_r = float(script_ratio.get("cjk", 0.0))
    jp_r = float(script_ratio.get("jp_kana", 0.0))
    kr_r = float(script_ratio.get("kr_hangul", 0.0))
    non_ascii = 1.0 - float(script_ratio.get("ascii", 1.0))

    if url_ratio > 0.02:
        guess.append("样本中 URL 比例较高，像网页抓取/论坛/新闻/营销文案混合语料。")
    if non_ascii > 0.02 and (cjk_r + jp_r + kr_r) > 0.005:
        guess.append("存在明显多语种/多脚本字符（含 CJK/日文/韩文等），不像单一来源（如 arXiv/维基）。")
    if not guess:
        guess.append("样本特征不明显，建议提高 random/pos_lines 再看。")

    md_lines = []
    md_lines.append("### corpus 抽样报告（自动生成）")
    md_lines.append("")
    md_lines.append(f"- **文件**: `{path}`")
    md_lines.append(f"- **大小**: **{_human_bytes(size)}**")
    if wc_lines is not None:
        md_lines.append(f"- **行数(wc -l)**: **{wc_lines}**")
    md_lines.append(f"- **抽样耗时**: **{report['elapsed_sec']}s**")
    md_lines.append("")
    md_lines.append("### 样本统计（仅基于抽样，不代表全量）")
    md_lines.append("")
    if summary:
        md_lines.append(f"- **URL 行占比**: {summary.get('url_line_ratio', 0.0):.3f}")
        md_lines.append(f"- **Email 行占比**: {summary.get('email_line_ratio', 0.0):.3f}")
        md_lines.append(f"- **疑似 HTML 行占比**: {summary.get('html_like_line_ratio', 0.0):.3f}")
        md_lines.append(f"- **解码替换(�) 行占比**: {summary.get('decode_replacement_line_ratio', 0.0):.3f}")
        md_lines.append(f"- **行长度(字符) p50/p90/p99**: {summary.get('len_chars_p50')}/{summary.get('len_chars_p90')}/{summary.get('len_chars_p99')}")
        md_lines.append("")
        md_lines.append("- **字符脚本占比（Top）**:")
        top_scripts = list(script_ratio.items())[:10]
        for k, v in top_scripts:
            md_lines.append(f"  - {k}: {v:.3f}")
    md_lines.append("")
    md_lines.append("### 启发式判断")
    md_lines.append("")
    for g in guess:
        md_lines.append(f"- {g}")
    md_lines.append("")
    md_lines.append("### 输出文件")
    md_lines.append("")
    md_lines.append(f"- `head.txt` / `tail.txt` / `random.txt` / `positions.txt` / `report.json`（以及可选的 `first_*.txt`）")

    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        f.write("\n")

    print(f"已输出到：{args.out_dir}")
    print(f"- head/tail/random/positions/report.json/report.md")
    if first_k:
        print(f"- first_{args.first_lines}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


