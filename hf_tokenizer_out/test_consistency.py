#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 HuggingFace tokenizer 包（remote-code）与 Rust 扩展（PyO3 wheel）的一致性。

目标：
- 纯 Python DP（fallback） 与 Rust DP（length_tokenizer_rs.DpTokenizer）在同一份 vocab.json 上输出完全一致的 token 序列。

说明：
- 该脚本不依赖 transformers（当前环境没有），会注入一个最小 stub 来加载 tokenization_length_tokenizer.py。
- 若系统里没有安装 wheel，本脚本可选：用“本地 cargo build 的 .so”来动态加载扩展模块（开发者模式）。

用法：
    python3 test_consistency.py --vocab ./vocab.json --module ./tokenization_length_tokenizer.py

开发者模式（可选）：
    python3 test_consistency.py --build-local-ext --try-load-local-ext
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional


def _install_transformers_stub() -> None:
    # 让 tokenization_length_tokenizer.py 在没有 transformers 时也能 import
    import types

    m = types.ModuleType("transformers")

    class PreTrainedTokenizer:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def tokenize(self, text: str) -> List[str]:
            return self._tokenize(text)  # type: ignore[attr-defined]

    m.PreTrainedTokenizer = PreTrainedTokenizer  # type: ignore[attr-defined]
    sys.modules["transformers"] = m


def _load_py_module_from_file(mod_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[call-arg]
    return mod


def _maybe_build_local_ext(crate_dir: Path) -> None:
    import subprocess

    cmd = ["cargo", "build", "--release", "--features", "python"]
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.check_call(cmd, cwd=str(crate_dir))


def _try_load_local_ext(crate_dir: Path) -> bool:
    """
    在未安装 wheel 的情况下，尝试从本地 target/release 的 .so 直接加载 PyO3 扩展。
    """
    import glob

    # cargo build --features python 会生成 liblength_tokenizer.so（名字不含 module 名）
    # 但里面导出了 PyInit_length_tokenizer_rs，所以我们可以用 spec_from_file_location 指定模块名来加载。
    patterns = [
        str(crate_dir / "target" / "release" / "liblength_tokenizer*.so"),
        str(crate_dir / "target" / "release" / "liblength_tokenizer*.dylib"),
        str(crate_dir / "target" / "release" / "liblength_tokenizer*.dll"),
    ]
    cands: List[str] = []
    for p in patterns:
        cands.extend(glob.glob(p))
    if not cands:
        return False
    # 取第一个
    so_path = Path(sorted(cands)[0])
    mod = _load_py_module_from_file(so_path, "length_tokenizer_rs")
    sys.modules["length_tokenizer_rs"] = mod
    return True


def _load_vocab(vocab_path: Path) -> dict:
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="测试 HF tokenizer（Python DP）与 Rust 扩展（DpTokenizer）一致性")
    ap.add_argument("--vocab", default="vocab.json", help="vocab.json 路径")
    ap.add_argument("--module", default="tokenization_length_tokenizer.py", help="remote-code tokenizer 文件路径")
    ap.add_argument("--n", type=int, default=200, help="随机生成测试文本数量")
    ap.add_argument("--seed", type=int, default=7, help="随机种子")
    ap.add_argument("--build-local-ext", action="store_true", help="先 cargo build --release --features python（开发者模式）")
    ap.add_argument("--try-load-local-ext", action="store_true", help="尝试从本地 target/release 直接加载扩展（开发者模式）")
    ap.add_argument("--crate-dir", default=str(Path(__file__).resolve().parents[1]), help="crate 根目录（默认上一层 tokenizers_rust）")
    args = ap.parse_args()

    vocab_path = Path(args.vocab).resolve()
    mod_path = Path(args.module).resolve()
    crate_dir = Path(args.crate_dir).resolve()

    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab not found: {vocab_path}")
    if not mod_path.exists():
        raise FileNotFoundError(f"module not found: {mod_path}")

    _install_transformers_stub()

    # 尝试确保 Rust 扩展可 import
    rust_ok = False
    if args.build_local_ext:
        _maybe_build_local_ext(crate_dir)
    try:
        import length_tokenizer_rs  # noqa: F401

        rust_ok = True
    except Exception:
        if args.try_load_local_ext:
            rust_ok = _try_load_local_ext(crate_dir)

    # 导入 remote-code tokenizer 模块
    hfmod = _load_py_module_from_file(mod_path, "tokenization_length_tokenizer")

    # 生成测试文本：从 vocab 里随机抽 token 拼接（避免大量 <unk> 干扰）
    vocab = _load_vocab(vocab_path)
    tokens = [t for t in vocab.keys() if t not in ("<pad>", "<mask>", "<s>", "</s>")]
    rng = random.Random(args.seed)

    def gen_text() -> str:
        # 随机拼一些 token，再把 END_TOKEN(Ġ) 替换成空格，让输入更“自然”
        k = rng.randint(5, 40)
        s = "".join(rng.choice(tokens) for _ in range(k))
        return s.replace("Ġ", " ")

    texts = [gen_text() for _ in range(args.n)]
    texts.extend([
        "hello world",
        "The quick brown fox jumps over the lazy dog",
        "中文 测试 一下 tokenizer",
        "Mixed 英文 and 中文 123 !!!",
    ])

    # 1) 纯 Python DP（强制禁用 Rust）
    os.environ["LENGTH_TOKENIZER_DISABLE_RUST"] = "1"
    tok_py = hfmod.LengthTokenizer(str(vocab_path))
    out_py = [tok_py.tokenize(t) for t in texts]

    # 2) Rust DP（如果可用）
    out_rust: Optional[List[List[str]]] = None
    if rust_ok:
        os.environ.pop("LENGTH_TOKENIZER_DISABLE_RUST", None)
        tok_rs = hfmod.LengthTokenizer(str(vocab_path))
        if getattr(tok_rs, "_rust", None) is None:
            raise RuntimeError("Rust 扩展应可用，但 tokenizer 没有启用 _rust（请检查 import/环境变量）")
        out_rust = [tok_rs.tokenize(t) for t in texts]

        # 对比
        for i, (a, b, text) in enumerate(zip(out_py, out_rust, texts)):
            if a != b:
                print("Mismatch at index:", i)
                print("text:", repr(text))
                print("python:", a)
                print("rust  :", b)
                return 1
        print(f"OK: python DP == rust DP on {len(texts)} texts")
    else:
        print("Rust 扩展不可用（未安装 wheel 且未加载本地 .so），仅跑了 Python DP。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


