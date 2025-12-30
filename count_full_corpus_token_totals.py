#!/usr/bin/env python3
"""
Count total token numbers on a corpus for two tokenizers:
  - Length-MAX (remote code; uses Rust DP batch path if extension is installed)
  - BPE (HF tokenizer.json)

We count with:
  add_special_tokens=False, padding=False, truncation=False

This matches the TPC estimate convention used in validate_modern_arch_llama.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer


SPECIAL = {"<unk>", "<pad>", "<s>", "</s>", "<mask>"}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _preview_vocab(vocab: dict[str, int], name: str, top_k: int = 20, max_show: int = 30) -> str:
    items = []
    for tok, tid in vocab.items():
        if tok in SPECIAL:
            continue
        items.append((len(tok), tok, int(tid)))
    items.sort(key=lambda x: (-x[0], x[1], x[2]))

    lines = []
    lines.append(f"[{name}] vocab_size={len(vocab)}")
    lines.append(f"[{name}] longest_tokens(top {top_k}):")
    for i, (l, tok, tid) in enumerate(items[:top_k]):
        # repr() for safety (may contain control chars)
        s = repr(tok)
        if len(s) > max_show:
            s = s[: max_show - 3] + "..."
        lines.append(f"  #{i+1:02d} len={l:>3d} id={tid:>5d} tok={s}")
    return "\n".join(lines)


def _iter_corpus_lines(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip():
                continue
            yield s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, required=True)
    ap.add_argument("--len_tok_dir", type=str, required=True)
    ap.add_argument("--bpe_tok_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--report_every", type=int, default=20000)
    ap.add_argument("--max_lines", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    len_dir = Path(args.len_tok_dir)
    bpe_dir = Path(args.bpe_tok_dir)

    # vocab preview
    len_vocab = _read_json(len_dir / "vocab.json")
    bpe_vocab = _read_json(bpe_dir / "vocab.json")
    print(_preview_vocab(len_vocab, "Length-MAX"))
    print()
    print(_preview_vocab(bpe_vocab, "BPE"))
    print()

    # load tokenizers
    print("[load] loading tokenizers ...")
    t0 = time.time()
    len_tok = AutoTokenizer.from_pretrained(str(len_dir), trust_remote_code=True)
    bpe_tok = AutoTokenizer.from_pretrained(str(bpe_dir), use_fast=True)
    print(f"[load] done in {time.time()-t0:.2f}s  len_rust_active={getattr(len_tok, '_rust', None) is not None}")
    print()

    total_chars = 0
    total_lines = 0
    len_tokens = 0
    bpe_tokens = 0

    buf: list[str] = []
    t_start = time.time()
    for s in _iter_corpus_lines(corpus):
        buf.append(s)
        total_chars += len(s)
        total_lines += 1
        if args.max_lines > 0 and total_lines >= args.max_lines:
            break
        if len(buf) >= args.batch_size:
            # Length-MAX
            enc_len = len_tok(
                buf,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            len_tokens += sum(len(ids) for ids in enc_len["input_ids"])

            # BPE
            enc_bpe = bpe_tok(
                buf,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            bpe_tokens += sum(len(ids) for ids in enc_bpe["input_ids"])

            buf.clear()

            if args.report_every > 0 and (total_lines % args.report_every == 0):
                elapsed = time.time() - t_start
                l_tpc = (len_tokens / total_chars) if total_chars else 0.0
                b_tpc = (bpe_tokens / total_chars) if total_chars else 0.0
                print(
                    f"[progress] lines={total_lines} chars={total_chars} "
                    f"len_tokens={len_tokens} (tpc={l_tpc:.6f}) "
                    f"bpe_tokens={bpe_tokens} (tpc={b_tpc:.6f}) "
                    f"elapsed={elapsed:.1f}s"
                )

    if buf:
        enc_len = len_tok(
            buf,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        len_tokens += sum(len(ids) for ids in enc_len["input_ids"])
        enc_bpe = bpe_tok(
            buf,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        bpe_tokens += sum(len(ids) for ids in enc_bpe["input_ids"])
        buf.clear()

    elapsed = time.time() - t_start
    len_tpc = (len_tokens / total_chars) if total_chars else 0.0
    bpe_tpc = (bpe_tokens / total_chars) if total_chars else 0.0
    print()
    print("[done]")
    print(f"lines        = {total_lines}")
    print(f"chars        = {total_chars}")
    print(f"len_tokens   = {len_tokens}  tpc={len_tpc:.6f}  chars/token={(total_chars/len_tokens if len_tokens else 0):.3f}")
    print(f"bpe_tokens   = {bpe_tokens}  tpc={bpe_tpc:.6f}  chars/token={(total_chars/bpe_tokens if bpe_tokens else 0):.3f}")
    print(f"elapsed      = {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())








