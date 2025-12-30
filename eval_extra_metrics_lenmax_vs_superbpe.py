#!/usr/bin/env python3
"""
Extra evaluation metrics for comparing tokenizer-produced models.

We already compare (tpc, eval_loss, bpc). This script adds:
- ppl_token = exp(loss)              (token-level, NOT comparable across tokenizers)
- ppl_char  = 2**bpc                 (char-level, comparable)
- bpb       = bits per byte          (byte-level, comparable; usually ~= bpc for ASCII corpora)
- eval_tpc  = tokens/char on eval batches
- unk_rate  = fraction of <unk> among active tokens (excluding pad)
- throughput: tok/s and char/s (end-to-end, includes tokenization)

Default model/tokenizer paths are wired to the current best comparison (v047 vs v044).
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# Reuse the exact packing/tokenization helpers used in training/eval.
import validate_modern_arch_llama as vml


@dataclass(frozen=True)
class ModelSpec:
    name: str
    tokenizer_dir: str
    model_dir: str


def _is_float_dtype(precision: str) -> torch.dtype | None:
    p = precision.lower().strip()
    if p in ("fp32", "float32"):
        return None
    if p in ("bf16", "bfloat16"):
        return torch.bfloat16
    if p in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"unknown precision: {precision!r}")


@torch.no_grad()
def eval_batches(
    *,
    model: LlamaForCausalLM,
    tok,
    lines: list[str],
    device: torch.device,
    seq_len: int,
    batch_size: int,
    batches: int,
    pack_chars: int,
    pack_mode: str,
    force_eos: bool,
    seed: int,
    precision: str,
) -> dict:
    if batches <= 0 or batch_size <= 0:
        raise ValueError("batches and batch_size must be > 0")

    rng = random.Random(int(seed))
    dtype = _is_float_dtype(precision)

    pad_id = int(tok.pad_token_id)
    unk_id = getattr(tok, "unk_token_id", None)
    unk_id = int(unk_id) if unk_id is not None else None

    total_loss = 0.0
    total_chars = 0
    total_bytes = 0
    total_tokens_for_loss = 0  # approximate: active - batch (causal shift)
    total_active_tokens = 0  # includes first token; excludes pad
    total_unk = 0

    t0 = time.perf_counter()
    model.eval()
    for _ in range(int(batches)):
        batch = [vml._pack_text(lines, int(pack_chars), rng=rng, mode=str(pack_mode)) for _ in range(int(batch_size))]
        enc = vml._encode_batch(tok, batch, seq_len=int(seq_len), force_eos=bool(force_eos))

        # Count "seen" chars/bytes by decoding the *active* tokens (post-truncation).
        seqs: list[list[int]] = []
        for ids, mask in zip(enc["input_ids"], enc["attention_mask"]):
            n = int(mask.sum().item())
            seqs.append(ids[:n].tolist())
        try:
            texts = tok.batch_decode(seqs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except TypeError:
            texts = tok.batch_decode(seqs, skip_special_tokens=True)
        total_chars += sum(len(t) for t in texts)
        total_bytes += sum(len(t.encode("utf-8", errors="ignore")) for t in texts)

        # Active tokens (exclude pad); tokens contributing to causal loss exclude 1 per sequence.
        active = int(enc["attention_mask"].sum().item())
        total_active_tokens += active
        total_tokens_for_loss += active - len(seqs)

        if unk_id is not None:
            ids = enc["input_ids"]
            mask = enc["attention_mask"].bool()
            total_unk += int(((ids == unk_id) & mask).sum().item())

        input_ids = enc["input_ids"].to(device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(device, non_blocking=True)
        labels = input_ids.clone()
        labels[labels == pad_id] = -100

        if dtype is None:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += float(out.loss.item())

    elapsed = max(1e-9, time.perf_counter() - t0)
    mean_loss = total_loss / float(batches)

    ln2 = math.log(2.0)
    eval_tpc = (float(total_tokens_for_loss) / float(total_chars)) if total_chars > 0 else 0.0
    eval_tpb = (float(total_tokens_for_loss) / float(total_bytes)) if total_bytes > 0 else 0.0

    bpc = (mean_loss * eval_tpc / ln2) if eval_tpc > 0 else float("nan")
    bpb = (mean_loss * eval_tpb / ln2) if eval_tpb > 0 else float("nan")

    ppl_token = math.exp(mean_loss)
    ppl_char = 2.0 ** bpc if math.isfinite(bpc) else float("nan")

    unk_rate = (float(total_unk) / float(total_active_tokens)) if (unk_id is not None and total_active_tokens > 0) else float("nan")

    return {
        "mean_loss": mean_loss,
        "ppl_token": ppl_token,
        "eval_tpc": eval_tpc,
        "bpc": bpc,
        "ppl_char": ppl_char,
        "bpb": bpb,
        "unk_rate": unk_rate,
        "tok_per_s": float(total_tokens_for_loss) / elapsed,
        "char_per_s": float(total_chars) / elapsed,
        "bytes_per_s": float(total_bytes) / elapsed,
        "batches": int(batches),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "pack_chars": int(pack_chars),
        "pack_mode": str(pack_mode),
        "force_eos": bool(force_eos),
        "seed": int(seed),
        "precision": str(precision),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--pack_chars", type=int, default=4000)
    ap.add_argument("--pack_mode", type=str, default="contiguous", choices=["random_lines", "contiguous"])
    ap.add_argument("--force_eos", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max_lines", type=int, default=0, help="0 = read all lines")

    ap.add_argument("--lenmax_tokenizer_dir", type=str, default="/home/arxiv_code/tokenizers_rust/tokenizer_out_wikitext103_32k_n9_full_maxchars48_punctnorm_nostage_lowtpc_crossmix_v047")
    ap.add_argument("--lenmax_model_dir", type=str, default="/home/arxiv_code/tokenizers_rust/model_lenmax_lenmax_punctnorm_nostage_crossmix_v047_vs_superbpe_pack4000_evalval_steps10000_v048/best_bpc")
    ap.add_argument("--superbpe_tokenizer_dir", type=str, default="/home/arxiv_code/tokenizers_rust/tokenizer_out_superbpe_wikitext103_32000_full_v043")
    ap.add_argument("--superbpe_model_dir", type=str, default="/home/arxiv_code/tokenizers_rust/model_superbpe_superbpe_pack4000_evalval_steps10000_v044/best_bpc")

    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    models = [
        ModelSpec("Length-MAX(v047)/best_bpc", args.lenmax_tokenizer_dir, args.lenmax_model_dir),
        ModelSpec("SuperBPE(v043)/best_bpc", args.superbpe_tokenizer_dir, args.superbpe_model_dir),
    ]

    corpora = [
        ("wt103_valid", "/home/arxiv_code/datasets/wikitext103_raw_txt/validation.txt"),
        ("wt103_test", "/home/arxiv_code/datasets/wikitext103_raw_txt/test.txt"),
        ("wt2_valid", "/home/arxiv_code/datasets/wikitext2_raw_txt/validation.txt"),
        ("wt2_test", "/home/arxiv_code/datasets/wikitext2_raw_txt/test.txt"),
    ]

    results: list[tuple[str, str, dict]] = []

    for m in models:
        tok = AutoTokenizer.from_pretrained(m.tokenizer_dir, trust_remote_code=True)
        model = LlamaForCausalLM.from_pretrained(m.model_dir)
        model.to(dev)
        model.eval()

        for corpus_name, corpus_path in corpora:
            lines = vml._read_lines(Path(corpus_path), int(args.max_lines))
            r = eval_batches(
                model=model,
                tok=tok,
                lines=lines,
                device=dev,
                seq_len=int(args.seq_len),
                batch_size=int(args.batch_size),
                batches=int(args.batches),
                pack_chars=int(args.pack_chars),
                pack_mode=str(args.pack_mode),
                force_eos=bool(args.force_eos),
                seed=int(args.seed),
                precision=str(args.precision),
            )
            results.append((corpus_name, m.name, r))

    # Pretty print
    cols = [
        ("loss", "mean_loss", "{:.4f}"),
        ("bpc", "bpc", "{:.4f}"),
        ("ppl_char", "ppl_char", "{:.3f}"),
        ("bpb", "bpb", "{:.4f}"),
        ("eval_tpc", "eval_tpc", "{:.6f}"),
        ("unk", "unk_rate", "{:.2e}"),
        ("tok/s", "tok_per_s", "{:.0f}"),
        ("char/s", "char_per_s", "{:.0f}"),
    ]

    print(f"device={dev} precision={args.precision} seq_len={args.seq_len} pack_chars={args.pack_chars} pack_mode={args.pack_mode} force_eos={bool(args.force_eos)} batches={args.batches} batch_size={args.batch_size}")
    print()
    print("corpus,model," + ",".join(k for k,_,_ in cols))
    for corpus_name, model_name, r in results:
        fields = [corpus_name, model_name]
        for _, key, fmt in cols:
            v = r.get(key, float("nan"))
            try:
                fields.append(fmt.format(v))
            except Exception:
                fields.append(str(v))
        print(",".join(fields))


if __name__ == "__main__":
    main()


