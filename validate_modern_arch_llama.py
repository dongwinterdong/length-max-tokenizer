#!/usr/bin/env python3
"""
用 LengthTokenizer（HF remote code + 可选 Rust DP 扩展）验证“现代 decoder-only 架构”的可训练性。

目标：
- 解决审稿人提出的“只在 GPT-2 上验证”的疑虑
- 用 Llama-style 组件（RoPE + RMSNorm + SwiGLU）构造一个小模型，从零训练若干步，验证流程可跑通

注意：
- 这不是完整复现实验（FineWeb/长训练/多次种子）；它是一个可复用的“现代架构验证模板”。
- 真正 rebuttal 里的结果建议用同语料、同 vocab、同超参做 Length-MAX vs BPE 的 head-to-head。
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


def _read_lines(path: Path, max_lines: int) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as r:
        for line in r:
            s = line.strip()
            if not s:
                continue
            lines.append(s)
            if max_lines > 0 and len(lines) >= max_lines:
                break
    if not lines:
        raise ValueError(f"empty corpus: {path}")
    return lines


def _ddp_setup() -> tuple[bool, int, int, int]:
    """
    Returns: (is_ddp, rank, local_rank, world_size)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.init_process_group(backend="nccl")
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def _is_main(rank: int) -> bool:
    return rank == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_dir", type=str, required=True, help="Length-MAX 导出的 HF tokenizer 目录（含 vocab.json 等）")
    ap.add_argument("--corpus_file", type=str, required=True, help="训练用文本：每行一句")
    ap.add_argument("--max_lines", type=int, default=2048, help="最多读取多少行（0 表示不限制）")
    ap.add_argument("--seed", type=int, default=42)

    # 训练设置（快速 sanity check）
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8, help="global batch size（DDP 下会按 world_size 自动切分）")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--grad_checkpointing", action="store_true")

    # Llama-style 小模型结构（可按需要调到 ~100M 量级）
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_kv_heads", type=int, default=8)
    ap.add_argument("--intermediate_size", type=int, default=688)  # 常见经验：约 2.7x hidden_size
    ap.add_argument("--rope_theta", type=float, default=10000.0)
    ap.add_argument("--rms_norm_eps", type=float, default=1e-6)

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--disable_rust", action="store_true", help="禁用 Rust 扩展分词（用于对齐/排障）")
    args = ap.parse_args()

    is_ddp, rank, local_rank, world_size = _ddp_setup()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.disable_rust:
        os.environ["LENGTH_TOKENIZER_DISABLE_RUST"] = "1"

    # 1) load corpus
    corpus_path = Path(args.corpus_file)
    lines = _read_lines(corpus_path, args.max_lines)

    # 2) load tokenizer (remote code)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    assert tok.pad_token_id is not None, "pad_token_id is required"
    assert tok.bos_token_id is not None, "bos_token_id is required"
    assert tok.eos_token_id is not None, "eos_token_id is required"

    # 3) build a tiny Llama model (RoPE + RMSNorm + SwiGLU)
    cfg = LlamaConfig(
        vocab_size=tok.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        max_position_embeddings=args.seq_len,
        rope_theta=float(args.rope_theta),
        rms_norm_eps=float(args.rms_norm_eps),
        bos_token_id=int(tok.bos_token_id),
        eos_token_id=int(tok.eos_token_id),
        pad_token_id=int(tok.pad_token_id),
    )
    model = LlamaForCausalLM(cfg)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if is_ddp:
        assert device == "cuda", "DDP requires --device cuda (or auto with CUDA available)"
        torch.cuda.set_device(local_rank)
        model.to(torch.device("cuda", local_rank))
    else:
        model.to(device)

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    num_params = sum(p.numel() for p in model.parameters())
    if _is_main(rank):
        print("== setup ==")
        print(f"tokenizer_dir = {args.tokenizer_dir}")
        print(f"corpus_file   = {args.corpus_file} (lines={len(lines)})")
        print(f"vocab_size    = {tok.vocab_size}")
        print(f"pad/bos/eos   = {tok.pad_token_id}/{tok.bos_token_id}/{tok.eos_token_id}")
        print(f"model         = LlamaForCausalLM (params={num_params})")
        print(f"device        = {device}")
        print(f"ddp           = {is_ddp} world_size={world_size}")
        print(f"seq_len       = {args.seq_len} batch_size={args.batch_size} steps={args.steps} grad_accum={args.grad_accum}")
        print(f"precision     = {args.precision} grad_ckpt={args.grad_checkpointing}")
        print(f"rust_active   = {getattr(tok, '_rust', None) is not None}")
        print()

    # 4) training loop (few steps)
    model.train()
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()
        autocast_dtype = torch.float16
    elif args.precision == "bf16":
        scaler = None
        autocast_dtype = torch.bfloat16
    else:
        scaler = None
        autocast_dtype = None

    t0 = time.perf_counter()
    for step in range(args.steps):
        # DDP：global batch 会按 world_size 切分到每个 rank
        per_rank_bs = max(1, args.batch_size // world_size)
        batch = [random.choice(lines) for _ in range(per_rank_bs)]
        enc = tok(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.seq_len,
            add_special_tokens=True,
        )
        if is_ddp:
            dev = torch.device("cuda", local_rank)
        else:
            dev = torch.device(device)
        input_ids = enc["input_ids"].to(dev)
        attention_mask = enc["attention_mask"].to(dev)

        labels = input_ids.clone()
        labels[labels == int(tok.pad_token_id)] = -100

        if autocast_dtype is None:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss / max(1, args.grad_accum)
            loss.backward()
        else:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss / max(1, args.grad_accum)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if (step + 1) % max(1, args.grad_accum) == 0:
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

        if _is_main(rank) and (step % args.print_every == 0 or step == args.steps - 1):
            elapsed = time.perf_counter() - t0
            toks = (step + 1) * args.batch_size * args.seq_len
            print(f"step={step:04d} loss={loss.item():.4f} tok/s={toks/elapsed:.1f}")

    # 5) quick generate
    if is_ddp:
        model.module.eval()
    else:
        model.eval()
    prompt = "hello world"
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    if is_ddp:
        dev = torch.device("cuda", local_rank)
    else:
        dev = torch.device(device)
    input_ids = enc["input_ids"].to(dev)
    with torch.no_grad():
        gen_model = model.module if is_ddp else model
        gen = gen_model.generate(
            input_ids=input_ids,
            max_new_tokens=32,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=int(tok.pad_token_id),
            eos_token_id=int(tok.eos_token_id),
        )
    text = tok.decode(gen[0].tolist(), skip_special_tokens=True)
    if _is_main(rank):
        print()
        print("== generate ==")
        print(f"prompt = {prompt!r}")
        print(f"out    = {text!r}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


