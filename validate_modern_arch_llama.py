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
import hashlib
import math
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


def _pack_text(lines: list[str], target_chars: int, *, rng: random.Random, mode: str) -> str:
    """
    Concatenate multiple lines to roughly reach target_chars.
    This reduces padding and makes tokenizer comparisons fairer.
    """
    if not lines:
        raise ValueError("lines is empty")
    if target_chars <= 0:
        return lines[rng.randrange(len(lines))]
    parts: list[str] = []
    n = 0
    # cap to avoid pathological loops on very short corpora
    if mode == "random_lines":
        while n < target_chars and len(parts) < 10_000:
            s = lines[rng.randrange(len(lines))]
            parts.append(s)
            n += len(s) + 1
    elif mode == "contiguous":
        i = rng.randrange(len(lines))
        while n < target_chars and len(parts) < 10_000:
            s = lines[i]
            parts.append(s)
            n += len(s) + 1
            i += 1
            if i >= len(lines):
                i = 0
    else:
        raise ValueError(f"unknown pack mode: {mode!r}")
    return "\n".join(parts)


def _encode_batch(tok, batch: list[str], *, seq_len: int, force_eos: bool):
    """
    Encode a batch into fixed-length tensors.

    Default path uses the tokenizer's built-in special-token handling.
    Optional force_eos path ensures each sample ends with EOS (useful when
    long packed samples would otherwise truncate EOS away).
    """
    if seq_len <= 2 and force_eos:
        raise ValueError("seq_len must be > 2 when --force_eos is enabled")

    if not force_eos:
        return tok(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            add_special_tokens=True,
        )

    # Force: [BOS] + tokens[:seq_len-2] + [EOS] (+ pad)
    enc_raw = tok(
        batch,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=seq_len - 2,
        return_attention_mask=False,
    )
    bos = int(tok.bos_token_id)
    eos = int(tok.eos_token_id)
    pad = int(tok.pad_token_id)

    bs = len(batch)
    input_ids = torch.full((bs, seq_len), pad, dtype=torch.long)
    attention_mask = torch.zeros((bs, seq_len), dtype=torch.long)
    for i, ids in enumerate(enc_raw["input_ids"]):
        full = [bos] + list(ids) + [eos]
        # Should already fit due to max_length=seq_len-2, but keep safe.
        if len(full) > seq_len:
            full = full[: seq_len - 1] + [eos]
        input_ids[i, : len(full)] = torch.tensor(full, dtype=torch.long)
        attention_mask[i, : len(full)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = float(lr)


def _compute_lr(
    *,
    base_lr: float,
    min_lr: float,
    schedule: str,
    warmup_steps: int,
    update_idx: int,
    num_updates: int,
) -> float:
    # Linear warmup to base_lr (if enabled).
    if warmup_steps > 0 and update_idx < warmup_steps:
        return base_lr * float(update_idx + 1) / float(warmup_steps)

    if schedule == "constant":
        return base_lr

    if schedule == "cosine":
        if num_updates <= warmup_steps + 1:
            return min_lr
        t = float(update_idx - warmup_steps) / float(max(1, num_updates - warmup_steps - 1))
        t = max(0.0, min(1.0, t))
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_lr + (base_lr - min_lr) * cos

    raise ValueError(f"unknown lr_schedule: {schedule!r}")


def _run_meta_lines(args: argparse.Namespace, *, tpc: float) -> list[str]:
    # Keep this fairly stable so runs are easy to diff.
    return [
        f"tokenizer_dir={args.tokenizer_dir}",
        f"corpus_file={args.corpus_file}",
        f"max_lines={args.max_lines}",
        f"seed={args.seed}",
        f"seq_len={args.seq_len}",
        f"batch_size={args.batch_size}",
        f"steps={args.steps}",
        f"step_offset={getattr(args, 'step_offset', 0)}",
        f"lr={args.lr}",
        f"lr_schedule={getattr(args, 'lr_schedule', 'constant')}",
        f"warmup_steps={getattr(args, 'warmup_steps', 0)}",
        f"min_lr={getattr(args, 'min_lr', 0.0)}",
        f"weight_decay={args.weight_decay}",
        f"adam_beta1={getattr(args, 'adam_beta1', 0.9)}",
        f"adam_beta2={getattr(args, 'adam_beta2', 0.999)}",
        f"adam_eps={getattr(args, 'adam_eps', 1e-8)}",
        f"grad_clip={getattr(args, 'grad_clip', 0.0)}",
        f"precision={args.precision}",
        f"grad_checkpointing={args.grad_checkpointing}",
        f"hidden_size={args.hidden_size}",
        f"num_layers={args.num_layers}",
        f"num_heads={args.num_heads}",
        f"num_kv_heads={args.num_kv_heads}",
        f"intermediate_size={args.intermediate_size}",
        f"rope_theta={args.rope_theta}",
        f"rms_norm_eps={args.rms_norm_eps}",
        f"pack_chars={args.pack_chars}",
        f"pack_mode={getattr(args, 'pack_mode', 'random_lines')}",
        f"force_eos={getattr(args, 'force_eos', False)}",
        f"tpc_est={tpc}",
        f"resume_model_dir={getattr(args, 'resume_model_dir', '')}",
        f"save_every={getattr(args, 'save_every', 0)}",
        f"save_best_metric={getattr(args, 'save_best_metric', 'none')}",
        f"save_best_min_delta={getattr(args, 'save_best_min_delta', 0.0)}",
        f"gen_prompt={getattr(args, 'gen_prompt', 'hello world')}",
        f"gen_greedy={getattr(args, 'gen_greedy', False)}",
        f"gen_temperature={getattr(args, 'gen_temperature', 0.8)}",
        f"gen_top_p={getattr(args, 'gen_top_p', 0.95)}",
        f"gen_top_k={getattr(args, 'gen_top_k', 0)}",
        f"gen_repetition_penalty={getattr(args, 'gen_repetition_penalty', 1.0)}",
        f"gen_no_repeat_ngram_size={getattr(args, 'gen_no_repeat_ngram_size', 0)}",
    ]


def _save_pretrained_cpu(*, to_save: LlamaForCausalLM, save_dir: Path, meta_lines: list[str]) -> None:
    """
    Save a model safely while it may still live on GPU.
    We snapshot a CPU state_dict and pass it into save_pretrained (safetensors-friendly).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.detach().cpu() for k, v in to_save.state_dict().items()}
    to_save.save_pretrained(str(save_dir), safe_serialization=True, state_dict=state_dict)
    (save_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n")


def _eval_model(
    *,
    model: LlamaForCausalLM,
    tok,
    lines: list[str],
    device: torch.device,
    seq_len: int,
    batch_size: int,
    pack_chars: int,
    pack_mode: str,
    force_eos: bool,
    autocast_dtype,
    seed: int,
    batches: int,
) -> tuple[float, float]:
    """
    Evaluate mean loss/bpc on a deterministic set of sampled batches.

    Note: this is still "train split" eval unless you point --corpus_file to a held-out file.
    It is nevertheless much less noisy than single-batch train loss.
    """
    if batches <= 0:
        raise ValueError("batches must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    # Deterministic sampling independent of training RNG.
    rng = random.Random(int(seed))

    model_was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    total_chars = 0
    total_tokens = 0
    ln2 = math.log(2.0)
    with torch.no_grad():
        for _ in range(int(batches)):
            batch = [_pack_text(lines, int(pack_chars), rng=rng, mode=str(pack_mode)) for _ in range(int(batch_size))]
            enc = _encode_batch(tok, batch, seq_len=int(seq_len), force_eos=bool(force_eos))
            # IMPORTANT: count chars on the *encoded & truncated* sequence, not on the
            # pre-truncation packed string. Otherwise, with pack_chars>0 and fixed seq_len,
            # we would systematically under-estimate TPC (and thus bpc) by including chars
            # that never enter the model due to token truncation.
            #
            # We approximate "chars seen by the model" by decoding the active (non-pad)
            # token ids back to text and counting its length.
            seqs: list[list[int]] = []
            for ids, mask in zip(enc["input_ids"], enc["attention_mask"]):
                n = int(mask.sum().item())
                seqs.append(ids[:n].tolist())
            try:
                texts = tok.batch_decode(seqs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            except TypeError:
                # Some tokenizers may not support clean_up_tokenization_spaces.
                texts = tok.batch_decode(seqs, skip_special_tokens=True)
            total_chars += sum(len(t) for t in texts)
            # Approximate token count that contributes to causal loss: exclude the first token
            # per sequence (HF causal LM shifts labels by 1). This avoids a tiny systematic
            # mismatch in TPC.
            total_tokens += int(enc["attention_mask"].sum().item()) - len(seqs)
            input_ids = enc["input_ids"].to(device, non_blocking=True)
            attention_mask = enc["attention_mask"].to(device, non_blocking=True)
            labels = input_ids.clone()
            labels[labels == int(tok.pad_token_id)] = -100

            if autocast_dtype is None:
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:
                with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total += float(out.loss.item())
            count += 1

    if model_was_training:
        model.train()

    mean_loss = total / max(1, count)
    # bpc proxy using observed active tokens and decoded (post-truncation) chars on eval batches.
    tpc = (float(total_tokens) / float(total_chars)) if total_chars > 0 else 0.0
    mean_bpc = (mean_loss * tpc / ln2) if tpc > 0 else float("nan")
    return mean_loss, mean_bpc


def _estimate_tpc(tok, lines: list[str], max_lines: int, batch_size: int = 4096) -> float:
    """
    Estimate tokens-per-character on a slice of the corpus (no special tokens).
    Note: token-level loss is not directly comparable across tokenizers; bits/char
    (loss * tpc / ln2) is a more apples-to-apples proxy.
    """
    n = len(lines) if max_lines <= 0 else min(len(lines), max_lines)
    if n <= 0:
        return 0.0
    total_chars = 0
    total_tokens = 0
    for i in range(0, n, batch_size):
        batch = lines[i : i + batch_size]
        total_chars += sum(len(s) for s in batch)
        enc = tok(batch, add_special_tokens=False, padding=False, truncation=False)
        total_tokens += sum(len(ids) for ids in enc["input_ids"])
    return (total_tokens / total_chars) if total_chars > 0 else 0.0


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
    ap.add_argument("--eval_corpus_file", type=str, default="", help="若非空，则评测时使用该语料文件（建议 validation/test）")
    ap.add_argument("--eval_max_lines", type=int, default=0, help="eval 语料最多读取多少行（0 表示不限制）")
    ap.add_argument("--seed", type=int, default=42)

    # 训练设置（快速 sanity check）
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8, help="global batch size（DDP 下会按 world_size 自动切分）")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "cosine"])
    ap.add_argument("--warmup_steps", type=int, default=0, help="optimizer warmup steps (in updates, not micro-steps)")
    ap.add_argument("--min_lr", type=float, default=0.0, help="for cosine schedule")
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--adam_beta1", type=float, default=0.9)
    ap.add_argument("--adam_beta2", type=float, default=0.999)
    ap.add_argument("--adam_eps", type=float, default=1e-8)
    ap.add_argument("--grad_clip", type=float, default=0.0, help="0 disables grad norm clipping")
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--grad_checkpointing", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=64, help="训练结束后生成样例的长度")
    ap.add_argument("--gen_prompt", type=str, default="hello world", help="训练结束后生成用的 prompt")
    ap.add_argument("--gen_greedy", action="store_true", help="使用 greedy 解码（关闭采样）")
    ap.add_argument("--gen_temperature", type=float, default=0.8)
    ap.add_argument("--gen_top_p", type=float, default=0.95)
    ap.add_argument("--gen_top_k", type=int, default=0, help="0 disables top-k")
    ap.add_argument("--gen_repetition_penalty", type=float, default=1.0)
    ap.add_argument("--gen_no_repeat_ngram_size", type=int, default=0)
    ap.add_argument(
        "--gen_from_best",
        action="store_true",
        help="若保存了 best checkpoint，则最终生成时从 best 模型生成（否则从 final 模型生成）",
    )
    ap.add_argument("--pack_chars", type=int, default=0, help="把多行拼接到至少这么多字符（减少 padding；0 表示不拼接）")
    ap.add_argument(
        "--pack_mode",
        type=str,
        default="random_lines",
        choices=["random_lines", "contiguous"],
        help="how to pack lines when --pack_chars>0",
    )
    ap.add_argument(
        "--force_eos",
        action="store_true",
        help="ensure each training sample ends with EOS (useful when packing+truncation would drop EOS)",
    )
    ap.add_argument("--tpc_estimate_lines", type=int, default=20000, help="估计 TPC 时用多少行（0 表示用全部）")
    ap.add_argument("--save_dir", type=str, default="", help="若非空，则在训练结束后保存模型到该目录（仅 rank0）")
    ap.add_argument("--resume_model_dir", type=str, default="", help="从已保存的 HF 目录恢复模型权重继续训练（stage-2）")
    ap.add_argument("--save_every", type=int, default=0, help="每 N 次 optimizer update 保存一次 checkpoint（0 禁用）")
    ap.add_argument("--save_best_metric", type=str, default="none", choices=["none", "loss", "bpc"])
    ap.add_argument("--save_best_on", type=str, default="train", choices=["train", "eval"])
    ap.add_argument("--save_best_min_delta", type=float, default=0.0, help="best 保存的最小改进幅度")
    ap.add_argument("--step_offset", type=int, default=0, help="仅用于日志：打印 step 时加上 offset（方便续训拼曲线）")
    ap.add_argument("--eval_every", type=int, default=0, help="每 N 次 optimizer update 跑一次 eval（0 禁用）")
    ap.add_argument("--eval_batches", type=int, default=10, help="每次 eval 统计多少个 batch 的平均 loss")
    ap.add_argument("--eval_seed", type=int, default=12345, help="eval 采样用的固定 seed（保证可复现）")
    ap.add_argument("--eval_batch_size", type=int, default=0, help="eval 的 batch size（0 表示使用训练 batch_size）")
    ap.add_argument(
        "--log_batch_digest",
        action="store_true",
        help="在打印 step 日志时附带 batch 文本的短 hash 与字符统计，便于分析 loss 起伏是否由样本难度驱动",
    )

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

    # Important: in DDP each rank should see different random batches; otherwise
    # two GPUs would repeatedly train on identical data (effective batch size
    # would not scale). We offset RNG seed by rank.
    seed = int(args.seed) + int(rank)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.disable_rust:
        os.environ["LENGTH_TOKENIZER_DISABLE_RUST"] = "1"

    # 1) load corpus
    corpus_path = Path(args.corpus_file)
    lines = _read_lines(corpus_path, args.max_lines)
    # Optional held-out eval corpus (recommended for comparing model quality)
    if str(args.eval_corpus_file).strip():
        eval_path = Path(str(args.eval_corpus_file).strip())
        eval_lines = _read_lines(eval_path, int(args.eval_max_lines))
    else:
        eval_path = None
        eval_lines = lines

    # 2) load tokenizer (remote code)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    assert tok.pad_token_id is not None, "pad_token_id is required"
    assert tok.bos_token_id is not None, "bos_token_id is required"
    assert tok.eos_token_id is not None, "eos_token_id is required"
    tpc = _estimate_tpc(tok, lines, int(args.tpc_estimate_lines))
    bpc_factor = (tpc / math.log(2.0)) if tpc > 0 else 0.0

    # 3) build / resume a tiny Llama model (RoPE + RMSNorm + SwiGLU)
    if str(args.resume_model_dir).strip():
        resume_dir = str(args.resume_model_dir).strip()
        model = LlamaForCausalLM.from_pretrained(resume_dir)
        # basic safety checks
        if getattr(model.config, "max_position_embeddings", args.seq_len) < int(args.seq_len):
            raise SystemExit(
                f"--seq_len={args.seq_len} exceeds resume model max_position_embeddings={model.config.max_position_embeddings}"
            )
        # Ensure special token ids align with tokenizer
        model.config.bos_token_id = int(tok.bos_token_id)
        model.config.eos_token_id = int(tok.eos_token_id)
        model.config.pad_token_id = int(tok.pad_token_id)
        # If vocab differs (shouldn't in our runs), resize safely.
        if int(getattr(model.config, "vocab_size", tok.vocab_size)) != int(tok.vocab_size):
            model.resize_token_embeddings(int(tok.vocab_size))
            model.config.vocab_size = int(tok.vocab_size)
    else:
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
        if eval_path is not None:
            print(f"eval_corpus   = {str(eval_path)} (lines={len(eval_lines)})")
        print(f"vocab_size    = {tok.vocab_size}")
        print(f"pad/bos/eos   = {tok.pad_token_id}/{tok.bos_token_id}/{tok.eos_token_id}")
        print(f"model         = LlamaForCausalLM (params={num_params})")
        if str(args.resume_model_dir).strip():
            print(f"resume_model_dir = {args.resume_model_dir}")
        print(f"device        = {device}")
        print(f"ddp           = {is_ddp} world_size={world_size}")
        print(f"seq_len       = {args.seq_len} batch_size={args.batch_size} steps={args.steps} grad_accum={args.grad_accum}")
        if int(getattr(args, "step_offset", 0)) != 0:
            print(f"step_offset   = {args.step_offset}")
        print(f"precision     = {args.precision} grad_ckpt={args.grad_checkpointing}")
        print(f"rust_active   = {getattr(tok, '_rust', None) is not None}")
        print(f"tpc_est       = {tpc:.6f} tok/char (bits/char ≈ loss * {bpc_factor:.6f})")
        print(f"pack_chars    = {args.pack_chars}")
        print(f"pack_mode     = {args.pack_mode}")
        print(f"force_eos     = {args.force_eos}")
        print(f"lr            = {args.lr} schedule={args.lr_schedule} warmup_steps={args.warmup_steps} min_lr={args.min_lr}")
        print(
            f"adamw         = beta1={args.adam_beta1} beta2={args.adam_beta2} eps={args.adam_eps} wd={args.weight_decay} grad_clip={args.grad_clip}"
        )
        if int(getattr(args, "save_every", 0)) > 0:
            print(f"save_every    = {args.save_every} updates")
        if str(getattr(args, "save_best_metric", "none")) != "none":
            print(f"save_best     = metric={args.save_best_metric} min_delta={args.save_best_min_delta}")
        print()

    # 4) training loop (few steps)
    model.train()
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        eps=float(args.adam_eps),
    )

    # LR schedule bookkeeping (updates, not micro-steps)
    grad_accum = max(1, int(args.grad_accum))
    num_updates = (int(args.steps) + grad_accum - 1) // grad_accum
    warmup_steps = max(0, int(args.warmup_steps))
    warmup_steps = min(warmup_steps, num_updates)
    update_idx = 0
    # Seeded sampler for deterministic packing per rank.
    pack_rng = random.Random(seed)
    best_metric = float("inf")
    best_source = ""
    meta_lines = _run_meta_lines(args, tpc=float(tpc))

    if args.precision == "fp16":
        # torch.cuda.amp.* is deprecated in newer PyTorch; prefer torch.amp.*
        scaler = torch.amp.GradScaler("cuda")
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
        batch = [
            _pack_text(lines, int(args.pack_chars), rng=pack_rng, mode=str(args.pack_mode)) for _ in range(per_rank_bs)
        ]
        enc = _encode_batch(tok, batch, seq_len=int(args.seq_len), force_eos=bool(args.force_eos))
        if is_ddp:
            dev = torch.device("cuda", local_rank)
        else:
            dev = torch.device(device)
        input_ids = enc["input_ids"].to(dev, non_blocking=True)
        attention_mask = enc["attention_mask"].to(dev, non_blocking=True)

        labels = input_ids.clone()
        labels[labels == int(tok.pad_token_id)] = -100
        # local effective tokens (non-pad) for debugging fairness
        local_active = int((labels != -100).sum().item())

        if autocast_dtype is None:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss / max(1, args.grad_accum)
            loss.backward()
        else:
            # torch.cuda.amp.autocast is deprecated in newer PyTorch; prefer torch.amp.autocast.
            # Use the actual device type to allow CPU fallback runs if needed.
            with torch.amp.autocast(device_type=dev.type, dtype=autocast_dtype):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss / max(1, args.grad_accum)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if (step + 1) % grad_accum == 0:
            # Schedule LR on optimizer-update boundaries.
            lr_now = _compute_lr(
                base_lr=float(args.lr),
                min_lr=float(args.min_lr),
                schedule=str(args.lr_schedule),
                warmup_steps=int(warmup_steps),
                update_idx=int(update_idx),
                num_updates=int(num_updates),
            )
            _set_lr(opt, lr_now)

            # Optional grad clipping (unscale first for fp16).
            if float(args.grad_clip) > 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))

            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            update_idx += 1
            if args.save_dir and int(getattr(args, "save_every", 0)) > 0 and _is_main(rank):
                if (update_idx % int(args.save_every)) == 0:
                    save_root = Path(args.save_dir)
                    ckpt_dir = save_root / f"ckpt_update{update_idx:06d}_step{(int(args.step_offset)+step):06d}"
                    to_save = model.module if is_ddp else model
                    _save_pretrained_cpu(to_save=to_save, save_dir=ckpt_dir, meta_lines=meta_lines + [f"ckpt=1"])
                    print(f"== saved_ckpt ==\nmodel_dir = {ckpt_dir}\n")

            # Optional eval on update boundaries (rank0 only)
            if int(getattr(args, "eval_every", 0)) > 0 and _is_main(rank):
                if (update_idx % int(args.eval_every)) == 0:
                    eval_bs = int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else int(args.batch_size)
                    eval_model = model.module if is_ddp else model
                    ev_loss, ev_bpc = _eval_model(
                        model=eval_model,
                        tok=tok,
                        lines=eval_lines,
                        device=dev,
                        seq_len=int(args.seq_len),
                        batch_size=int(eval_bs),
                        pack_chars=int(args.pack_chars),
                        pack_mode=str(args.pack_mode),
                        force_eos=bool(args.force_eos),
                        autocast_dtype=autocast_dtype,
                        seed=int(args.eval_seed),
                        batches=int(args.eval_batches),
                    )
                    gstep = int(getattr(args, "step_offset", 0)) + int(step)
                    print(
                        f"== eval == update={update_idx} step={gstep} "
                        f"eval_loss={ev_loss:.4f} eval_bpc≈{ev_bpc:.3f} "
                        f"(batches={int(args.eval_batches)} bs={eval_bs} seed={int(args.eval_seed)})"
                    )

                    # Optional best checkpoint based on eval
                    if (
                        args.save_dir
                        and str(getattr(args, "save_best_metric", "none")) != "none"
                        and str(getattr(args, "save_best_on", "train")) == "eval"
                    ):
                        cur = float(ev_bpc) if args.save_best_metric == "bpc" else float(ev_loss)
                        if cur < (best_metric - float(getattr(args, "save_best_min_delta", 0.0))):
                            best_metric = cur
                            best_source = f"eval update={update_idx} step={gstep} eval_loss={ev_loss:.6f} eval_bpc={ev_bpc:.6f}"
                            save_root = Path(args.save_dir)
                            best_dir = save_root / f"best_{args.save_best_metric}"
                            to_save = model.module if is_ddp else model
                            _save_pretrained_cpu(
                                to_save=to_save,
                                save_dir=best_dir,
                                meta_lines=meta_lines
                                + [
                                    f"best_metric={args.save_best_metric}",
                                    f"best_value={best_metric}",
                                    f"best_step={gstep}",
                                    f"best_source={best_source}",
                                ],
                            )
                            print(
                                f"== saved_best ==\nmetric = {args.save_best_metric}\nvalue  = {best_metric}\nmodel_dir = {best_dir}\n"
                            )

        if _is_main(rank) and (step % args.print_every == 0 or step == args.steps - 1):
            elapsed = time.perf_counter() - t0
            toks = (step + 1) * args.batch_size * args.seq_len
            # bits/char proxy (more comparable across tokenizers than token-level loss)
            bpc = (loss.item() * bpc_factor) if bpc_factor > 0 else float("nan")
            # approximate global active tokens (assume similar per-rank)
            global_active = local_active * world_size
            active_frac = global_active / float(args.batch_size * args.seq_len)
            extra = ""
            if args.log_batch_digest:
                joined = "\n".join(batch)
                h = hashlib.blake2b(joined.encode("utf-8", errors="ignore"), digest_size=4).hexdigest()
                n_chars = len(joined)
                if n_chars > 0:
                    n_digits = sum(ch.isdigit() for ch in joined)
                    n_non_ascii = sum(ord(ch) > 127 for ch in joined)
                    digits_frac = n_digits / n_chars
                    non_ascii_frac = n_non_ascii / n_chars
                else:
                    digits_frac = 0.0
                    non_ascii_frac = 0.0
                extra = f" batch={h} chars={n_chars} dig={digits_frac:.3f} nonascii={non_ascii_frac:.3f}"
            # LR (safe to append at end; extract_loss_curve.py regex still matches prefix)
            extra = f"{extra} lr={opt.param_groups[0]['lr']:.2e}"

            # Optional best checkpoint (saved on print points only)
            if (
                args.save_dir
                and str(getattr(args, "save_best_metric", "none")) != "none"
                and str(getattr(args, "save_best_on", "train")) == "train"
            ):
                cur = float(bpc) if args.save_best_metric == "bpc" else float(loss.item())
                if cur < (best_metric - float(getattr(args, "save_best_min_delta", 0.0))):
                    best_metric = cur
                    best_source = f"train step={int(getattr(args, 'step_offset', 0))+step} loss={loss.item():.6f} bpc={bpc:.6f}"
                    save_root = Path(args.save_dir)
                    best_dir = save_root / f"best_{args.save_best_metric}"
                    to_save = model.module if is_ddp else model
                    _save_pretrained_cpu(
                        to_save=to_save,
                        save_dir=best_dir,
                        meta_lines=meta_lines
                        + [
                            f"best_metric={args.save_best_metric}",
                            f"best_value={best_metric}",
                            f"best_step={(int(args.step_offset)+step)}",
                            f"best_source={best_source}",
                        ],
                    )
                    print(f"== saved_best ==\nmetric = {args.save_best_metric}\nvalue  = {best_metric}\nmodel_dir = {best_dir}\n")

            gstep = int(getattr(args, "step_offset", 0)) + int(step)
            print(
                f"step={gstep:04d} loss={loss.item():.4f} bpc≈{bpc:.3f} "
                f"tok/s={toks/elapsed:.1f} active_frac≈{active_frac:.3f}{extra}"
            )

    # 5) quick generate
    if is_ddp:
        model.module.eval()
    else:
        model.eval()
    prompt = str(getattr(args, "gen_prompt", "hello world"))
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    if is_ddp:
        dev = torch.device("cuda", local_rank)
    else:
        dev = torch.device(device)
    input_ids = enc["input_ids"].to(dev)
    with torch.no_grad():
        gen_model = model.module if is_ddp else model
        # If requested, generate from best checkpoint (rank0 only uses disk).
        if bool(getattr(args, "gen_from_best", False)) and args.save_dir and _is_main(rank):
            metric = str(getattr(args, "save_best_metric", "none"))
            best_dir = Path(args.save_dir) / f"best_{metric}"
            if metric != "none" and best_dir.exists():
                gen_model = LlamaForCausalLM.from_pretrained(str(best_dir)).to(dev)
                gen_model.eval()
        do_sample = not bool(getattr(args, "gen_greedy", False))
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=do_sample,
            pad_token_id=int(tok.pad_token_id),
            eos_token_id=int(tok.eos_token_id),
        )
        if do_sample:
            gen_kwargs["temperature"] = float(getattr(args, "gen_temperature", 0.8))
            gen_kwargs["top_p"] = float(getattr(args, "gen_top_p", 0.95))
            top_k = int(getattr(args, "gen_top_k", 0))
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
        rep = float(getattr(args, "gen_repetition_penalty", 1.0))
        if rep != 1.0:
            gen_kwargs["repetition_penalty"] = rep
        nrep = int(getattr(args, "gen_no_repeat_ngram_size", 0))
        if nrep > 0:
            gen_kwargs["no_repeat_ngram_size"] = nrep
        gen = gen_model.generate(**gen_kwargs)
    text = tok.decode(gen[0].tolist(), skip_special_tokens=True)
    if _is_main(rank):
        print()
        print("== generate ==")
        print(f"prompt = {prompt!r}")
        print(f"out    = {text!r}")

    if is_ddp:
        # Ensure all ranks finish (especially generation on rank0) before teardown.
        dist.barrier()

    # 6) optional save (rank0 only)
    if args.save_dir and _is_main(rank):
        save_dir = Path(args.save_dir)
        to_save = model.module if is_ddp else model
        _save_pretrained_cpu(to_save=to_save, save_dir=save_dir, meta_lines=meta_lines + ["final=1"])
        print(f"== saved ==\nmodel_dir = {save_dir}\n")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


