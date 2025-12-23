#!/usr/bin/env python3
"""
Compare "semantic vector density" between two tokenizer+model pairs.

Why this exists:
- Token-level LM loss is not directly comparable across tokenizers.
- Users often ask whether a tokenizer yields "denser" semantic vectors.

We report two kinds of "density":
1) Compression density: tokens-per-character (TPC) and chars-per-token.
2) Vector density (continuous): statistics of vectors (token embeddings and pooled
   sentence embeddings) such as:
   - near-zero fraction (|x| < eps)
   - mean L1/L2 ratio (higher => more evenly distributed magnitude, i.e. "denser")
   - effective rank (participation ratio) of covariance
   - cosine anisotropy (mean pairwise cosine among random pairs)

Notes:
- Spaces across two separately trained models are not aligned, so we compare
  distributional statistics, not per-example cosine between model A and B.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaForCausalLM


def _read_lines(path: Path, max_lines: int) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as r:
        for line in r:
            s = line.strip()
            if not s:
                continue
            out.append(s)
            if max_lines > 0 and len(out) >= max_lines:
                break
    if not out:
        raise ValueError(f"empty corpus: {path}")
    return out


def _pack_text(lines: list[str], rng: random.Random, target_chars: int) -> str:
    if target_chars <= 0:
        return rng.choice(lines)
    parts: list[str] = []
    n = 0
    while n < target_chars and len(parts) < 10_000:
        s = rng.choice(lines)
        parts.append(s)
        n += len(s) + 1
    return "\n".join(parts)


def _estimate_tpc(tok, lines: list[str], max_lines: int, batch_size: int = 4096) -> float:
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


@dataclass
class VectorStats:
    n: int
    d: int
    mean_l2: float
    mean_l1_over_sqrt_d_l2: float
    near_zero_frac: float
    eff_rank: float
    mean_pair_cos: float
    std_pair_cos: float


def _vector_stats(x: torch.Tensor, *, eps: float, pair_samples: int, seed: int) -> VectorStats:
    """
    x: [N, D] float tensor on CPU
    """
    if x.ndim != 2:
        raise ValueError(f"expected [N,D], got {tuple(x.shape)}")
    n, d = x.shape
    if n < 2:
        raise ValueError("need at least 2 vectors")
    x = x.float()
    abs_x = x.abs()
    l2 = x.norm(dim=-1) + 1e-12
    l1 = abs_x.sum(dim=-1)
    l1_over = (l1 / (math.sqrt(d) * l2)).mean().item()
    near_zero = (abs_x < eps).float().mean().item()
    mean_l2 = l2.mean().item()

    # effective rank (participation ratio) of covariance
    xc = x - x.mean(dim=0, keepdim=True)
    cov = (xc.T @ xc) / float(n)  # [D,D]
    # symmetric eigendecomposition
    evals = torch.linalg.eigvalsh(cov).clamp_min(0)
    tr = evals.sum().item()
    sq = (evals * evals).sum().item()
    eff_rank = (tr * tr / sq) if sq > 0 else float("nan")

    # mean pairwise cosine for random pairs (anisotropy proxy)
    rng = random.Random(seed)
    idx = [rng.randrange(0, n) for _ in range(2 * pair_samples)]
    a = torch.tensor(idx[0::2], dtype=torch.long)
    b = torch.tensor(idx[1::2], dtype=torch.long)
    xa = x[a]
    xb = x[b]
    xa = xa / (xa.norm(dim=-1, keepdim=True) + 1e-12)
    xb = xb / (xb.norm(dim=-1, keepdim=True) + 1e-12)
    cos = (xa * xb).sum(dim=-1)
    mean_pair = cos.mean().item()
    std_pair = cos.std(unbiased=False).item()

    return VectorStats(
        n=n,
        d=d,
        mean_l2=mean_l2,
        mean_l1_over_sqrt_d_l2=l1_over,
        near_zero_frac=near_zero,
        eff_rank=eff_rank,
        mean_pair_cos=mean_pair,
        std_pair_cos=std_pair,
    )


def _pooled_sentence_vectors(
    *,
    model: LlamaForCausalLM,
    tok,
    lines: list[str],
    num_samples: int,
    pack_chars: int,
    seq_len: int,
    batch_size: int,
    device: str,
    seed: int,
) -> torch.Tensor:
    rng = random.Random(seed)
    model.eval()
    dev = torch.device(device)
    vecs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            bs = min(batch_size, num_samples - i)
            batch = [_pack_text(lines, rng, pack_chars) for _ in range(bs)]
            enc = tok(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(dev)
            attention_mask = enc["attention_mask"].to(dev)
            out = model.model(input_ids=input_ids, attention_mask=attention_mask)
            h = out.last_hidden_state  # [B,T,D]
            mask = attention_mask.unsqueeze(-1).to(h.dtype)  # [B,T,1]
            pooled = (h * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1.0))
            vecs.append(pooled.detach().cpu())
    return torch.cat(vecs, dim=0)


def _report(name: str, stats: VectorStats) -> None:
    print(f"== {name} ==")
    print(f"n={stats.n} d={stats.d}")
    print(f"mean_l2                 = {stats.mean_l2:.4f}")
    print(f"mean_L1/(sqrt(d)*L2)    = {stats.mean_l1_over_sqrt_d_l2:.4f}  (higher => 'denser')")
    print(f"near_zero_frac(|x|<eps) = {stats.near_zero_frac:.6f}")
    print(f"effective_rank(PR)      = {stats.eff_rank:.1f}")
    print(f"mean_pair_cos           = {stats.mean_pair_cos:.4f}  (closer to 0 => more isotropic)")
    print(f"std_pair_cos            = {stats.std_pair_cos:.4f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_file", type=str, required=True)
    ap.add_argument("--corpus_max_lines", type=int, default=50000)
    ap.add_argument("--tpc_estimate_lines", type=int, default=20000)

    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--pack_chars", type=int, default=3000)
    ap.add_argument("--num_samples", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--pair_samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--a_name", type=str, default="A")
    ap.add_argument("--a_model_dir", type=str, required=True)
    ap.add_argument("--a_tokenizer_dir", type=str, required=True)

    ap.add_argument("--b_name", type=str, default="B")
    ap.add_argument("--b_model_dir", type=str, required=True)
    ap.add_argument("--b_tokenizer_dir", type=str, required=True)

    ap.add_argument("--out_csv", type=str, default="")
    args = ap.parse_args()

    corpus_file = Path(args.corpus_file)
    lines = _read_lines(corpus_file, int(args.corpus_max_lines))

    def one(name: str, model_dir: str, tok_dir: str, seed: int):
        tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
        model = LlamaForCausalLM.from_pretrained(model_dir)
        model.to(torch.device(args.device))

        tpc = _estimate_tpc(tok, lines, int(args.tpc_estimate_lines))
        cpt = (1.0 / tpc) if tpc > 0 else float("inf")

        # token embedding table stats
        emb = model.get_input_embeddings().weight.detach().cpu()
        emb_stats = _vector_stats(emb, eps=float(args.eps), pair_samples=int(args.pair_samples), seed=seed)

        # pooled sentence vector stats
        sent = _pooled_sentence_vectors(
            model=model,
            tok=tok,
            lines=lines,
            num_samples=int(args.num_samples),
            pack_chars=int(args.pack_chars),
            seq_len=int(args.seq_len),
            batch_size=int(args.batch_size),
            device=str(args.device),
            seed=seed,
        )
        sent_stats = _vector_stats(sent, eps=float(args.eps), pair_samples=int(args.pair_samples), seed=seed + 1)

        return tpc, cpt, emb_stats, sent_stats

    print("== semantic vector density comparison ==")
    print("corpus_file =", corpus_file)
    print("corpus_lines=", len(lines))
    print("seq_len     =", args.seq_len)
    print("pack_chars  =", args.pack_chars)
    print("num_samples =", args.num_samples)
    print("eps         =", args.eps)
    print()

    a = one(args.a_name, args.a_model_dir, args.a_tokenizer_dir, int(args.seed))
    b = one(args.b_name, args.b_model_dir, args.b_tokenizer_dir, int(args.seed) + 123)

    for name, (tpc, cpt, emb_stats, sent_stats) in [
        (args.a_name, a),
        (args.b_name, b),
    ]:
        print(f"## {name}")
        print(f"TPC (tok/char)       = {tpc:.6f}")
        print(f"chars/token (1/TPC)  = {cpt:.3f}")
        print()
        _report(f"{name} token-embedding vectors", emb_stats)
        _report(f"{name} pooled sentence vectors", sent_stats)

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "name",
                    "tpc",
                    "chars_per_token",
                    "kind",
                    "n",
                    "d",
                    "mean_l2",
                    "mean_l1_over_sqrt_d_l2",
                    "near_zero_frac",
                    "effective_rank",
                    "mean_pair_cos",
                    "std_pair_cos",
                ]
            )
            for name, (tpc, cpt, emb_stats, sent_stats) in [
                (args.a_name, a),
                (args.b_name, b),
            ]:
                for kind, s in [("token_embedding", emb_stats), ("pooled_sentence", sent_stats)]:
                    w.writerow(
                        [
                            name,
                            tpc,
                            cpt,
                            kind,
                            s.n,
                            s.d,
                            s.mean_l2,
                            s.mean_l1_over_sqrt_d_l2,
                            s.near_zero_frac,
                            s.eff_rank,
                            s.mean_pair_cos,
                            s.std_pair_cos,
                        ]
                    )
        print("wrote", out)


if __name__ == "__main__":
    main()


