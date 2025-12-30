#!/usr/bin/env python3
"""
Benchmark generation throughput in *characters per second*.

Motivation:
- Autoregressive decoding cost is per token.
- If Length-MAX has larger chars/token, it can yield higher chars/s even when tokens/s is similar.

This script loads two (tokenizer, model) pairs and measures:
- tokens/s (generated tokens per second)
- chars/s (decoded characters per second)
- chars/token (decoded characters per generated token)

Note:
- This is a pragmatic throughput benchmark; quality varies across models.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaForCausalLM


def _sync_if_cuda(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _load_model(model_dir: str, dev: torch.device, dtype: str):
    model = LlamaForCausalLM.from_pretrained(model_dir)
    if dtype == "bf16":
        model = model.to(dev, dtype=torch.bfloat16)
    elif dtype == "fp16":
        model = model.to(dev, dtype=torch.float16)
    else:
        model = model.to(dev)
    model.eval()
    return model


def _run_one(
    *,
    name: str,
    tok,
    model,
    dev: torch.device,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    # Warmup
    for _ in range(warmup):
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(dev)
        with torch.no_grad():
            _ = model.generate(input_ids=input_ids, max_new_tokens=8)

    tok_s_list: list[float] = []
    char_s_list: list[float] = []
    cpt_list: list[float] = []

    for _ in range(repeats):
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(dev)
        in_len = int(input_ids.shape[1])
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            eos_token_id=int(tok.eos_token_id) if tok.eos_token_id is not None else None,
            pad_token_id=int(tok.pad_token_id) if tok.pad_token_id is not None else None,
        )
        if do_sample:
            gen_kwargs.update(dict(temperature=float(temperature), top_p=float(top_p)))
        if float(repetition_penalty) != 1.0:
            gen_kwargs["repetition_penalty"] = float(repetition_penalty)
        if int(no_repeat_ngram_size) > 0:
            gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)

        _sync_if_cuda(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**gen_kwargs)
        _sync_if_cuda(dev)
        dt = time.perf_counter() - t0

        out_ids = out[0].tolist()
        gen_ids = out_ids[in_len:]
        gen_tokens = len(gen_ids)
        gen_text = tok.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        gen_chars = len(gen_text)

        tok_s = gen_tokens / dt if dt > 0 else float("nan")
        char_s = gen_chars / dt if dt > 0 else float("nan")
        cpt = (gen_chars / gen_tokens) if gen_tokens > 0 else float("nan")
        tok_s_list.append(tok_s)
        char_s_list.append(char_s)
        cpt_list.append(cpt)

    def mean(x: list[float]) -> float:
        return float(statistics.mean(x)) if x else float("nan")

    def stdev(x: list[float]) -> float:
        return float(statistics.pstdev(x)) if x else float("nan")

    out = dict(
        name=name,
        tok_s_mean=mean(tok_s_list),
        tok_s_std=stdev(tok_s_list),
        char_s_mean=mean(char_s_list),
        char_s_std=stdev(char_s_list),
        chars_per_token_mean=mean(cpt_list),
        chars_per_token_std=stdev(cpt_list),
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--len_tok_dir", type=str, required=True)
    ap.add_argument("--len_model_dir", type=str, required=True)
    ap.add_argument("--bpe_tok_dir", type=str, required=True)
    ap.add_argument("--bpe_model_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--prompt", type=str, default="= Valkyria Chronicles III =")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.12)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--out_csv", type=str, default="")
    args = ap.parse_args()

    dev = torch.device(args.device)

    print("[load] tokenizers ...")
    len_tok = AutoTokenizer.from_pretrained(args.len_tok_dir, trust_remote_code=True)
    bpe_tok = AutoTokenizer.from_pretrained(args.bpe_tok_dir, use_fast=True)

    print("[load] models ...")
    len_model = _load_model(args.len_model_dir, dev, str(args.dtype))
    bpe_model = _load_model(args.bpe_model_dir, dev, str(args.dtype))

    prompt = str(args.prompt)
    do_sample = bool(args.do_sample)

    res_len = _run_one(
        name="Length-MAX",
        tok=len_tok,
        model=len_model,
        dev=dev,
        prompt=prompt,
        max_new_tokens=int(args.max_new_tokens),
        do_sample=do_sample,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        no_repeat_ngram_size=int(args.no_repeat_ngram_size),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )
    res_bpe = _run_one(
        name="BPE",
        tok=bpe_tok,
        model=bpe_model,
        dev=dev,
        prompt=prompt,
        max_new_tokens=int(args.max_new_tokens),
        do_sample=do_sample,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        no_repeat_ngram_size=int(args.no_repeat_ngram_size),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )

    def show(r: dict[str, float]) -> None:
        print(
            f"[{r['name']}] tok/s={r['tok_s_mean']:.1f}±{r['tok_s_std']:.1f}  "
            f"chars/s={r['char_s_mean']:.1f}±{r['char_s_std']:.1f}  "
            f"chars/token={r['chars_per_token_mean']:.3f}±{r['chars_per_token_std']:.3f}"
        )

    show(res_len)
    show(res_bpe)

    if str(args.out_csv).strip():
        out_csv = Path(str(args.out_csv).strip())
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            import csv

            w = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "tok_s_mean",
                    "tok_s_std",
                    "char_s_mean",
                    "char_s_std",
                    "chars_per_token_mean",
                    "chars_per_token_std",
                ],
            )
            w.writeheader()
            w.writerow(res_len)
            w.writerow(res_bpe)
        print("wrote", out_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())







