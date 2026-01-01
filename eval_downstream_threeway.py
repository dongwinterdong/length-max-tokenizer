#!/usr/bin/env python3
"""
Downstream + long-context evaluations for three tokenizer-produced models:
  - BPE (baseline)
  - SuperBPE
  - Length-MAX

Design goals:
- Focus on *fair* comparisons across tokenizers:
  - Use multiple-choice scoring via conditional log-likelihood (accuracy is comparable).
  - For long-context, include an offline task that stresses token-budget truncation.
- Track tokenizer/task adaptation signals:
  - trunc_rate (prompt got truncated to fit model token window)
  - unk_rate (fraction of <unk> among active tokens)
  - avg prompt tokens / chars

Tasks included:
1) hellaswag (validation)        [requires internet on first run; via `datasets`]
2) piqa (validation)             [datasets; uses `trust_remote_code=True`]
3) winogrande_xl (validation)    [datasets]
4) wt103_longmc (offline)        [Wikitext-103 test set; long-context multiple-choice continuation]
5) passkey_mc (offline synthetic)[long-context "key at start; question at end" MC, measures token-budget retention]

NOTE:
These models are small (58.5M) and trained on WT103 only in this repo, so absolute downstream
scores may be low; the intent here is *relative* comparison and long-context truncation behavior.
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


def _maybe_space(prefix: str, s: str) -> str:
    """Append s as a continuation; ensure a single space if prefix doesn't end with whitespace."""
    if not s:
        return s
    if not prefix:
        return s
    if prefix[-1].isspace():
        return s.lstrip("\n")  # keep newlines as-is; but avoid accidental leading newline duplication
    # default: insert one space
    return " " + s.lstrip()


def _count_unk(ids: list[int], *, unk_id: int | None, pad_id: int | None = None) -> int:
    if unk_id is None:
        return 0
    out = 0
    for t in ids:
        if pad_id is not None and t == pad_id:
            continue
        if t == unk_id:
            out += 1
    return out


@torch.no_grad()
def score_options_conditional(
    *,
    model: LlamaForCausalLM,
    tok,
    prompt: str,
    options: list[str],
    device: torch.device,
    seq_len: int,
    precision: str,
    score_norm: str,
) -> tuple[list[float], dict]:
    """
    Score each option by log P(option | prompt).

    By default we compute the SUM of token log-probs over the option tokens.
    For fairer comparison when options have different lengths, we support normalization:
      - sum        : sum log-prob over option tokens (classic)
      - mean_token : average log-prob per option token
      - mean_char  : average log-prob per option character (tokenizer-invariant length unit)
      - mean_byte  : average log-prob per option UTF-8 byte

    Truncation policy:
    - Keep ALL option tokens.
    - If (BOS + prompt + option) exceeds seq_len, truncate prompt from the LEFT (keep the suffix).
    """
    if not options:
        raise ValueError("options is empty")
    norm = str(score_norm).strip().lower()
    if norm not in ("sum", "mean_token", "mean_char", "mean_byte"):
        raise ValueError(f"unknown score_norm: {score_norm!r}")

    dtype = _is_float_dtype(precision)
    bos = int(tok.bos_token_id) if getattr(tok, "bos_token_id", None) is not None else None
    pad = int(tok.pad_token_id) if getattr(tok, "pad_token_id", None) is not None else 0
    unk = int(getattr(tok, "unk_token_id", -1)) if getattr(tok, "unk_token_id", None) is not None else None

    prompt_ids_full = tok.encode(prompt, add_special_tokens=False)
    opt_ids_list = [tok.encode(o, add_special_tokens=False) for o in options]

    # Truncate prompt ONCE based on the longest option so all options see the same context.
    max_opt_len = max(len(x) for x in opt_ids_list)
    max_prompt = int(seq_len) - (1 if bos is not None else 0) - int(max_opt_len)
    if max_prompt < 0:
        # Degenerate case: an option doesn't fit. Keep option suffix and drop prompt entirely.
        max_prompt = 0
        opt_ids_list = [x[-(int(seq_len) - (1 if bos is not None else 0)) :] for x in opt_ids_list]
        max_opt_len = max(len(x) for x in opt_ids_list)

    trunc = len(prompt_ids_full) > int(max_prompt)
    prompt_ids = prompt_ids_full[-max_prompt:] if max_prompt > 0 else []

    # Build a padded batch.
    input_ids_batch: list[list[int]] = []
    opt_starts: list[int] = []  # start index of option tokens in input_ids
    for opt_ids in opt_ids_list:
        ids: list[int] = []
        if bos is not None:
            ids.append(bos)
        ids.extend(prompt_ids)
        opt_start = len(ids)
        ids.extend(opt_ids)
        input_ids_batch.append(ids)
        opt_starts.append(opt_start)

    max_len = max(len(x) for x in input_ids_batch)
    max_len = min(max_len, int(seq_len))

    input_ids = torch.full((len(options), max_len), pad, dtype=torch.long)
    attn = torch.zeros((len(options), max_len), dtype=torch.long)

    total_unk = 0
    total_active = 0
    for i, ids in enumerate(input_ids_batch):
        ids = ids[:max_len]
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, : len(ids)] = 1
        total_active += len(ids)
        total_unk += _count_unk(ids, unk_id=unk, pad_id=pad)

    input_ids = input_ids.to(device, non_blocking=True)
    attn = attn.to(device, non_blocking=True)

    if device.type == "cuda":
        torch.cuda.synchronize()

    if dtype is None:
        out = model(input_ids=input_ids, attention_mask=attn)
    else:
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            out = model(input_ids=input_ids, attention_mask=attn)

    logits = out.logits  # [B, T, V]
    logp = torch.log_softmax(logits, dim=-1)

    scores: list[float] = []
    opt_lens_token = [max(1, len(x)) for x in opt_ids_list]
    opt_lens_char = [max(1, len(s)) for s in options]
    opt_lens_byte = [max(1, len(str(s).encode("utf-8", errors="ignore"))) for s in options]

    for i, opt_ids in enumerate(opt_ids_list):
        start = int(opt_starts[i])
        # positions predicting the option tokens
        # token at position j is predicted at logp[:, j-1, token]
        s = 0.0
        for k, tid in enumerate(opt_ids):
            pos = start + k
            if pos <= 0:
                continue
            if pos - 1 >= logp.size(1):
                break
            s += float(logp[i, pos - 1, int(tid)].item())
        if norm == "sum":
            scores.append(s)
        elif norm == "mean_token":
            scores.append(s / float(opt_lens_token[i]))
        elif norm == "mean_char":
            scores.append(s / float(opt_lens_char[i]))
        elif norm == "mean_byte":
            scores.append(s / float(opt_lens_byte[i]))
        else:
            raise AssertionError(norm)

    info = {
        "prompt_chars": len(prompt),
        "prompt_tokens_full": len(prompt_ids_full),
        "prompt_tokens_used": len(prompt_ids),
        "max_option_tokens": int(max_opt_len),
        "trunc_any": bool(trunc),
        "trunc_rate_over_options": 1.0 if trunc else 0.0,
        "unk_rate": (float(total_unk) / float(max(1, total_active))) if unk is not None else float("nan"),
        "max_len_tokens": int(max_len),
        "score_norm": norm,
    }
    return scores, info


@torch.no_grad()
def score_full_text(
    *,
    model: LlamaForCausalLM,
    tok,
    text: str,
    device: torch.device,
    seq_len: int,
    precision: str,
) -> tuple[float, dict]:
    """Score a full text by log P(text) (sum over tokens), truncating from left to fit seq_len."""
    dtype = _is_float_dtype(precision)
    bos = int(tok.bos_token_id) if getattr(tok, "bos_token_id", None) is not None else None
    pad = int(tok.pad_token_id) if getattr(tok, "pad_token_id", None) is not None else 0
    unk = int(getattr(tok, "unk_token_id", -1)) if getattr(tok, "unk_token_id", None) is not None else None

    ids = tok.encode(text, add_special_tokens=False)
    max_body = int(seq_len) - (1 if bos is not None else 0)
    trunc = len(ids) > max_body
    ids = ids[-max_body:] if max_body > 0 else []
    input_ids_list: list[int] = []
    if bos is not None:
        input_ids_list.append(bos)
    input_ids_list.extend(ids)

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids, dtype=torch.long, device=device)

    if dtype is None:
        out = model(input_ids=input_ids, attention_mask=attn)
    else:
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            out = model(input_ids=input_ids, attention_mask=attn)
    logp = torch.log_softmax(out.logits, dim=-1)

    # Sum log-prob of each token given previous token (exclude BOS).
    s = 0.0
    for t in range(1, input_ids.size(1)):
        tid = int(input_ids[0, t].item())
        s += float(logp[0, t - 1, tid].item())

    total_unk = _count_unk(input_ids_list, unk_id=unk, pad_id=pad)
    info = {
        "chars": len(text),
        "tokens_full": len(tok.encode(text, add_special_tokens=False)),
        "tokens_used": int(input_ids.size(1)),
        "trunc": bool(trunc),
        "unk_rate": (float(total_unk) / float(max(1, input_ids.size(1)))) if unk is not None else float("nan"),
    }
    return s, info


@dataclass
class Agg:
    n: int = 0
    correct: int = 0
    trunc: int = 0
    prompt_tokens_full_sum: int = 0
    prompt_tokens_used_sum: int = 0
    prompt_tokens_used_n: int = 0
    prompt_chars_sum: int = 0
    unk_sum: float = 0.0
    needle_present_sum: int = 0
    needle_present_n: int = 0

    def add(self, *, ok: bool, info: dict) -> None:
        self.n += 1
        self.correct += 1 if ok else 0
        self.trunc += 1 if bool(info.get("trunc_any", False) or info.get("trunc", False)) else 0
        self.prompt_tokens_full_sum += int(info.get("prompt_tokens_full", info.get("tokens_full", 0)))
        if "prompt_tokens_used" in info:
            try:
                self.prompt_tokens_used_sum += int(info.get("prompt_tokens_used", 0))
                self.prompt_tokens_used_n += 1
            except Exception:
                pass
        self.prompt_chars_sum += int(info.get("prompt_chars", info.get("chars", 0)))
        u = info.get("unk_rate", float("nan"))
        if isinstance(u, float) and math.isfinite(u):
            self.unk_sum += float(u)
        if "needle_present" in info:
            self.needle_present_n += 1
            self.needle_present_sum += 1 if bool(info.get("needle_present", False)) else 0

    def summary(self) -> dict:
        n = max(1, self.n)
        used_n = max(1, self.prompt_tokens_used_n)
        needle_n = max(1, self.needle_present_n)
        return {
            "n": int(self.n),
            "acc": float(self.correct) / float(n),
            "trunc_rate": float(self.trunc) / float(n),
            "avg_prompt_tokens_full": float(self.prompt_tokens_full_sum) / float(n),
            "avg_prompt_tokens_used": (
                float(self.prompt_tokens_used_sum) / float(used_n) if self.prompt_tokens_used_n > 0 else float("nan")
            ),
            "avg_prompt_chars": float(self.prompt_chars_sum) / float(n),
            "avg_unk_rate": float(self.unk_sum) / float(n),
            "needle_present_rate": (
                float(self.needle_present_sum) / float(needle_n) if self.needle_present_n > 0 else float("nan")
            ),
        }


def _read_lines(path: Path) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            out.append(line.rstrip("\n"))
    if not out:
        raise ValueError(f"empty corpus: {path}")
    return out


def _pack_contiguous(lines: list[str], start: int, target_chars: int) -> str:
    if target_chars <= 0:
        return lines[start % len(lines)]
    parts: list[str] = []
    n = 0
    i = start % len(lines)
    while n < target_chars and len(parts) < 100_000:
        s = lines[i]
        parts.append(s)
        n += len(s) + 1
        i += 1
        if i >= len(lines):
            i = 0
    return "\n".join(parts)


def _contains_subseq(hay: list[int], needle: list[int]) -> bool:
    if not needle:
        return False
    if len(needle) > len(hay):
        return False
    m = len(needle)
    for i in range(0, len(hay) - m + 1):
        if hay[i : i + m] == needle:
            return True
    return False


def _insert_needle_into_filler(*, filler: str, needle_line: str, pos_frac: float) -> str:
    """
    Insert `needle_line` into `filler` at approximately pos_frac of its length.
    Attempts to snap to a nearby whitespace boundary to avoid splitting words.
    """
    s = str(filler)
    needle = str(needle_line).strip("\n")
    if not needle:
        return s
    f = float(pos_frac)
    if not math.isfinite(f):
        f = 0.5
    f = max(0.0, min(1.0, f))

    n = len(s)
    if n <= 0:
        return needle + "\n"
    ins = int(n * f)
    ins = max(0, min(n, ins))

    # Snap to nearest whitespace boundary within a small window.
    best = ins
    for d in range(0, 64):
        for j in (ins - d, ins + d):
            if 0 <= j < n and s[j].isspace():
                best = j
                d = 10**9
                break
    ins = best

    # Keep a newline around the needle line.
    return s[:ins] + "\n" + needle + "\n" + s[ins:]


def _safe_decode(tok, ids: list[int]) -> str:
    """Decode helper compatible with both fast and python tokenizers."""
    try:
        return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except TypeError:
        return tok.decode(ids, skip_special_tokens=True)


def eval_wt103_longmc(
    *,
    model,
    tok,
    device: torch.device,
    precision: str,
    seq_len: int,
    score_norm: str,
    corpus_path: Path,
    num_examples: int,
    context_chars: int,
    cont_chars: int,
    num_choices: int,
    seed: int,
) -> dict:
    """
    Offline long-context MC continuation:
      prompt = first context_chars of a packed WT103 text
      options = [true continuation (next cont_chars)] + (num_choices-1) distractors from other locations
    """
    rng = random.Random(int(seed))
    lines = _read_lines(corpus_path)
    agg = Agg()

    for _ in range(int(num_examples)):
        # Make a long sample containing context+continuation.
        start = rng.randrange(len(lines))
        packed = _pack_contiguous(lines, start, int(context_chars + cont_chars + 64))
        if len(packed) < context_chars + cont_chars + 1:
            continue
        prompt = packed[:context_chars]
        gold = packed[context_chars : context_chars + cont_chars]

        # Distractors: same length slices from other random places.
        opts = [gold]
        while len(opts) < int(num_choices):
            s2 = rng.randrange(len(lines))
            packed2 = _pack_contiguous(lines, s2, int(cont_chars + 64))
            if len(packed2) < cont_chars:
                continue
            cand = packed2[:cont_chars]
            if cand == gold:
                continue
            opts.append(cand)

        # Shuffle choices
        order = list(range(len(opts)))
        rng.shuffle(order)
        opts_shuf = [opts[i] for i in order]
        label = order.index(0)

        scores, info = score_options_conditional(
            model=model,
            tok=tok,
            prompt=prompt,
            options=opts_shuf,
            device=device,
            seq_len=int(seq_len),
            precision=str(precision),
            score_norm=str(score_norm),
        )
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        agg.add(ok=(pred == int(label)), info=info)

    out = agg.summary()
    out.update(
        {
            "task": "wt103_longmc",
            "context_chars": int(context_chars),
            "cont_chars": int(cont_chars),
            "num_choices": int(num_choices),
        }
    )
    return out


def eval_passkey_mc(
    *,
    model,
    tok,
    device: torch.device,
    precision: str,
    seq_len: int,
    score_norm: str,
    num_examples: int,
    total_chars: int,
    num_choices: int,
    seed: int,
) -> dict:
    """
    Synthetic passkey multiple-choice:
      - Key is placed at the very beginning.
      - Question is at the end.
      - If prompt is truncated (token budget), the key is lost => accuracy should drop to chance.
    """
    rng = random.Random(int(seed))
    agg = Agg()

    for _ in range(int(num_examples)):
        key = f"{rng.randrange(10000, 99999)}"
        header = f"PASSKEY: {key}\n"
        tail = "\nWhat is the passkey? Answer:"

        # Fill to total_chars with a simple, in-domain-ish filler.
        filler_target = max(0, int(total_chars) - len(header) - len(tail))
        filler_unit = "This is filler text about Wikipedia and Wikitext. "
        reps = (filler_target // len(filler_unit)) + 1
        filler = (filler_unit * reps)[:filler_target]
        prompt = header + filler + tail

        # Build options: correct + distractors
        opts = [key]
        while len(opts) < int(num_choices):
            d = f"{rng.randrange(10000, 99999)}"
            if d != key:
                opts.append(d)
        order = list(range(len(opts)))
        rng.shuffle(order)
        opts_shuf = [opts[i] for i in order]
        label = order.index(0)

        # Add a leading space to make them continuations after "Answer:"
        opts_shuf = [" " + o for o in opts_shuf]

        scores, info = score_options_conditional(
            model=model,
            tok=tok,
            prompt=prompt,
            options=opts_shuf,
            device=device,
            seq_len=int(seq_len),
            precision=str(precision),
            score_norm=str(score_norm),
        )
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        agg.add(ok=(pred == int(label)), info=info)

    out = agg.summary()
    out.update({"task": "passkey_mc", "total_chars": int(total_chars), "num_choices": int(num_choices)})
    return out


def eval_wt103_passkey_mc(
    *,
    model,
    tok,
    device: torch.device,
    precision: str,
    seq_len: int,
    score_norm: str,
    num_examples: int,
    total_chars: int,
    num_choices: int,
    seed: int,
    corpus_path: Path,
) -> dict:
    """
    Passkey-MC with WT103 in-domain filler.

    This is more representative than synthetic filler because tokenizer compression differs
    substantially on natural WT103 text, and it better matches the model's training domain.
    """
    rng = random.Random(int(seed))
    lines = _read_lines(Path(corpus_path))
    agg = Agg()

    for _ in range(int(num_examples)):
        # Use a longer, low-collision key (hex) and ensure it's not present in filler.
        key = f"{rng.randrange(16**16):016x}"
        header = f"PASSKEY: {key}\n"
        tail = "\nWhat is the passkey? Answer:"

        filler_target = max(0, int(total_chars) - len(header) - len(tail))
        start = rng.randrange(len(lines))
        filler = _pack_contiguous(lines, start, int(filler_target + 64))
        filler = filler[:filler_target]
        # Rare but safe: avoid accidental key occurrence in filler.
        if key in filler:
            # retry with a different key once
            key = f"{rng.randrange(16**16):016x}"
            header = f"PASSKEY: {key}\n"
            filler_target = max(0, int(total_chars) - len(header) - len(tail))
            filler = filler[:filler_target]

        prompt = header + filler + tail

        # MC options: correct key + distractors, all same length.
        opts = [key]
        while len(opts) < int(num_choices):
            d = f"{rng.randrange(16**16):016x}"
            if d != key and d not in opts:
                opts.append(d)
        order = list(range(len(opts)))
        rng.shuffle(order)
        opts_shuf = [opts[i] for i in order]
        label = order.index(0)
        opts_shuf = [" " + o for o in opts_shuf]

        scores, info = score_options_conditional(
            model=model,
            tok=tok,
            prompt=prompt,
            options=opts_shuf,
            device=device,
            seq_len=int(seq_len),
            precision=str(precision),
            score_norm=str(score_norm),
        )
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        agg.add(ok=(pred == int(label)), info=info)

    out = agg.summary()
    out.update(
        {
            "task": "wt103_passkey_mc",
            "total_chars": int(total_chars),
            "num_choices": int(num_choices),
            "corpus": str(corpus_path),
        }
    )
    return out


def eval_kv_retrieval_mc(
    *,
    model,
    tok,
    device: torch.device,
    precision: str,
    seq_len: int,
    score_norm: str,
    num_examples: int,
    total_chars: int,
    num_choices: int,
    seed: int,
    corpus_path: Path,
) -> dict:
    """
    Key-Value retrieval MC with WT103 filler:

      KV: name=<name> value=<value>
      (WT103 filler)
      Q: What is the value of <name>? Answer:

    This is a long-context-sensitive retrieval task: if the KV line gets truncated away,
    accuracy should drop toward chance.
    """
    rng = random.Random(int(seed))
    lines = _read_lines(Path(corpus_path))
    agg = Agg()

    for _ in range(int(num_examples)):
        name = f"item{rng.randrange(10000, 99999)}"
        value = f"{rng.randrange(16**16):016x}"
        header = f"KV: name={name} value={value}\n"
        tail = f"\nQ: What is the value of {name}? Answer:"

        filler_target = max(0, int(total_chars) - len(header) - len(tail))
        start = rng.randrange(len(lines))
        filler = _pack_contiguous(lines, start, int(filler_target + 64))
        filler = filler[:filler_target]

        prompt = header + filler + tail

        opts = [value]
        while len(opts) < int(num_choices):
            d = f"{rng.randrange(16**16):016x}"
            if d != value and d not in opts:
                opts.append(d)
        order = list(range(len(opts)))
        rng.shuffle(order)
        opts_shuf = [opts[i] for i in order]
        label = order.index(0)
        opts_shuf = [" " + o for o in opts_shuf]

        scores, info = score_options_conditional(
            model=model,
            tok=tok,
            prompt=prompt,
            options=opts_shuf,
            device=device,
            seq_len=int(seq_len),
            precision=str(precision),
            score_norm=str(score_norm),
        )
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        agg.add(ok=(pred == int(label)), info=info)

    out = agg.summary()
    out.update(
        {
            "task": "kv_retrieval_mc",
            "total_chars": int(total_chars),
            "num_choices": int(num_choices),
            "corpus": str(corpus_path),
        }
    )
    return out


def eval_wt103_passkey_needle_mc(
    *,
    model,
    tok,
    device: torch.device,
    precision: str,
    seq_len: int,
    score_norm: str,
    num_examples: int,
    total_chars: int,
    num_choices: int,
    seed: int,
    corpus_path: Path,
    needle_pos_frac: float,
) -> dict:
    """
    WT103-filler Passkey retrieval where the needle is inserted into the middle of the WT103 filler,
    not fixed at the very beginning. This is closer to realistic long-context QA where the relevant
    fact may appear anywhere in the context.
    """
    rng = random.Random(int(seed))
    lines = _read_lines(Path(corpus_path))
    agg = Agg()

    bos = int(tok.bos_token_id) if getattr(tok, "bos_token_id", None) is not None else None

    for _ in range(int(num_examples)):
        key = f"{rng.randrange(16**16):016x}"
        tail = "\nWhat is the passkey? Answer:"
        needle_line = f"PASSKEY: {key}"

        filler_target = max(0, int(total_chars) - len(needle_line) - len(tail) - 2)  # rough newline budget
        start = rng.randrange(len(lines))
        filler = _pack_contiguous(lines, start, int(filler_target + 64))
        filler = filler[:filler_target]

        filler2 = _insert_needle_into_filler(filler=filler, needle_line=needle_line, pos_frac=float(needle_pos_frac))
        prompt = filler2 + tail

        opts = [key]
        while len(opts) < int(num_choices):
            d = f"{rng.randrange(16**16):016x}"
            if d != key and d not in opts:
                opts.append(d)
        order = list(range(len(opts)))
        rng.shuffle(order)
        opts_shuf = [opts[i] for i in order]
        label = order.index(0)
        opts_shuf = [" " + o for o in opts_shuf]

        # Needle retention in the actually visible prompt window (token-based, left truncation).
        prompt_ids_full = tok.encode(prompt, add_special_tokens=False)
        opt_ids_list = [tok.encode(o, add_special_tokens=False) for o in opts_shuf]
        max_opt_len = max(len(x) for x in opt_ids_list)
        max_prompt = int(seq_len) - (1 if bos is not None else 0) - int(max_opt_len)
        max_prompt = max(0, max_prompt)
        prompt_ids_used = prompt_ids_full[-max_prompt:] if max_prompt > 0 else []

        # Retention check: decode the actually-visible prompt suffix and search for the key string.
        # (Token-ID subsequence checks are NOT reliable for SuperBPE due to cross-boundary merges.)
        prompt_text_used = _safe_decode(tok, prompt_ids_used)
        needle_present = key in prompt_text_used

        scores, info = score_options_conditional(
            model=model,
            tok=tok,
            prompt=prompt,
            options=opts_shuf,
            device=device,
            seq_len=int(seq_len),
            precision=str(precision),
            score_norm=str(score_norm),
        )
        info = dict(info)
        info["needle_present"] = bool(needle_present)

        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        agg.add(ok=(pred == int(label)), info=info)

    out = agg.summary()
    out.update(
        {
            "task": "wt103_passkey_needle_mc",
            "total_chars": int(total_chars),
            "num_choices": int(num_choices),
            "needle_pos_frac": float(needle_pos_frac),
            "corpus": str(corpus_path),
        }
    )
    return out


def eval_kv_needle_mc(
    *,
    model,
    tok,
    device: torch.device,
    precision: str,
    seq_len: int,
    score_norm: str,
    num_examples: int,
    total_chars: int,
    num_choices: int,
    seed: int,
    corpus_path: Path,
    needle_pos_frac: float,
) -> dict:
    """
    WT103-filler KV retrieval where the KV needle is inserted into the middle of WT103 filler.
    """
    rng = random.Random(int(seed))
    lines = _read_lines(Path(corpus_path))
    agg = Agg()

    bos = int(tok.bos_token_id) if getattr(tok, "bos_token_id", None) is not None else None

    for _ in range(int(num_examples)):
        name = f"item{rng.randrange(10000, 99999)}"
        value = f"{rng.randrange(16**16):016x}"
        tail = f"\nQ: What is the value of {name}? Answer:"
        needle_line = f"KV: name={name} value={value}"

        filler_target = max(0, int(total_chars) - len(needle_line) - len(tail) - 2)
        start = rng.randrange(len(lines))
        filler = _pack_contiguous(lines, start, int(filler_target + 64))
        filler = filler[:filler_target]

        filler2 = _insert_needle_into_filler(filler=filler, needle_line=needle_line, pos_frac=float(needle_pos_frac))
        prompt = filler2 + tail

        opts = [value]
        while len(opts) < int(num_choices):
            d = f"{rng.randrange(16**16):016x}"
            if d != value and d not in opts:
                opts.append(d)
        order = list(range(len(opts)))
        rng.shuffle(order)
        opts_shuf = [opts[i] for i in order]
        label = order.index(0)
        opts_shuf = [" " + o for o in opts_shuf]

        prompt_ids_full = tok.encode(prompt, add_special_tokens=False)
        opt_ids_list = [tok.encode(o, add_special_tokens=False) for o in opts_shuf]
        max_opt_len = max(len(x) for x in opt_ids_list)
        max_prompt = int(seq_len) - (1 if bos is not None else 0) - int(max_opt_len)
        max_prompt = max(0, max_prompt)
        prompt_ids_used = prompt_ids_full[-max_prompt:] if max_prompt > 0 else []

        prompt_text_used = _safe_decode(tok, prompt_ids_used)
        needle_present = value in prompt_text_used

        scores, info = score_options_conditional(
            model=model,
            tok=tok,
            prompt=prompt,
            options=opts_shuf,
            device=device,
            seq_len=int(seq_len),
            precision=str(precision),
            score_norm=str(score_norm),
        )
        info = dict(info)
        info["needle_present"] = bool(needle_present)

        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        agg.add(ok=(pred == int(label)), info=info)

    out = agg.summary()
    out.update(
        {
            "task": "kv_needle_mc",
            "total_chars": int(total_chars),
            "num_choices": int(num_choices),
            "needle_pos_frac": float(needle_pos_frac),
            "corpus": str(corpus_path),
        }
    )
    return out


def eval_hellaswag(
    *,
    model,
    tok,
    device: torch.device,
    precision: str,
    seq_len: int,
    score_norm: str,
    max_examples: int,
    seed: int,
) -> dict:
    from datasets import load_dataset

    ds = load_dataset("hellaswag", split="validation")
    rng = random.Random(int(seed))
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: int(max_examples)]

    agg = Agg()
    for i in idxs:
        ex = ds[int(i)]
        prompt = str(ex["ctx"])
        endings = list(ex["endings"])
        options = [_maybe_space(prompt, e) for e in endings]
        label = int(ex["label"])

        scores, info = score_options_conditional(
            model=model,
            tok=tok,
            prompt=prompt,
            options=options,
            device=device,
            seq_len=int(seq_len),
            precision=str(precision),
            score_norm=str(score_norm),
        )
        pred = int(max(range(len(scores)), key=lambda j: scores[j]))
        agg.add(ok=(pred == int(label)), info=info)

    out = agg.summary()
    out.update({"task": "hellaswag", "split": "validation"})
    return out


def eval_lambada_mc(
    *,
    model,
    tok,
    device: torch.device,
    precision: str,
    seq_len: int,
    score_norm: str,
    max_examples: int,
    num_choices: int,
    seed: int,
) -> dict:
    """
    LAMBADA as multiple-choice last-word prediction:
      prompt = text without the final whitespace-separated token
      choices = {true last token} ∪ random last tokens from other examples
    """
    import re

    from datasets import load_dataset

    ds = load_dataset("lambada", split="validation")
    # Precompute candidate pool of last tokens (indexing is more robust than iterating in some envs).
    pat = re.compile(r"^(.*\S)\s+(\S+)$")
    last_tokens: list[str] = []
    for i in range(len(ds)):
        t = str(ds[int(i)]["text"]).rstrip()
        m = pat.match(t)
        if m:
            last_tokens.append(m.group(2))
    if len(last_tokens) < 100:
        raise ValueError(f"unexpected lambada schema: too few last tokens (n={len(last_tokens)})")

    rng = random.Random(int(seed))
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: int(max_examples)]

    agg = Agg()
    for i in idxs:
        ex = ds[int(i)]
        t = str(ex["text"]).rstrip()
        m = pat.match(t)
        if not m:
            continue
        prompt = m.group(1)
        gold = m.group(2)

        opts = [gold]
        # Sample distractors from other examples' last words.
        while len(opts) < int(num_choices):
            d = last_tokens[rng.randrange(len(last_tokens))]
            if d != gold and d not in opts:
                opts.append(d)

        order = list(range(len(opts)))
        rng.shuffle(order)
        opts_shuf = [opts[j] for j in order]
        label = order.index(0)

        # Keep the last word separated by a single space.
        opts_shuf = [" " + o for o in opts_shuf]

        scores, info = score_options_conditional(
            model=model,
            tok=tok,
            prompt=prompt,
            options=opts_shuf,
            device=device,
            seq_len=int(seq_len),
            precision=str(precision),
            score_norm=str(score_norm),
        )
        pred = int(max(range(len(scores)), key=lambda j: scores[j]))
        agg.add(ok=(pred == int(label)), info=info)

    out = agg.summary()
    out.update({"task": "lambada_mc", "split": "validation", "num_choices": int(num_choices)})
    return out


def eval_piqa(
    *,
    model,
    tok,
    device: torch.device,
    precision: str,
    seq_len: int,
    score_norm: str,
    max_examples: int,
    seed: int,
) -> dict:
    from datasets import load_dataset

    ds = load_dataset("piqa", split="validation", trust_remote_code=True)
    rng = random.Random(int(seed))
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: int(max_examples)]

    agg = Agg()
    for i in idxs:
        ex = ds[int(i)]
        prompt = str(ex["goal"]).strip()
        options = [_maybe_space(prompt, str(ex["sol1"]).strip()), _maybe_space(prompt, str(ex["sol2"]).strip())]
        label = int(ex["label"])

        scores, info = score_options_conditional(
            model=model,
            tok=tok,
            prompt=prompt,
            options=options,
            device=device,
            seq_len=int(seq_len),
            precision=str(precision),
            score_norm=str(score_norm),
        )
        pred = int(max(range(len(scores)), key=lambda j: scores[j]))
        agg.add(ok=(pred == int(label)), info=info)

    out = agg.summary()
    out.update({"task": "piqa", "split": "validation"})
    return out


def eval_winogrande(
    *,
    model,
    tok,
    device: torch.device,
    precision: str,
    seq_len: int,
    max_examples: int,
    seed: int,
    config: str,
) -> dict:
    from datasets import load_dataset

    ds = load_dataset("winogrande", config, split="validation")
    rng = random.Random(int(seed))
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: int(max_examples)]

    agg = Agg()
    for i in idxs:
        ex = ds[int(i)]
        sent = str(ex["sentence"])
        opt1 = str(ex["option1"])
        opt2 = str(ex["option2"])
        ans = int(str(ex["answer"])) - 1  # '1' or '2'

        s1 = sent.replace("_", opt1)
        s2 = sent.replace("_", opt2)

        sc1, info1 = score_full_text(model=model, tok=tok, text=s1, device=device, seq_len=int(seq_len), precision=str(precision))
        sc2, info2 = score_full_text(model=model, tok=tok, text=s2, device=device, seq_len=int(seq_len), precision=str(precision))

        pred = 0 if sc1 >= sc2 else 1
        # For aggregation, use info1 (both are similar lengths).
        agg.add(ok=(pred == ans), info={"trunc": info1["trunc"], "tokens_full": info1["tokens_full"], "chars": info1["chars"], "unk_rate": info1["unk_rate"]})

    out = agg.summary()
    out.update({"task": "winogrande", "split": "validation", "config": str(config)})
    return out


def _warmup_model(model: LlamaForCausalLM, *, device: torch.device, precision: str, seq_len: int, seed: int, steps: int) -> None:
    dtype = _is_float_dtype(precision)
    if steps <= 0:
        return
    vocab_sz = int(getattr(getattr(model, "config", None), "vocab_size", 32000))
    vocab_sz = max(4, vocab_sz)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) + 999)
    ids = torch.randint(low=0, high=vocab_sz, size=(4, int(seq_len)), generator=g, device=device, dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.long, device=device)
    labels = ids.clone()
    # -100 for pad (none here)
    for _ in range(int(steps)):
        if dtype is None:
            _ = model(input_ids=ids, attention_mask=mask, labels=labels)
        else:
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                _ = model(input_ids=ids, attention_mask=mask, labels=labels)
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--warmup_steps", type=int, default=8)
    ap.add_argument(
        "--mc_score_norm",
        type=str,
        default="mean_char",
        choices=["sum", "mean_token", "mean_char", "mean_byte"],
        help="MC scoring normalization: sum (classic) or mean per token/char/byte (fairer for varying option lengths)",
    )

    # Models (defaults wired to this repo's current best 3-way setup)
    ap.add_argument("--bpe_tokenizer_dir", type=str, default="/home/arxiv_code/tokenizers_rust/tokenizer_out_bpe_wikitext103_32k_full_v019")
    ap.add_argument("--bpe_model_dir", type=str, default="/home/arxiv_code/tokenizers_rust/model_bpe_lenmax_maxchars64_full_tokfixed_pack4000_evalval_steps10000_v027/best_bpc")
    ap.add_argument("--superbpe_tokenizer_dir", type=str, default="/home/arxiv_code/tokenizers_rust/tokenizer_out_superbpe_wikitext103_32000_full_v043")
    ap.add_argument("--superbpe_model_dir", type=str, default="/home/arxiv_code/tokenizers_rust/model_superbpe_superbpe_pack4000_evalval_steps10000_v044/best_bpc")
    ap.add_argument("--lenmax_tokenizer_dir", type=str, default="/home/arxiv_code/tokenizers_rust/tokenizer_out_wikitext103_32k_n9_full_maxchars48_punctnorm_nostage_lowtpc_crossmix_v047")
    ap.add_argument("--lenmax_model_dir", type=str, default="/home/arxiv_code/tokenizers_rust/model_lenmax_lenmax_punctnorm_nostage_crossmix_v047_vs_superbpe_pack4000_evalval_steps10000_v048/best_bpc")

    # Tasks
    ap.add_argument("--run_lambada", action="store_true")
    ap.add_argument("--run_hellaswag", action="store_true")
    ap.add_argument("--run_piqa", action="store_true")
    ap.add_argument("--run_winogrande", action="store_true")
    ap.add_argument("--winogrande_config", type=str, default="winogrande_xl")
    ap.add_argument("--max_examples", type=int, default=500, help="per HF task")
    ap.add_argument("--lambada_choices", type=int, default=10)

    ap.add_argument("--run_wt103_longmc", action="store_true")
    ap.add_argument("--wt103_test", type=str, default="/home/arxiv_code/datasets/wikitext103_raw_txt/test.txt")
    ap.add_argument("--longmc_examples", type=int, default=400)
    ap.add_argument("--longmc_context_chars", type=str, default="1500,2500,3500")
    ap.add_argument("--longmc_cont_chars", type=int, default=128)
    ap.add_argument("--longmc_choices", type=int, default=4)

    ap.add_argument("--run_passkey", action="store_true")
    ap.add_argument("--passkey_examples", type=int, default=400)
    ap.add_argument("--passkey_total_chars", type=str, default="1500,2500,3500")
    ap.add_argument("--passkey_choices", type=int, default=4)

    ap.add_argument("--run_wt103_passkey", action="store_true")
    ap.add_argument("--wt103_passkey_examples", type=int, default=500)
    ap.add_argument("--wt103_passkey_total_chars", type=str, default="2000,2400,2800,3200,3600")

    ap.add_argument("--run_kv_retrieval", action="store_true")
    ap.add_argument("--kv_examples", type=int, default=500)
    ap.add_argument("--kv_total_chars", type=str, default="2000,2400,2800,3200,3600")
    ap.add_argument("--kv_choices", type=int, default=4)

    ap.add_argument("--run_wt103_passkey_needle", action="store_true")
    ap.add_argument("--wt103_passkey_needle_examples", type=int, default=400)
    ap.add_argument("--wt103_passkey_needle_total_chars", type=str, default="2000,2400,2800,3200,3600")
    ap.add_argument("--wt103_passkey_needle_pos_fracs", type=str, default="0.5")

    ap.add_argument("--run_kv_needle", action="store_true")
    ap.add_argument("--kv_needle_examples", type=int, default=400)
    ap.add_argument("--kv_needle_total_chars", type=str, default="2000,2400,2800,3200,3600")
    ap.add_argument("--kv_needle_pos_fracs", type=str, default="0.5")

    ap.add_argument("--wt103_filler_corpus", type=str, default="/home/arxiv_code/datasets/wikitext103_raw_txt/test.txt")

    ap.add_argument("--out_csv", type=str, default="/home/arxiv_code/tokenizers_rust/downstream_threeway_results.csv")

    args = ap.parse_args()

    # Default behavior: run everything if no explicit flags.
    if not (
        args.run_lambada
        or args.run_hellaswag
        or args.run_piqa
        or args.run_winogrande
        or args.run_wt103_longmc
        or args.run_passkey
        or args.run_wt103_passkey
        or args.run_kv_retrieval
        or args.run_wt103_passkey_needle
        or args.run_kv_needle
    ):
        args.run_lambada = True
        args.run_hellaswag = True
        args.run_piqa = True
        args.run_winogrande = True
        args.run_wt103_longmc = True
        args.run_passkey = True
        args.run_wt103_passkey = True
        args.run_kv_retrieval = True
        args.run_wt103_passkey_needle = True
        args.run_kv_needle = True

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    specs = [
        ModelSpec("BPE", args.bpe_tokenizer_dir, args.bpe_model_dir),
        ModelSpec("SuperBPE", args.superbpe_tokenizer_dir, args.superbpe_model_dir),
        ModelSpec("Length-MAX", args.lenmax_tokenizer_dir, args.lenmax_model_dir),
    ]

    # Parse per-length lists
    longmc_contexts = [int(x.strip()) for x in str(args.longmc_context_chars).split(",") if x.strip()]
    passkey_chars = [int(x.strip()) for x in str(args.passkey_total_chars).split(",") if x.strip()]
    wt103_passkey_chars = [int(x.strip()) for x in str(args.wt103_passkey_total_chars).split(",") if x.strip()]
    kv_chars = [int(x.strip()) for x in str(args.kv_total_chars).split(",") if x.strip()]
    wt103_passkey_needle_chars = [int(x.strip()) for x in str(args.wt103_passkey_needle_total_chars).split(",") if x.strip()]
    kv_needle_chars = [int(x.strip()) for x in str(args.kv_needle_total_chars).split(",") if x.strip()]
    wt103_passkey_needle_pos = [float(x.strip()) for x in str(args.wt103_passkey_needle_pos_fracs).split(",") if x.strip()]
    kv_needle_pos = [float(x.strip()) for x in str(args.kv_needle_pos_fracs).split(",") if x.strip()]

    rows: list[dict] = []

    for ms in specs:
        print()
        print("== load ==", ms.name)
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(ms.tokenizer_dir, trust_remote_code=True)
        model = LlamaForCausalLM.from_pretrained(ms.model_dir)
        model.to(dev)
        model.eval()
        print(f"loaded in {time.perf_counter()-t0:.2f}s  device={dev} precision={args.precision} seq_len={args.seq_len}")

        _warmup_model(model, device=dev, precision=str(args.precision), seq_len=int(args.seq_len), seed=int(args.seed), steps=int(args.warmup_steps))

        if args.run_lambada:
            print(f"[{ms.name}] lambada_mc(k={int(args.lambada_choices)}) ...")
            r = eval_lambada_mc(
                model=model,
                tok=tok,
                device=dev,
                precision=str(args.precision),
                seq_len=int(args.seq_len),
                score_norm=str(args.mc_score_norm),
                max_examples=int(args.max_examples),
                num_choices=int(args.lambada_choices),
                seed=int(args.seed),
            )
            r["model"] = ms.name
            rows.append(r)

        if args.run_hellaswag:
            print(f"[{ms.name}] hellaswag ...")
            r = eval_hellaswag(
                model=model,
                tok=tok,
                device=dev,
                precision=str(args.precision),
                seq_len=int(args.seq_len),
                score_norm=str(args.mc_score_norm),
                max_examples=int(args.max_examples),
                seed=int(args.seed),
            )
            r["model"] = ms.name
            rows.append(r)

        if args.run_piqa:
            print(f"[{ms.name}] piqa ...")
            r = eval_piqa(
                model=model,
                tok=tok,
                device=dev,
                precision=str(args.precision),
                seq_len=int(args.seq_len),
                score_norm=str(args.mc_score_norm),
                max_examples=int(args.max_examples),
                seed=int(args.seed),
            )
            r["model"] = ms.name
            rows.append(r)

        if args.run_winogrande:
            print(f"[{ms.name}] winogrande({args.winogrande_config}) ...")
            r = eval_winogrande(model=model, tok=tok, device=dev, precision=str(args.precision), seq_len=int(args.seq_len), max_examples=int(args.max_examples), seed=int(args.seed), config=str(args.winogrande_config))
            r["model"] = ms.name
            rows.append(r)

        if args.run_wt103_longmc:
            for cc in longmc_contexts:
                print(f"[{ms.name}] wt103_longmc context_chars={cc} ...")
                r = eval_wt103_longmc(
                    model=model,
                    tok=tok,
                    device=dev,
                    precision=str(args.precision),
                    seq_len=int(args.seq_len),
                    score_norm=str(args.mc_score_norm),
                    corpus_path=Path(str(args.wt103_test)),
                    num_examples=int(args.longmc_examples),
                    context_chars=int(cc),
                    cont_chars=int(args.longmc_cont_chars),
                    num_choices=int(args.longmc_choices),
                    seed=int(args.seed),
                )
                r["model"] = ms.name
                rows.append(r)

        if args.run_passkey:
            for tc in passkey_chars:
                print(f"[{ms.name}] passkey_mc total_chars={tc} ...")
                r = eval_passkey_mc(
                    model=model,
                    tok=tok,
                    device=dev,
                    precision=str(args.precision),
                    seq_len=int(args.seq_len),
                    score_norm=str(args.mc_score_norm),
                    num_examples=int(args.passkey_examples),
                    total_chars=int(tc),
                    num_choices=int(args.passkey_choices),
                    seed=int(args.seed),
                )
                r["model"] = ms.name
                rows.append(r)

        if args.run_wt103_passkey:
            for tc in wt103_passkey_chars:
                print(f"[{ms.name}] wt103_passkey_mc total_chars={tc} ...")
                r = eval_wt103_passkey_mc(
                    model=model,
                    tok=tok,
                    device=dev,
                    precision=str(args.precision),
                    seq_len=int(args.seq_len),
                    score_norm=str(args.mc_score_norm),
                    num_examples=int(args.wt103_passkey_examples),
                    total_chars=int(tc),
                    num_choices=4,
                    seed=int(args.seed),
                    corpus_path=Path(str(args.wt103_filler_corpus)),
                )
                r["model"] = ms.name
                rows.append(r)

        if args.run_kv_retrieval:
            for tc in kv_chars:
                print(f"[{ms.name}] kv_retrieval_mc total_chars={tc} ...")
                r = eval_kv_retrieval_mc(
                    model=model,
                    tok=tok,
                    device=dev,
                    precision=str(args.precision),
                    seq_len=int(args.seq_len),
                    score_norm=str(args.mc_score_norm),
                    num_examples=int(args.kv_examples),
                    total_chars=int(tc),
                    num_choices=int(args.kv_choices),
                    seed=int(args.seed),
                    corpus_path=Path(str(args.wt103_filler_corpus)),
                )
                r["model"] = ms.name
                rows.append(r)

        if args.run_wt103_passkey_needle:
            for pos in wt103_passkey_needle_pos:
                for tc in wt103_passkey_needle_chars:
                    print(f"[{ms.name}] wt103_passkey_needle_mc pos={pos} total_chars={tc} ...")
                    r = eval_wt103_passkey_needle_mc(
                        model=model,
                        tok=tok,
                        device=dev,
                        precision=str(args.precision),
                        seq_len=int(args.seq_len),
                        score_norm=str(args.mc_score_norm),
                        num_examples=int(args.wt103_passkey_needle_examples),
                        total_chars=int(tc),
                        num_choices=4,
                        seed=int(args.seed),
                        corpus_path=Path(str(args.wt103_filler_corpus)),
                        needle_pos_frac=float(pos),
                    )
                    r["model"] = ms.name
                    rows.append(r)

        if args.run_kv_needle:
            for pos in kv_needle_pos:
                for tc in kv_needle_chars:
                    print(f"[{ms.name}] kv_needle_mc pos={pos} total_chars={tc} ...")
                    r = eval_kv_needle_mc(
                        model=model,
                        tok=tok,
                        device=dev,
                        precision=str(args.precision),
                        seq_len=int(args.seq_len),
                        score_norm=str(args.mc_score_norm),
                        num_examples=int(args.kv_needle_examples),
                        total_chars=int(tc),
                        num_choices=int(args.kv_choices),
                        seed=int(args.seed),
                        corpus_path=Path(str(args.wt103_filler_corpus)),
                        needle_pos_frac=float(pos),
                    )
                    r["model"] = ms.name
                    rows.append(r)

        # free CUDA memory between models
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # Write CSV
    out_csv = Path(str(args.out_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # stable columns
    keys = set()
    for r in rows:
        keys.update(r.keys())
    keys = sorted(keys)
    out_csv.write_text("")  # create early
    import csv

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print()
    print("[done] wrote", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


