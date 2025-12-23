#!/usr/bin/env python3
"""
Train a ByteLevel BPE tokenizer (GPT-2 style) on a plain-text corpus and export
an HF-loadable tokenizer directory.

We use the same special tokens as LengthTokenizer:
  <unk>=0, <pad>=1, <s>=2, </s>=3, <mask>=4   (by construction/order)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_file", type=str, required=True, help="Plain text corpus, one sentence per line")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory (HF tokenizer)")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--limit_alphabet", type=int, default=1000)
    ap.add_argument("--add_prefix_space", action="store_true", help="Match GPT-2 behavior (optional)")
    args = ap.parse_args()

    corpus_file = Path(args.corpus_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = ["<unk>", "<pad>", "<s>", "</s>", "<mask>"]

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=bool(args.add_prefix_space))
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=int(args.vocab_size),
        min_frequency=int(args.min_frequency),
        special_tokens=special_tokens,
        limit_alphabet=int(args.limit_alphabet),
        show_progress=True,
    )

    tokenizer.train([str(corpus_file)], trainer)

    # Ensure BOS/EOS are actually inserted when add_special_tokens=True in Transformers.
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    if bos_id is None or eos_id is None:
        raise RuntimeError("BOS/EOS not found in vocab after training; check special_tokens list.")
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", int(bos_id)), ("</s>", int(eos_id))],
    )

    # Save both the tokenizer.json and (optionally) vocab/merges for transparency.
    tokenizer_json = out_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))
    # tokenizers' BPE model supports saving vocab.json + merges.txt
    tokenizer.model.save(str(out_dir))  # type: ignore[attr-defined]

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        mask_token="<mask>",
    )
    hf_tok.save_pretrained(str(out_dir))

    print("== done ==")
    print("corpus_file =", corpus_file)
    print("out_dir     =", out_dir)
    print("vocab_size  =", hf_tok.vocab_size)
    print("pad/bos/eos =", hf_tok.pad_token_id, hf_tok.bos_token_id, hf_tok.eos_token_id)
    print("files       =", ", ".join(sorted(p.name for p in out_dir.iterdir() if p.is_file())))


if __name__ == "__main__":
    main()


