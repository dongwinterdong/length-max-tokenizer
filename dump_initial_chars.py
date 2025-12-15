#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump the initial character vocabulary used by the trainer.

Training normalization (matches Rust `encode_sentence_str` / `normalize_chars`):
- split_whitespace
- expand each word into Unicode `char`s
- append END_TOKEN ('Ġ') after each word

Output is a TSV sorted by codepoint:
U+XXXX <tab> char <tab> category <tab> unicode_name
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path


END_TOKEN = "Ġ"

def _display_char(ch: str) -> str:
    # Avoid writing control/format chars directly into the TSV (some editors treat it as binary).
    cat = unicodedata.category(ch)
    if cat.startswith("C"):  # Cc/Cf/Cs/Co/Cn
        return f"\\u{ord(ch):04X}"
    if ch == "\t":
        return "\\t"
    if ch == "\n":
        return "\\n"
    if ch == "\r":
        return "\\r"
    return ch


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump unique initial chars (+ END_TOKEN) from corpus")
    ap.add_argument("--corpus", default="corpus_py.txt", help="Corpus file (one sentence per line)")
    ap.add_argument("--out", default="corpus_py_initial_chars.tsv", help="Output TSV path")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    out = Path(args.out)

    chars: set[str] = set()
    # Stream scan: set size is ~10k, so memory is tiny.
    with corpus.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for w in line.split():
                chars.update(w)

    chars.add(END_TOKEN)

    # Stable order for inspection
    ordered = sorted(chars, key=ord)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as w:
        w.write(f"# corpus={corpus.resolve()}\n")
        w.write(f"# unique_chars_plus_END={len(ordered)} (includes END_TOKEN={END_TOKEN!r})\n")
        w.write("# columns: codepoint\\tdisplay\\tcategory\\tname\n")
        for ch in ordered:
            cp = ord(ch)
            cat = unicodedata.category(ch)
            name = unicodedata.name(ch, "<no name>")
            disp = _display_char(ch)
            w.write(f"U+{cp:04X}\t{disp}\t{cat}\t{name}\n")

    print(str(out.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


