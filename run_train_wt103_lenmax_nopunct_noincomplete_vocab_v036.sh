#!/usr/bin/env bash
set -euo pipefail

# v036: Train a new Length-MAX vocab on full WikiText-103 with stronger constraints:
# - forbid_punct_tokens=True
#     Disallow creating any *multi-character* token that contains punctuation/symbols.
#     (Single-character punctuation tokens still exist to encode punctuation chars.)
# - forbid_incomplete_cross_word=True
#     Disallow creating tokens that contain END_TOKEN('Ġ') but do NOT end with 'Ġ'
#     (i.e. "cross-word but incomplete" tokens such as `000Ġkg`, `'sĠdeath`).

CORPUS_TRAIN="/home/arxiv_code/datasets/wikitext103_raw_txt/train.txt"

# Tokenizer training params (keep aligned with v033 for fair comparison)
LEN_NUM_MERGES=40000
LEN_AIM_VOCAB=32000
LEN_N_MIN=2
LEN_N_MAX=9
LEN_MAX_TOKEN_CHARS=48
LEN_WORKERS=64
LEN_MULTI_PROCESS=1
LEN_USE_HEAP=0

LEN_FORBID_PUNCT_WORD_MIX=0
LEN_FORBID_PUNCT_TOKENS=1
LEN_FORBID_INCOMPLETE_CROSS_WORD=1

TAG="lenmax_n${LEN_N_MAX}_full_maxchars${LEN_MAX_TOKEN_CHARS}_nopunct_noincomplete_v036"
OUT_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_wikitext103_32k_n${LEN_N_MAX}_full_maxchars${LEN_MAX_TOKEN_CHARS}_nopunct_noincomplete_v036"
LOG="/home/arxiv_code/tokenizers_rust/train_wikitext103_32k_n${LEN_N_MAX}_full_maxchars${LEN_MAX_TOKEN_CHARS}_nopunct_noincomplete_v036.log"

echo "[vocab-train] start $(date -Is)"
echo "[vocab-train] corpus=$CORPUS_TRAIN"
echo "[vocab-train] out_dir=$OUT_DIR"
echo "[vocab-train] merges=$LEN_NUM_MERGES aim_vocab=$LEN_AIM_VOCAB n_min=$LEN_N_MIN n_max=$LEN_N_MAX max_token_chars=$LEN_MAX_TOKEN_CHARS workers=$LEN_WORKERS multi_process=$LEN_MULTI_PROCESS"
echo "[vocab-train] forbid_punct_tokens=$LEN_FORBID_PUNCT_TOKENS forbid_incomplete_cross_word=$LEN_FORBID_INCOMPLETE_CROSS_WORD forbid_punct_word_mix=$LEN_FORBID_PUNCT_WORD_MIX"
echo "[vocab-train] log=$LOG"

mkdir -p "$OUT_DIR"

python3 - <<PY > "$LOG" 2>&1
from pathlib import Path
import length_tokenizer_rs

print("length_tokenizer_rs", getattr(length_tokenizer_rs, "__version__", None))

corpus = Path("${CORPUS_TRAIN}")
out_dir = Path("${OUT_DIR}")
out_dir.mkdir(parents=True, exist_ok=True)

length_tokenizer_rs.train_to_hf(
    corpus_file=str(corpus),
    out_dir=str(out_dir),
    num_merges=int(${LEN_NUM_MERGES}),
    aim_token_num=int(${LEN_AIM_VOCAB}),
    n_min=int(${LEN_N_MIN}),
    n_max=int(${LEN_N_MAX}),
    max_token_chars=int(${LEN_MAX_TOKEN_CHARS}),
    num_workers=int(${LEN_WORKERS}),
    multi_process=bool(${LEN_MULTI_PROCESS}),
    use_heap=bool(${LEN_USE_HEAP}),
    forbid_punct_word_mix=bool(${LEN_FORBID_PUNCT_WORD_MIX}),
    forbid_punct_tokens=bool(${LEN_FORBID_PUNCT_TOKENS}),
    forbid_incomplete_cross_word=bool(${LEN_FORBID_INCOMPLETE_CROSS_WORD}),
)

print("DONE vocab.json bytes=", (out_dir/"vocab.json").stat().st_size)
PY

echo "[vocab-train] done $(date -Is)"
echo "[vocab-train] out_dir=$OUT_DIR"
echo "[vocab-train] log=$LOG"




