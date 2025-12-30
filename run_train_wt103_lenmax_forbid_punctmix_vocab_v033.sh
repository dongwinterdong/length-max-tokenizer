#!/usr/bin/env bash
set -euo pipefail

# v033: Train a new Length-MAX vocab on full WikiText-103 with an additional constraint:
# forbid_punct_word_mix=True
#
# This filters candidate merges that would create tokens mixing "hard punctuation" and
# alphanumeric word pieces (e.g. "\"Ġ)ĠandĠ"), which you flagged as undesirable.

CORPUS_TRAIN="/home/arxiv_code/datasets/wikitext103_raw_txt/train.txt"

# Tokenizer training params
LEN_NUM_MERGES=40000
LEN_AIM_VOCAB=32000
LEN_N_MIN=2
LEN_N_MAX=9
LEN_MAX_TOKEN_CHARS=48
LEN_WORKERS=64
LEN_MULTI_PROCESS=1
LEN_USE_HEAP=0
LEN_FORBID_PUNCT_WORD_MIX=1

TAG="lenmax_n${LEN_N_MAX}_full_maxchars${LEN_MAX_TOKEN_CHARS}_forbidpunctmix_v033"
OUT_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_wikitext103_32k_n${LEN_N_MAX}_full_maxchars${LEN_MAX_TOKEN_CHARS}_forbidpunctmix_v033"
LOG="/home/arxiv_code/tokenizers_rust/train_wikitext103_32k_n${LEN_N_MAX}_full_maxchars${LEN_MAX_TOKEN_CHARS}_forbidpunctmix_v033.log"

echo "[vocab-train] start $(date -Is)"
echo "[vocab-train] corpus=$CORPUS_TRAIN"
echo "[vocab-train] out_dir=$OUT_DIR"
echo "[vocab-train] merges=$LEN_NUM_MERGES aim_vocab=$LEN_AIM_VOCAB n_min=$LEN_N_MIN n_max=$LEN_N_MAX max_token_chars=$LEN_MAX_TOKEN_CHARS workers=$LEN_WORKERS multi_process=$LEN_MULTI_PROCESS forbid_punct_word_mix=$LEN_FORBID_PUNCT_WORD_MIX"
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
)

print("DONE vocab.json bytes=", (out_dir/"vocab.json").stat().st_size)
PY

echo "[vocab-train] done $(date -Is)"
echo "[vocab-train] out_dir=$OUT_DIR"
echo "[vocab-train] log=$LOG"





