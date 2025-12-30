#!/usr/bin/env bash
set -euo pipefail

# v046: “恢复非两段式版本”的 Length-MAX 词表训练（但保留词表规范，并更激进降低 TPC）。
#
# 目标：在保持词表规范（禁不完整跨词、禁噪声标点混合）的前提下，让 TPC 尽量逼近 SuperBPE。
#
# 关键设置：
# - 禁用 staged superword（cross_word_start_vocab=0, max_token_words=0）=> 允许从一开始自由生成多词短语 token
# - 词表规范（沿用 v038 思路）：
#     forbid_punct_tokens=1 + allow_space_punct_tokens=1 + allow_abbrev_tokens=1 + allow_hyphen_tokens=1
#     forbid_incomplete_cross_word=1
# - 新增两类“受控混合”以降低 TPC（更贴近 SuperBPE 的优势）：
#     allow_word_final_punct_tokens=1  （word,Ġ / word.Ġ 等）
#     allow_apostrophe_tokens=1        （don'tĠ / John'sĠ 等）
#
# 注意：max_token_chars 仍保持 48（更稳内存）。如果 TPC 仍明显高于 SuperBPE，可再考虑放宽到 96/128。

CORPUS_TRAIN="/home/arxiv_code/datasets/wikitext103_raw_txt/train.txt"

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
LEN_ALLOW_SPACE_PUNCT_TOKENS=1
LEN_ALLOW_ABBREV_TOKENS=1
LEN_ALLOW_HYPHEN_TOKENS=1
LEN_ALLOW_WORD_FINAL_PUNCT_TOKENS=1
LEN_ALLOW_APOSTROPHE_TOKENS=1

# 非两段式：不延迟 superword、不限制词数
LEN_CROSS_WORD_START_VOCAB=0
LEN_MAX_TOKEN_WORDS=0

LEN_FORBID_INCOMPLETE_CROSS_WORD=1

TAG="lenmax_n${LEN_N_MAX}_full_maxchars${LEN_MAX_TOKEN_CHARS}_punctnorm_nostage_lowtpc_v046"
OUT_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_wikitext103_32k_n${LEN_N_MAX}_full_maxchars${LEN_MAX_TOKEN_CHARS}_punctnorm_nostage_lowtpc_v046"
LOG="/home/arxiv_code/tokenizers_rust/train_wikitext103_32k_n${LEN_N_MAX}_full_maxchars${LEN_MAX_TOKEN_CHARS}_punctnorm_nostage_lowtpc_v046.log"

echo "[vocab-train] start $(date -Is)"
echo "[vocab-train] corpus=$CORPUS_TRAIN"
echo "[vocab-train] out_dir=$OUT_DIR"
echo "[vocab-train] merges=$LEN_NUM_MERGES aim_vocab=$LEN_AIM_VOCAB n_min=$LEN_N_MIN n_max=$LEN_N_MAX max_token_chars=$LEN_MAX_TOKEN_CHARS workers=$LEN_WORKERS multi_process=$LEN_MULTI_PROCESS"
echo "[vocab-train] forbid_punct_tokens=$LEN_FORBID_PUNCT_TOKENS allow_space_punct_tokens=$LEN_ALLOW_SPACE_PUNCT_TOKENS allow_abbrev_tokens=$LEN_ALLOW_ABBREV_TOKENS allow_hyphen_tokens=$LEN_ALLOW_HYPHEN_TOKENS"
echo "[vocab-train] allow_word_final_punct_tokens=$LEN_ALLOW_WORD_FINAL_PUNCT_TOKENS allow_apostrophe_tokens=$LEN_ALLOW_APOSTROPHE_TOKENS"
echo "[vocab-train] cross_word_start_vocab=$LEN_CROSS_WORD_START_VOCAB max_token_words=$LEN_MAX_TOKEN_WORDS (nostage)"
echo "[vocab-train] forbid_incomplete_cross_word=$LEN_FORBID_INCOMPLETE_CROSS_WORD forbid_punct_word_mix=$LEN_FORBID_PUNCT_WORD_MIX"
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
    allow_space_punct_tokens=bool(${LEN_ALLOW_SPACE_PUNCT_TOKENS}),
    allow_abbrev_tokens=bool(${LEN_ALLOW_ABBREV_TOKENS}),
    allow_hyphen_tokens=bool(${LEN_ALLOW_HYPHEN_TOKENS}),
    allow_word_final_punct_tokens=bool(${LEN_ALLOW_WORD_FINAL_PUNCT_TOKENS}),
    allow_apostrophe_tokens=bool(${LEN_ALLOW_APOSTROPHE_TOKENS}),
    cross_word_start_vocab=int(${LEN_CROSS_WORD_START_VOCAB}),
    max_token_words=int(${LEN_MAX_TOKEN_WORDS}),
    forbid_incomplete_cross_word=bool(${LEN_FORBID_INCOMPLETE_CROSS_WORD}),
)

print("DONE vocab.json bytes=", (out_dir/"vocab.json").stat().st_size)
PY

echo "[vocab-train] done $(date -Is)"
echo "[vocab-train] out_dir=$OUT_DIR"
echo "[vocab-train] log=$LOG"


