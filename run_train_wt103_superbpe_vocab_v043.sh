#!/usr/bin/env bash
set -euo pipefail

# v043: 训练 SuperBPE（跨空格 merge 的 ByteLevel BPE）词表，并导出 HF tokenizer 目录。
# 从现在起，所有对照实验的 baseline 将优先使用 SuperBPE，而不是旧的 BPE(v019)。

CORPUS_TRAIN="/home/arxiv_code/datasets/wikitext103_raw_txt/train.txt"

VOCAB_SIZE=32000
MIN_FREQ=2
LIMIT_ALPHA=1000

TAG="superbpe_wikitext103_${VOCAB_SIZE}_full_v043"
OUT_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_superbpe_wikitext103_${VOCAB_SIZE}_full_v043"
LOG="/home/arxiv_code/tokenizers_rust/train_superbpe_wikitext103_${VOCAB_SIZE}_full_v043.log"

echo "[superbpe-vocab] start $(date -Is)"
echo "[superbpe-vocab] corpus=$CORPUS_TRAIN"
echo "[superbpe-vocab] out_dir=$OUT_DIR"
echo "[superbpe-vocab] vocab_size=$VOCAB_SIZE min_freq=$MIN_FREQ limit_alphabet=$LIMIT_ALPHA"
echo "[superbpe-vocab] NOTE: ByteLevel(use_regex=False) to allow cross-space merges"

mkdir -p "$OUT_DIR"

python3 /home/arxiv_code/tokenizers_rust/train_superbpe_hf.py \
  --corpus_file "$CORPUS_TRAIN" \
  --out_dir "$OUT_DIR" \
  --vocab_size "$VOCAB_SIZE" \
  --min_frequency "$MIN_FREQ" \
  --limit_alphabet "$LIMIT_ALPHA" \
  > "$LOG" 2>&1

echo "[superbpe-vocab] done $(date -Is)"
echo "[superbpe-vocab] out_dir=$OUT_DIR"
echo "[superbpe-vocab] log=$LOG"


