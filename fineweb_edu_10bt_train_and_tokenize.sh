#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline for FineWeb-Edu sample-10BT:
# 1) Download all 14 parquet shards
# 2) Train tokenizer vocab (token_table.json) on a *sample* of docs (TRAIN_MAX_DOCS)
# 3) Export vocab.json (HF format)
# 4) Tokenize ALL parquet docs with vocab.json (DP min-token) and write ids per shard
#
# IMPORTANT:
# - Training on the full 10BT with the current trainer is not practical; TRAIN_MAX_DOCS controls sampling.
# - Tokenization is streaming and can process all shards; output can be huge (tens of GB). We gzip each shard.
#
# Usage:
#   bash fineweb_edu_10bt_train_and_tokenize.sh \
#     /home/arxiv_code/data/fineweb-edu-sample-10BT \
#     /home/arxiv_code/data/fineweb-edu-10BT-out \
#     200000   # TRAIN_MAX_DOCS
#
# Optional env vars:
#   NUM_MERGES (default 50000)
#   AIM_TOKEN_NUM (default 20000)
#   N_MAX (default 6)
#   NUM_WORKERS (default 0)
#   MULTI_PROCESS (default 0/1)

DATA_DIR="${1:-}"
OUT_DIR="${2:-}"
TRAIN_MAX_DOCS="${3:-}"
if [[ -z "${DATA_DIR}" || -z "${OUT_DIR}" || -z "${TRAIN_MAX_DOCS}" ]]; then
  echo "Usage: $0 <data_dir> <out_dir> <train_max_docs>" >&2
  exit 2
fi

NUM_MERGES="${NUM_MERGES:-50000}"
AIM_TOKEN_NUM="${AIM_TOKEN_NUM:-20000}"
N_MAX="${N_MAX:-6}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MULTI_PROCESS="${MULTI_PROCESS:-0}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${DATA_DIR}"
mkdir -p "${OUT_DIR}"

echo "[1/4] Download sample-10BT parquet shards..." >&2
bash "${REPO_DIR}/download_fineweb_edu_sample_10bt.sh" "${DATA_DIR}"

echo "[2/4] Build release binaries..." >&2
(cd "${REPO_DIR}" && cargo build --release)

TOKEN_TABLE="${OUT_DIR}/token_table.json"

echo "[3/4] Train token_table.json on parquet sample (max_docs=${TRAIN_MAX_DOCS})..." >&2
TRAIN_CMD=(
  "${REPO_DIR}/target/release/length_tokenizer"
  --corpus "${DATA_DIR}"
  --corpus-format parquet
  --parquet-recursive
  --text-column text
  --max-docs "${TRAIN_MAX_DOCS}"
  --output "${TOKEN_TABLE}"
  --num-merges "${NUM_MERGES}"
  --aim-token-num "${AIM_TOKEN_NUM}"
  --n-max "${N_MAX}"
  --num-workers "${NUM_WORKERS}"
)
if [[ "${MULTI_PROCESS}" == "1" ]]; then
  TRAIN_CMD+=(--multi-process)
fi
"${TRAIN_CMD[@]}"

echo "[3b/4] Export vocab.json..." >&2
"${REPO_DIR}/target/release/export_hf_tokenizer" "${TOKEN_TABLE}" "${OUT_DIR}"

VOCAB="${OUT_DIR}/vocab.json"
if [[ ! -f "${VOCAB}" ]]; then
  echo "ERROR: vocab.json not found at ${VOCAB}" >&2
  exit 1
fi

echo "[4/4] Tokenize ALL parquet docs (write ids per shard, gzip)..." >&2
IDS_DIR="${OUT_DIR}/ids"
mkdir -p "${IDS_DIR}"

shopt -s nullglob
FILES=("${DATA_DIR}"/*.parquet)
if [[ "${#FILES[@]}" -eq 0 ]]; then
  echo "ERROR: no parquet files in ${DATA_DIR}" >&2
  exit 1
fi

for f in "${FILES[@]}"; do
  base="$(basename "${f}")"
  out_bin="${IDS_DIR}/${base}.ltbin"
  out_gz="${out_bin}.gz"
  if [[ -f "${out_gz}" ]]; then
    echo "skip (already exists): ${out_gz}" >&2
    continue
  fi
  echo "+ tokenize ${base}" >&2
  "${REPO_DIR}/target/release/tokenize_parquet_vocab" \
    --vocab "${VOCAB}" \
    --parquet "${f}" \
    --text-column text \
    --max-docs 0 \
    --out "${out_bin}" \
    --progress-every 200000
  gzip -1 -f "${out_bin}"
done

echo "done: ${OUT_DIR}" >&2



