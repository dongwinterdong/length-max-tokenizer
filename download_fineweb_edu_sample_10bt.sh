#!/usr/bin/env bash
set -euo pipefail

# Download HuggingFaceFW/fineweb-edu sample/10BT parquet shards (14 files).
#
# Usage:
#   bash download_fineweb_edu_sample_10bt.sh /path/to/output_dir
#
# Notes:
# - Uses wget -c for resume.
# - Total size ~28.5 GB.

OUT_DIR="${1:-}"
if [[ -z "${OUT_DIR}" ]]; then
  echo "Usage: $0 /path/to/output_dir" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

BASE="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT"

FILES=(
  "000_00000.parquet"
  "001_00000.parquet"
  "002_00000.parquet"
  "003_00000.parquet"
  "004_00000.parquet"
  "005_00000.parquet"
  "006_00000.parquet"
  "007_00000.parquet"
  "008_00000.parquet"
  "009_00000.parquet"
  "010_00000.parquet"
  "011_00000.parquet"
  "012_00000.parquet"
  "013_00000.parquet"
)

for f in "${FILES[@]}"; do
  echo "+ wget -c -O ${f} ${BASE}/${f}" >&2
  wget -c -O "${f}" "${BASE}/${f}"
done

echo "done: ${OUT_DIR}" >&2



