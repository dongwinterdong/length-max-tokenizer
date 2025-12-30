#!/usr/bin/env bash
set -euo pipefail

# 通用模板：Length-MAX vs SuperBPE 的全流程对比（TPC + 10k steps 模型训练 + 曲线 + summary）。
#
# 用法示例：
#   LEN_TOK_DIR=/path/to/lenmax_tokenizer_dir \
#   TAG=my_experiment_tag \
#   bash /home/arxiv_code/tokenizers_rust/run_full_wt103_lenmax_vs_superbpe_template.sh
#
# 说明：
# - baseline 改为 SuperBPE（v043/v044），不再使用旧 BPE(v019)。
# - 若 SuperBPE baseline 模型日志已存在并完成，会复用；否则会训练一个。

unset LENGTH_TOKENIZER_CROSS_WORD_WHOLE_ONLY || true
unset LENGTH_TOKENIZER_FORBID_END_INNER || true
unset LENGTH_TOKENIZER_FORBID_PUNCT_TOKENS || true

if [[ -z "${LEN_TOK_DIR:-}" ]]; then
  echo "ERROR: LEN_TOK_DIR is required (HF dir with vocab.json/tokenization_length_tokenizer.py)" >&2
  exit 2
fi
TAG="${TAG:-lenmax_vs_superbpe_pack4000_evalval_steps10000}"
MAX_TPC_GAP="${MAX_TPC_GAP:-}"  # optional: gate model training by (len_tpc - superbpe_tpc) <= MAX_TPC_GAP

CORPUS_TRAIN="/home/arxiv_code/datasets/wikitext103_raw_txt/train.txt"
CORPUS_EVAL="/home/arxiv_code/datasets/wikitext103_raw_txt/validation.txt"

# Baseline: SuperBPE (v043 tokenizer)
SUPER_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_superbpe_wikitext103_32000_full_v043"
# Reuse an existing SuperBPE baseline model log if present
SUPER_MODEL_LOG_DEFAULT="/home/arxiv_code/tokenizers_rust/run_llama_superbpe_superbpe_pack4000_evalval_steps10000_v044.log"

# Model training params
PACK_CHARS=4000
SEQ_LEN=512
BATCH_SIZE=64
STEPS=10000
PRINT_EVERY=200
PRECISION=bf16

HIDDEN_SIZE=512
NUM_LAYERS=8
NUM_HEADS=8
NUM_KV_HEADS=8
INTERMEDIATE_SIZE=1408

LR=3e-4
LR_SCHEDULE=cosine
WARMUP_STEPS=200
MIN_LR=3e-5
WEIGHT_DECAY=0.1
ADAM_BETA2=0.95
GRAD_CLIP=1.0

EVAL_EVERY=200
EVAL_BATCHES=8
EVAL_BATCH_SIZE=32
EVAL_SEED=12345
SAVE_BEST_METRIC=bpc
SAVE_BEST_ON=eval
SAVE_BEST_MIN_DELTA=0.0002

GEN_FROM_BEST=1
GEN_PROMPT="= Valkyria Chronicles III ="
GEN_T=0.7
GEN_TOP_P=0.9
GEN_REP=1.12
GEN_NREP=3
MAX_NEW_TOKENS=160

# Outputs
PIPE_LOG="/home/arxiv_code/tokenizers_rust/pipeline_${TAG}.log"

LEN_MODEL_LOG="/home/arxiv_code/tokenizers_rust/run_llama_lenmax_${TAG}.log"
SUPER_MODEL_LOG="/home/arxiv_code/tokenizers_rust/run_llama_superbpe_${TAG}.log"

LEN_SAVE_DIR="/home/arxiv_code/tokenizers_rust/model_lenmax_${TAG}"
SUPER_SAVE_DIR="/home/arxiv_code/tokenizers_rust/model_superbpe_${TAG}"

LEN_CURVE="/home/arxiv_code/tokenizers_rust/loss_curve_lenmax_${TAG}.csv"
SUPER_CURVE="/home/arxiv_code/tokenizers_rust/loss_curve_superbpe_${TAG}.csv"

LOSS_OVERLAY_PNG="/home/arxiv_code/tokenizers_rust/loss_overlay_lenmax_vs_superbpe_${TAG}.png"
LOSS_OVERLAY_PDF="/home/arxiv_code/tokenizers_rust/loss_overlay_lenmax_vs_superbpe_${TAG}.pdf"
BPC_OVERLAY_PNG="/home/arxiv_code/tokenizers_rust/bpc_overlay_lenmax_vs_superbpe_${TAG}.png"
BPC_OVERLAY_PDF="/home/arxiv_code/tokenizers_rust/bpc_overlay_lenmax_vs_superbpe_${TAG}.pdf"

TOKEN_COUNTS="/home/arxiv_code/tokenizers_rust/token_counts_${TAG}.txt"
SUMMARY_CSV="/home/arxiv_code/tokenizers_rust/summary_${TAG}.csv"

echo "[pipeline] start $(date -Is)" | tee -a "$PIPE_LOG"
echo "[pipeline] train=$CORPUS_TRAIN eval=$CORPUS_EVAL" | tee -a "$PIPE_LOG"
echo "[pipeline] len_tok_dir=$LEN_TOK_DIR" | tee -a "$PIPE_LOG"
echo "[pipeline] superbpe_tok_dir=$SUPER_TOK_DIR" | tee -a "$PIPE_LOG"

wait_for_file() {
  local f="$1"
  local name="$2"
  echo "[pipeline] waiting for $name: $f" | tee -a "$PIPE_LOG"
  while [[ ! -f "$f" ]]; do
    sleep 60
  done
  echo "[pipeline] ready: $name" | tee -a "$PIPE_LOG"
}

best_eval_loss() {
  local log="$1"
  python3 - <<PY
import re
from pathlib import Path
log=Path("${log}")
pat=re.compile(r'^== eval ==.*eval_loss=([0-9.]+)')
vals=[]
for line in log.read_text().splitlines():
    m=pat.match(line.strip())
    if m:
        vals.append(float(m.group(1)))
print(min(vals) if vals else "nan")
PY
}

calc_bpc() {
  local loss="$1"
  local tpc="$2"
  python3 - <<PY
import math
loss=float("${loss}")
tpc=float("${tpc}")
print(loss*tpc/math.log(2.0))
PY
}

run_model() {
  local tok_dir="$1"
  local out_log="$2"
  local save_dir="$3"
  local steps="$4"
  local name="$5"
  if [[ -f "$out_log" ]] && grep -q "^== saved ==$" "$out_log"; then
    echo "[pipeline] skip $name model (already finished): $out_log" | tee -a "$PIPE_LOG"
    return 0
  fi
  echo "[pipeline] start $name model -> $out_log" | tee -a "$PIPE_LOG"
  rm -f "$out_log"
  PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 \
    /home/arxiv_code/tokenizers_rust/validate_modern_arch_llama.py \
    --tokenizer_dir "$tok_dir" \
    --corpus_file "$CORPUS_TRAIN" \
    --max_lines 0 \
    --eval_corpus_file "$CORPUS_EVAL" \
    --eval_max_lines 0 \
    --seed 42 \
    --device cuda \
    --precision "$PRECISION" \
    --seq_len "$SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --steps "$steps" \
    --hidden_size "$HIDDEN_SIZE" \
    --num_layers "$NUM_LAYERS" \
    --num_heads "$NUM_HEADS" \
    --num_kv_heads "$NUM_KV_HEADS" \
    --intermediate_size "$INTERMEDIATE_SIZE" \
    --print_every "$PRINT_EVERY" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --pack_chars "$PACK_CHARS" \
    --pack_mode contiguous \
    --force_eos \
    --lr "$LR" \
    --lr_schedule "$LR_SCHEDULE" \
    --warmup_steps "$WARMUP_STEPS" \
    --min_lr "$MIN_LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --adam_beta2 "$ADAM_BETA2" \
    --grad_clip "$GRAD_CLIP" \
    --tpc_estimate_lines 20000 \
    --save_dir "$save_dir" \
    --eval_every "$EVAL_EVERY" \
    --eval_batches "$EVAL_BATCHES" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --eval_seed "$EVAL_SEED" \
    --save_best_metric "$SAVE_BEST_METRIC" \
    --save_best_on "$SAVE_BEST_ON" \
    --save_best_min_delta "$SAVE_BEST_MIN_DELTA" \
    $( [[ "$GEN_FROM_BEST" == "1" ]] && echo --gen_from_best ) \
    --gen_prompt "$GEN_PROMPT" \
    --gen_temperature "$GEN_T" \
    --gen_top_p "$GEN_TOP_P" \
    --gen_repetition_penalty "$GEN_REP" \
    --gen_no_repeat_ngram_size "$GEN_NREP" \
    > "$out_log" 2>&1
  echo "[pipeline] done $name model" | tee -a "$PIPE_LOG"
}

# 0) ensure tokenizers exist
wait_for_file "$SUPER_TOK_DIR/tokenizer.json" "SuperBPE HF tokenizer.json"
wait_for_file "$LEN_TOK_DIR/vocab.json" "Length-MAX HF vocab.json"

# 1) token totals (exact full-corpus TPC under default decode)
echo "[pipeline] count full-corpus token totals (lenmax vs superbpe)" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/count_full_corpus_token_totals.py \
  --corpus "$CORPUS_TRAIN" \
  --len_tok_dir "$LEN_TOK_DIR" \
  --bpe_tok_dir "$SUPER_TOK_DIR" \
  --batch_size 1024 \
  --report_every 200000 \
  > "$TOKEN_COUNTS" 2>&1 || true
echo "[pipeline] token totals written: $TOKEN_COUNTS" | tee -a "$PIPE_LOG"

# Parse TPCs once (reused later)
# NOTE: sed backref must be '\1' (NOT '\\1'), otherwise bash -> python will see "\1" => control char 0x01 and float() will fail.
LEN_TPC=$(grep -E '^len_tokens' "$TOKEN_COUNTS" | sed -E 's/.*tpc=([0-9.]+).*/\1/' | tail -n 1)
SUPER_TPC=$(grep -E '^bpe_tokens' "$TOKEN_COUNTS" | sed -E 's/.*tpc=([0-9.]+).*/\1/' | tail -n 1)

is_float() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

if ! is_float "$LEN_TPC" || ! is_float "$SUPER_TPC"; then
  echo "[pipeline] ERROR: failed to parse TPC from token_counts" | tee -a "$PIPE_LOG"
  echo "[pipeline] len_tpc=$LEN_TPC super_tpc=$SUPER_TPC token_counts=$TOKEN_COUNTS" | tee -a "$PIPE_LOG"
  echo "[pipeline] tail token_counts:" | tee -a "$PIPE_LOG"
  tail -n 120 "$TOKEN_COUNTS" | tee -a "$PIPE_LOG"
  exit 1
fi

echo "[pipeline] tpc(lenmax)=$LEN_TPC tpc(superbpe)=$SUPER_TPC" | tee -a "$PIPE_LOG"

if [[ -n "$MAX_TPC_GAP" ]]; then
  echo "[pipeline] gate enabled: require (len_tpc - superbpe_tpc) <= MAX_TPC_GAP=$MAX_TPC_GAP" | tee -a "$PIPE_LOG"
  python3 - <<PY
import math, sys
len_tpc=float("${LEN_TPC}")
super_tpc=float("${SUPER_TPC}")
max_gap=float("${MAX_TPC_GAP}")
gap=len_tpc-super_tpc
print(f"[gate] len_tpc={len_tpc} super_tpc={super_tpc} gap={gap} max_gap={max_gap}")
sys.exit(0 if gap <= max_gap else 10)
PY
  code=$?
  if [[ $code -eq 10 ]]; then
    echo "[pipeline] gate FAIL: tpc gap too large, skip model training; see token_counts=$TOKEN_COUNTS" | tee -a "$PIPE_LOG"
    echo "[pipeline] DONE (gate) $(date -Is)" | tee -a "$PIPE_LOG"
    exit 0
  fi
  if [[ $code -ne 0 ]]; then
    echo "[pipeline] ERROR: gate computation failed (code=$code)" | tee -a "$PIPE_LOG"
    exit 1
  fi
  echo "[pipeline] gate PASS: proceed to model training" | tee -a "$PIPE_LOG"
fi

# 2) SuperBPE baseline log
SUPER_BASE_LOG="$SUPER_MODEL_LOG_DEFAULT"
if [[ -f "$SUPER_BASE_LOG" ]] && grep -q "^== saved ==$" "$SUPER_BASE_LOG"; then
  echo "[pipeline] reuse SuperBPE baseline log: $SUPER_BASE_LOG" | tee -a "$PIPE_LOG"
else
  echo "[pipeline] WARNING: default SuperBPE baseline missing; training a new one: $SUPER_MODEL_LOG" | tee -a "$PIPE_LOG"
  SUPER_BASE_LOG="$SUPER_MODEL_LOG"
  run_model "$SUPER_TOK_DIR" "$SUPER_MODEL_LOG" "$SUPER_SAVE_DIR" "$STEPS" "SuperBPE(baseline)"
fi

# 3) train Length-MAX model
run_model "$LEN_TOK_DIR" "$LEN_MODEL_LOG" "$LEN_SAVE_DIR" "$STEPS" "Length-MAX"

# 4) curves + overlay
python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$SUPER_BASE_LOG" --out_csv "$SUPER_CURVE" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$LEN_MODEL_LOG" --out_csv "$LEN_CURVE" | tee -a "$PIPE_LOG"

python3 /home/arxiv_code/tokenizers_rust/plot_loss_overlay.py \
  --csv "$LEN_CURVE" --label "Length-MAX" \
  --csv "$SUPER_CURVE" --label "SuperBPE (baseline)" \
  --out_png "$LOSS_OVERLAY_PNG" \
  --out_pdf "$LOSS_OVERLAY_PDF" \
  --title "Loss vs step (lenmax vs superbpe; eval=validation; pack=${PACK_CHARS})"

python3 /home/arxiv_code/tokenizers_rust/plot_loss_overlay.py \
  --y_col bpc \
  --csv "$LEN_CURVE" --label "Length-MAX" \
  --csv "$SUPER_CURVE" --label "SuperBPE (baseline)" \
  --out_png "$BPC_OVERLAY_PNG" \
  --out_pdf "$BPC_OVERLAY_PDF" \
  --title "Train bpc proxy vs step (lenmax vs superbpe; quality uses eval_loss->bpc)"

# 5) summary (stable bpc from eval_loss + full-corpus TPC)

SUPER_BEST_LOSS=$(best_eval_loss "$SUPER_BASE_LOG")
LEN_BEST_LOSS=$(best_eval_loss "$LEN_MODEL_LOG")

SUPER_BPC=$(calc_bpc "$SUPER_BEST_LOSS" "$SUPER_TPC")
LEN_BPC=$(calc_bpc "$LEN_BEST_LOSS" "$LEN_TPC")

echo "len_tok_dir,baseline_tok_dir,decode_mode,len_tpc,super_tpc,super_best_eval_loss,len_best_eval_loss,super_best_bpc,len_best_bpc,len_model_log,super_model_log" > "$SUMMARY_CSV"
echo "${LEN_TOK_DIR},${SUPER_TOK_DIR},default,${LEN_TPC},${SUPER_TPC},${SUPER_BEST_LOSS},${LEN_BEST_LOSS},${SUPER_BPC},${LEN_BPC},${LEN_MODEL_LOG},${SUPER_BASE_LOG}" >> "$SUMMARY_CSV"
echo "[pipeline] SUMMARY: len_tpc=${LEN_TPC} super_tpc=${SUPER_TPC} len_best_bpc=${LEN_BPC} super_best_bpc=${SUPER_BPC}" | tee -a "$PIPE_LOG"
echo "[pipeline] summary_csv=$SUMMARY_CSV" | tee -a "$PIPE_LOG"
echo "[pipeline] DONE $(date -Is)" | tee -a "$PIPE_LOG"


