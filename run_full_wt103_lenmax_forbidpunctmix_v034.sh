#!/usr/bin/env bash
set -euo pipefail

# v034: Train a Llama-style model using the new Length-MAX vocab trained with:
# - forbid_punct_word_mix=True (v033 vocab)
#
# Decode mode: default (no cross_word_whole_only / forbid_end_inner)

unset LENGTH_TOKENIZER_CROSS_WORD_WHOLE_ONLY || true
unset LENGTH_TOKENIZER_FORBID_END_INNER || true

CORPUS_TRAIN="/home/arxiv_code/datasets/wikitext103_raw_txt/train.txt"
CORPUS_EVAL="/home/arxiv_code/datasets/wikitext103_raw_txt/validation.txt"

# Tokenizers
BPE_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_bpe_wikitext103_32k_full_v019"
LEN_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_wikitext103_32k_n9_full_maxchars48_forbidpunctmix_v033"

# Reuse an existing BPE baseline model log if present
BPE_MODEL_LOG_DEFAULT="/home/arxiv_code/tokenizers_rust/run_llama_full_bpe_wt103_full_ddp2_h512l8_sl512_bs64_steps10000_bf16_pack4000_lenmax_maxchars64_full_tokfixed_pack4000_evalval_steps10000_v027.log"

# Model training params (same as other full runs)
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

TAG="lenmax_forbidpunctmix_pack${PACK_CHARS}_evalval_steps${STEPS}_v034"

# Logs / outputs
PIPE_LOG="/home/arxiv_code/tokenizers_rust/pipeline_${TAG}.log"

LEN_MODEL_LOG="/home/arxiv_code/tokenizers_rust/run_llama_lenmax_${TAG}.log"
BPE_MODEL_LOG="/home/arxiv_code/tokenizers_rust/run_llama_bpe_${TAG}.log"

LEN_SAVE_DIR="/home/arxiv_code/tokenizers_rust/model_lenmax_${TAG}"
BPE_SAVE_DIR="/home/arxiv_code/tokenizers_rust/model_bpe_${TAG}"

LEN_CURVE="/home/arxiv_code/tokenizers_rust/loss_curve_lenmax_${TAG}.csv"
BPE_CURVE="/home/arxiv_code/tokenizers_rust/loss_curve_bpe_${TAG}.csv"

LOSS_OVERLAY_PNG="/home/arxiv_code/tokenizers_rust/loss_overlay_lenmax_vs_bpe_${TAG}.png"
LOSS_OVERLAY_PDF="/home/arxiv_code/tokenizers_rust/loss_overlay_lenmax_vs_bpe_${TAG}.pdf"
BPC_OVERLAY_PNG="/home/arxiv_code/tokenizers_rust/bpc_overlay_lenmax_vs_bpe_${TAG}.png"
BPC_OVERLAY_PDF="/home/arxiv_code/tokenizers_rust/bpc_overlay_lenmax_vs_bpe_${TAG}.pdf"

TOKEN_COUNTS="/home/arxiv_code/tokenizers_rust/token_counts_${TAG}.txt"
SUMMARY_CSV="/home/arxiv_code/tokenizers_rust/summary_${TAG}.csv"

echo "[pipeline] start $(date -Is)" | tee -a "$PIPE_LOG"
echo "[pipeline] env LENGTH_TOKENIZER_CROSS_WORD_WHOLE_ONLY=(unset) LENGTH_TOKENIZER_FORBID_END_INNER=(unset)" | tee -a "$PIPE_LOG"
echo "[pipeline] train=$CORPUS_TRAIN eval=$CORPUS_EVAL" | tee -a "$PIPE_LOG"
echo "[pipeline] len_tok_dir=$LEN_TOK_DIR" | tee -a "$PIPE_LOG"
echo "[pipeline] bpe_tok_dir=$BPE_TOK_DIR" | tee -a "$PIPE_LOG"

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
wait_for_file "$BPE_TOK_DIR/tokenizer.json" "BPE HF tokenizer.json"
wait_for_file "$LEN_TOK_DIR/vocab.json" "Length-MAX HF vocab.json (forbidpunctmix)"

# 1) token totals (exact full-corpus TPC under default decode)
echo "[pipeline] count full-corpus token totals (default decode)" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/count_full_corpus_token_totals.py \
  --corpus "$CORPUS_TRAIN" \
  --len_tok_dir "$LEN_TOK_DIR" \
  --bpe_tok_dir "$BPE_TOK_DIR" \
  --batch_size 1024 \
  --report_every 200000 \
  > "$TOKEN_COUNTS" 2>&1 || true
echo "[pipeline] token totals written: $TOKEN_COUNTS" | tee -a "$PIPE_LOG"

# 2) BPE baseline log
BPE_BASE_LOG="$BPE_MODEL_LOG_DEFAULT"
if [[ -f "$BPE_BASE_LOG" ]] && grep -q "^== saved ==$" "$BPE_BASE_LOG"; then
  echo "[pipeline] reuse BPE baseline log: $BPE_BASE_LOG" | tee -a "$PIPE_LOG"
else
  echo "[pipeline] WARNING: default BPE baseline missing; training a new one: $BPE_MODEL_LOG" | tee -a "$PIPE_LOG"
  BPE_BASE_LOG="$BPE_MODEL_LOG"
  run_model "$BPE_TOK_DIR" "$BPE_MODEL_LOG" "$BPE_SAVE_DIR" "$STEPS" "BPE(baseline)"
fi

# 3) train Length-MAX model (forbidpunctmix vocab)
run_model "$LEN_TOK_DIR" "$LEN_MODEL_LOG" "$LEN_SAVE_DIR" "$STEPS" "Length-MAX(forbidpunctmix vocab)"

# 4) curves + overlay
python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$BPE_BASE_LOG" --out_csv "$BPE_CURVE" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$LEN_MODEL_LOG" --out_csv "$LEN_CURVE" | tee -a "$PIPE_LOG"

python3 /home/arxiv_code/tokenizers_rust/plot_loss_overlay.py \
  --csv "$LEN_CURVE" --label "Length-MAX (forbidpunctmix vocab)" \
  --csv "$BPE_CURVE" --label "BPE (baseline)" \
  --out_png "$LOSS_OVERLAY_PNG" \
  --out_pdf "$LOSS_OVERLAY_PDF" \
  --title "Loss vs step (lenmax forbidpunctmix vocab; eval=validation; pack=${PACK_CHARS})"

python3 /home/arxiv_code/tokenizers_rust/plot_loss_overlay.py \
  --y_col bpc \
  --csv "$LEN_CURVE" --label "Length-MAX (forbidpunctmix vocab)" \
  --csv "$BPE_CURVE" --label "BPE (baseline)" \
  --out_png "$BPC_OVERLAY_PNG" \
  --out_pdf "$BPC_OVERLAY_PDF" \
  --title "Train bpc proxy vs step (lenmax forbidpunctmix vocab; quality uses eval_loss->bpc)"

# 5) summary (stable bpc from eval_loss + full-corpus TPC)
LEN_TPC=$(grep -E '^len_tokens' "$TOKEN_COUNTS" | sed -E 's/.*tpc=([0-9.]+).*/\1/' | tail -n 1)
BPE_TPC=$(grep -E '^bpe_tokens' "$TOKEN_COUNTS" | sed -E 's/.*tpc=([0-9.]+).*/\1/' | tail -n 1)

BPE_BEST_LOSS=$(best_eval_loss "$BPE_BASE_LOG")
LEN_BEST_LOSS=$(best_eval_loss "$LEN_MODEL_LOG")

BPE_BPC=$(calc_bpc "$BPE_BEST_LOSS" "$BPE_TPC")
LEN_BPC=$(calc_bpc "$LEN_BEST_LOSS" "$LEN_TPC")

echo "len_tok_dir,decode_mode,len_tpc,bpe_tpc,bpe_best_eval_loss,len_best_eval_loss,bpe_best_bpc,len_best_bpc,len_model_log,bpe_model_log" > "$SUMMARY_CSV"
echo "${LEN_TOK_DIR},default,${LEN_TPC},${BPE_TPC},${BPE_BEST_LOSS},${LEN_BEST_LOSS},${BPE_BPC},${LEN_BPC},${LEN_MODEL_LOG},${BPE_BASE_LOG}" >> "$SUMMARY_CSV"
echo "[pipeline] SUMMARY: len_tpc=${LEN_TPC} bpe_tpc=${BPE_TPC} len_best_bpc=${LEN_BPC} bpe_best_bpc=${BPE_BPC}" | tee -a "$PIPE_LOG"
echo "[pipeline] summary_csv=$SUMMARY_CSV" | tee -a "$PIPE_LOG"
echo "[pipeline] DONE $(date -Is)" | tee -a "$PIPE_LOG"


