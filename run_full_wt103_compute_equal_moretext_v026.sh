#!/usr/bin/env bash
set -euo pipefail

# Compute-equal "more training text" experiment (held-out eval on validation.txt).
#
# Goal:
# - Keep model + global batch the same.
# - Use Length-MAX's lower TPC to reduce seq_len (tokens) while keeping roughly the same
#   *character* context per sequence as BPE.
# - With smaller seq_len, per-step compute drops, so we can run more steps at ~same compute.
#
# Setup:
# - BPE:      seq_len=512, steps=10000
# - LengthMAX: seq_len≈512*(TPC_len/TPC_bpe) ≈ 425, steps≈10000*(512/425)^2 ≈ 14500
#
# We evaluate on validation.txt via validate_modern_arch_llama.py --eval_corpus_file.

CORPUS_TRAIN="/home/arxiv_code/datasets/wikitext103_raw_txt/train.txt"
CORPUS_EVAL="/home/arxiv_code/datasets/wikitext103_raw_txt/validation.txt"

LEN_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_wikitext103_32k_n9_full_v019"
BPE_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_bpe_wikitext103_32k_full_v019"

PACK_CHARS=4000

# Baseline (BPE) config
SEQ_BPE=512
STEPS_BPE=10000

# Full-corpus TPC values (measured by count_full_corpus_token_totals.py)
TPC_LEN=0.176099
TPC_BPE=0.211906

# Derived Length-MAX config (keep char context ~equal, then scale steps for compute)
SEQ_LEN=$(python3 - <<PY
tpc_len=${TPC_LEN}
tpc_bpe=${TPC_BPE}
seq_bpe=${SEQ_BPE}
seq_len=int(round(seq_bpe*(tpc_len/tpc_bpe)))
print(seq_len)
PY
)
STEPS_LEN=$(python3 - <<PY
import math
seq_bpe=${SEQ_BPE}
seq_len=${SEQ_LEN}
steps_bpe=${STEPS_BPE}
steps_len=int(round(steps_bpe*((seq_bpe/seq_len)**2)))
print(steps_len)
PY
)

# Training hyperparams
PRECISION=bf16
BATCH_SIZE=64
PRINT_EVERY=200
LR=3e-4
LR_SCHEDULE=cosine
MIN_LR=3e-5
WEIGHT_DECAY=0.1
ADAM_BETA2=0.95
GRAD_CLIP=1.0

# Warmup as % of total updates (2%)
WARMUP_BPE=200
WARMUP_LEN=$(python3 - <<PY
steps_len=${STEPS_LEN}
steps_bpe=${STEPS_BPE}
warmup_bpe=${WARMUP_BPE}
print(int(round(warmup_bpe*steps_len/steps_bpe)))
PY
)

# Eval-best (validation)
EVAL_EVERY=200
EVAL_BATCHES=8
EVAL_BATCH_SIZE=32
EVAL_SEED=12345
SAVE_BEST_METRIC=bpc
SAVE_BEST_ON=eval
SAVE_BEST_MIN_DELTA=0.0002

# Generation
GEN_PROMPT="= Valkyria Chronicles III ="
GEN_FROM_BEST=1
GEN_T=0.7
GEN_TOP_P=0.9
GEN_REP=1.12
GEN_NREP=3
MAX_NEW_TOKENS=160

TAG="compute_equal_moretext_evalval_v026"

PIPE_LOG="/home/arxiv_code/tokenizers_rust/pipeline_${TAG}.log"

LEN_LOG="/home/arxiv_code/tokenizers_rust/run_llama_lenmax_${TAG}_ddp2_h512l8_sl${SEQ_LEN}_bs${BATCH_SIZE}_steps${STEPS_LEN}_${PRECISION}_pack${PACK_CHARS}.log"
BPE_LOG="/home/arxiv_code/tokenizers_rust/run_llama_bpe_${TAG}_ddp2_h512l8_sl${SEQ_BPE}_bs${BATCH_SIZE}_steps${STEPS_BPE}_${PRECISION}_pack${PACK_CHARS}.log"

LEN_SAVE="/home/arxiv_code/tokenizers_rust/model_lenmax_${TAG}"
BPE_SAVE="/home/arxiv_code/tokenizers_rust/model_bpe_${TAG}"

LEN_CSV="/home/arxiv_code/tokenizers_rust/loss_curve_lenmax_${TAG}.csv"
BPE_CSV="/home/arxiv_code/tokenizers_rust/loss_curve_bpe_${TAG}.csv"

LOSS_OVERLAY_PNG="/home/arxiv_code/tokenizers_rust/loss_overlay_lenmax_vs_bpe_${TAG}.png"
LOSS_OVERLAY_PDF="/home/arxiv_code/tokenizers_rust/loss_overlay_lenmax_vs_bpe_${TAG}.pdf"
BPC_OVERLAY_PNG="/home/arxiv_code/tokenizers_rust/bpc_overlay_lenmax_vs_bpe_${TAG}.png"
BPC_OVERLAY_PDF="/home/arxiv_code/tokenizers_rust/bpc_overlay_lenmax_vs_bpe_${TAG}.pdf"

SYNC_DIR="/home/arxiv_code/tokenizers_rust/loss_sync_${TAG}"

echo "[pipeline] start $(date -Is)" | tee -a "$PIPE_LOG"
echo "[pipeline] train=$CORPUS_TRAIN eval=$CORPUS_EVAL" | tee -a "$PIPE_LOG"
echo "[pipeline] TPC_LEN=$TPC_LEN TPC_BPE=$TPC_BPE" | tee -a "$PIPE_LOG"
echo "[pipeline] BPE:     seq_len=$SEQ_BPE steps=$STEPS_BPE warmup=$WARMUP_BPE" | tee -a "$PIPE_LOG"
echo "[pipeline] Length:  seq_len=$SEQ_LEN steps=$STEPS_LEN warmup=$WARMUP_LEN" | tee -a "$PIPE_LOG"
echo "[pipeline] pack_chars=$PACK_CHARS bs=$BATCH_SIZE precision=$PRECISION" | tee -a "$PIPE_LOG"

run_one() {
  local tok_dir="$1"
  local out_log="$2"
  local save_dir="$3"
  local seq_len="$4"
  local steps="$5"
  local warmup="$6"
  local name="$7"
  if [[ -f "$out_log" ]] && grep -q "^== saved ==$" "$out_log"; then
    echo "[pipeline] skip $name (already finished): $out_log" | tee -a "$PIPE_LOG"
    return 0
  fi
  echo "[pipeline] start $name -> $out_log" | tee -a "$PIPE_LOG"
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
    --seq_len "$seq_len" \
    --batch_size "$BATCH_SIZE" \
    --steps "$steps" \
    --hidden_size 512 \
    --num_layers 8 \
    --num_heads 8 \
    --num_kv_heads 8 \
    --intermediate_size 1408 \
    --print_every "$PRINT_EVERY" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --pack_chars "$PACK_CHARS" \
    --pack_mode contiguous \
    --force_eos \
    --lr "$LR" \
    --lr_schedule "$LR_SCHEDULE" \
    --warmup_steps "$warmup" \
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
  echo "[pipeline] done $name" | tee -a "$PIPE_LOG"
}

# Run BPE baseline first (defines compute budget), then Length-MAX "more text" run
run_one "$BPE_TOK_DIR" "$BPE_LOG" "$BPE_SAVE" "$SEQ_BPE" "$STEPS_BPE" "$WARMUP_BPE" "BPE"
run_one "$LEN_TOK_DIR" "$LEN_LOG" "$LEN_SAVE" "$SEQ_LEN" "$STEPS_LEN" "$WARMUP_LEN" "Length-MAX"

echo "[pipeline] extract curves" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$BPE_LOG" --out_csv "$BPE_CSV" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$LEN_LOG" --out_csv "$LEN_CSV" | tee -a "$PIPE_LOG"

echo "[pipeline] plot overlays (step axis)" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/plot_loss_overlay.py \
  --csv "$LEN_CSV" --label "Length-MAX (compute-eq moretext, sl=${SEQ_LEN})" \
  --csv "$BPE_CSV" --label "BPE (baseline, sl=${SEQ_BPE})" \
  --out_png "$LOSS_OVERLAY_PNG" \
  --out_pdf "$LOSS_OVERLAY_PDF" \
  --title "Loss vs step (validation-eval, compute-equal moretext v026)"

python3 /home/arxiv_code/tokenizers_rust/plot_loss_overlay.py \
  --y_col bpc \
  --csv "$LEN_CSV" --label "Length-MAX (compute-eq moretext, sl=${SEQ_LEN})" \
  --csv "$BPE_CSV" --label "BPE (baseline, sl=${SEQ_BPE})" \
  --out_png "$BPC_OVERLAY_PNG" \
  --out_pdf "$BPC_OVERLAY_PDF" \
  --title "Train bpc proxy vs step (note: best metric uses eval_bpc on validation)"

echo "[pipeline] delta sync plots (step-grid mismatch may fail; ignore if so)" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/analyze_loss_sync.py \
  --a_csv "$LEN_CSV" --b_csv "$BPE_CSV" \
  --a_name lenmax_moretext --b_name bpe_baseline \
  --tpc_a "$TPC_LEN" --tpc_b "$TPC_BPE" \
  --out_dir "$SYNC_DIR" | tee -a "$PIPE_LOG" || true

echo "[pipeline] DONE $(date -Is)" | tee -a "$PIPE_LOG"







