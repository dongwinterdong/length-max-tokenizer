#!/usr/bin/env bash
set -euo pipefail

# Sweep max_token_chars values for Length-MAX and run full model training comparisons.
#
# Strategy:
# - Reuse a fixed BPE tokenizer (v019).
# - Reuse (or train once) a fixed BPE model baseline with eval on validation.txt, pack4000.
# - For each max_token_chars value:
#   - Train a new Length-MAX tokenizer (unless max=0 -> reuse v019 tokenizer).
#   - Train a Length-MAX model with identical hyperparams.
#   - Record best eval_bpc (validation) + tokenizer TPC for each setting.
#
# WARNING: This is expensive (tokenizer training on full train.txt is hours).

CORPUS_TRAIN="/home/arxiv_code/datasets/wikitext103_raw_txt/train.txt"
CORPUS_EVAL="/home/arxiv_code/datasets/wikitext103_raw_txt/validation.txt"

BPE_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_bpe_wikitext103_32k_full_v019"

# BPE baseline model (reuse from v027 if present)
BPE_MODEL_LOG_DEFAULT="/home/arxiv_code/tokenizers_rust/run_llama_full_bpe_wt103_full_ddp2_h512l8_sl512_bs64_steps10000_bf16_pack4000_lenmax_maxchars64_full_tokfixed_pack4000_evalval_steps10000_v027.log"

# Training hyperparams (same for all)
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

# Length tokenizer training params
LEN_NUM_MERGES=40000
LEN_AIM_VOCAB=32000
LEN_N_MAX=9
LEN_WORKERS=64
LEN_MULTI_PROCESS=1
LEN_USE_HEAP=0

# Sweep list (edit as needed). "0" means no limit (reuse existing v019 tokenizer).
MAX_LIST=("0" "32" "48")

TAG="lenmax_maxchars_sweep_full_tokfixed_pack${PACK_CHARS}_evalval_steps${STEPS}_v028"
PIPE_LOG="/home/arxiv_code/tokenizers_rust/pipeline_${TAG}.log"
SUMMARY_CSV="/home/arxiv_code/tokenizers_rust/summary_${TAG}.csv"

echo "[pipeline] start $(date -Is)" | tee -a "$PIPE_LOG"
echo "[pipeline] train=$CORPUS_TRAIN eval=$CORPUS_EVAL" | tee -a "$PIPE_LOG"
echo "[pipeline] sweep max_token_chars=${MAX_LIST[*]}" | tee -a "$PIPE_LOG"
echo "[pipeline] summary_csv=$SUMMARY_CSV" | tee -a "$PIPE_LOG"

wait_for_file() {
  local f="$1"
  local name="$2"
  echo "[pipeline] waiting for $name: $f" | tee -a "$PIPE_LOG"
  while [[ ! -f "$f" ]]; do
    sleep 60
  done
  echo "[pipeline] ready: $name" | tee -a "$PIPE_LOG"
}

best_eval_bpc() {
  local log="$1"
  python3 - <<PY
import re
from pathlib import Path
log=Path("${log}")
pat=re.compile(r'^== eval ==.*eval_bpc≈([0-9.]+)') 
vals=[]
for line in log.read_text().splitlines():
    m=pat.match(line.strip())
    if m:
        vals.append(float(m.group(1)))
print(min(vals) if vals else "nan")
PY
}

train_len_tokenizer() {
  local max_chars="$1"
  local out_dir="$2"
  local out_log="$3"
  if [[ -f "$out_dir/vocab.json" ]]; then
    echo "[pipeline] skip Length-MAX tokenizer max_chars=$max_chars (exists): $out_dir" | tee -a "$PIPE_LOG"
    return 0
  fi
  echo "[pipeline] start Length-MAX tokenizer max_chars=$max_chars -> $out_log" | tee -a "$PIPE_LOG"
  mkdir -p "$out_dir"
  python3 - <<PY > "$out_log" 2>&1
from pathlib import Path
import length_tokenizer_rs
print("length_tokenizer_rs", getattr(length_tokenizer_rs, "__version__", None))

corpus = Path("${CORPUS_TRAIN}")
out_dir = Path("${out_dir}")
out_dir.mkdir(parents=True, exist_ok=True)

length_tokenizer_rs.train_to_hf(
    corpus_file=str(corpus),
    out_dir=str(out_dir),
    num_merges=int(${LEN_NUM_MERGES}),
    aim_token_num=int(${LEN_AIM_VOCAB}),
    n_max=int(${LEN_N_MAX}),
    max_token_chars=int(${max_chars}),
    num_workers=int(${LEN_WORKERS}),
    multi_process=bool(${LEN_MULTI_PROCESS}),
    use_heap=bool(${LEN_USE_HEAP}),
)
print("DONE vocab.json bytes=", (out_dir/"vocab.json").stat().st_size)
PY
  echo "[pipeline] done Length-MAX tokenizer max_chars=$max_chars" | tee -a "$PIPE_LOG"
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

ensure_bpe_baseline() {
  local log="$1"
  if [[ -f "$log" ]] && grep -q "^== saved ==$" "$log"; then
    echo "[pipeline] reuse BPE baseline: $log" | tee -a "$PIPE_LOG"
    return 0
  fi
  # If baseline missing, train it once with this pipeline tag.
  local out_log="/home/arxiv_code/tokenizers_rust/run_llama_bpe_${TAG}.log"
  local save_dir="/home/arxiv_code/tokenizers_rust/model_bpe_${TAG}"
  run_model "$BPE_TOK_DIR" "$out_log" "$save_dir" "$STEPS" "BPE(baseline)"
  echo "$out_log"
}

wait_for_file "$BPE_TOK_DIR/tokenizer.json" "BPE tokenizer.json"

# Decide baseline
BPE_MODEL_LOG="$BPE_MODEL_LOG_DEFAULT"
if [[ ! -f "$BPE_MODEL_LOG" ]] || ! grep -q "^== saved ==$" "$BPE_MODEL_LOG"; then
  echo "[pipeline] WARNING: default BPE baseline missing; training a new one" | tee -a "$PIPE_LOG"
  BPE_MODEL_LOG="/home/arxiv_code/tokenizers_rust/run_llama_bpe_${TAG}.log"
  BPE_SAVE_DIR="/home/arxiv_code/tokenizers_rust/model_bpe_${TAG}"
  run_model "$BPE_TOK_DIR" "$BPE_MODEL_LOG" "$BPE_SAVE_DIR" "$STEPS" "BPE(baseline)"
fi

# Extract BPE curve (once)
BPE_CURVE="/home/arxiv_code/tokenizers_rust/loss_curve_bpe_${TAG}.csv"
if [[ ! -f "$BPE_CURVE" ]]; then
  python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$BPE_MODEL_LOG" --out_csv "$BPE_CURVE" | tee -a "$PIPE_LOG"
fi

echo "max_token_chars,len_tok_dir,len_tpc,bpe_tpc,bpe_best_eval_bpc,len_best_eval_bpc,len_model_log" > "$SUMMARY_CSV"
echo "[pipeline] BPE best_eval_bpc=$(best_eval_bpc "$BPE_MODEL_LOG")" | tee -a "$PIPE_LOG"

for max_chars in "${MAX_LIST[@]}"; do
  if [[ "$max_chars" == "0" ]]; then
    LEN_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_wikitext103_32k_n9_full_v019"
    LEN_TAG="maxchars0"
    LEN_TOK_LOG="/home/arxiv_code/tokenizers_rust/train_wikitext103_32k_n9_full_maxchars0_reuse_v019.log"
  else
    LEN_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_wikitext103_32k_n9_full_maxchars${max_chars}_v028"
    LEN_TAG="maxchars${max_chars}"
    LEN_TOK_LOG="/home/arxiv_code/tokenizers_rust/train_wikitext103_32k_n9_full_maxchars${max_chars}_v028.log"
    train_len_tokenizer "$max_chars" "$LEN_TOK_DIR" "$LEN_TOK_LOG"
  fi

  wait_for_file "$LEN_TOK_DIR/vocab.json" "Length-MAX vocab.json ($LEN_TAG)"

  # Token totals
  TOK_COUNTS="/home/arxiv_code/tokenizers_rust/token_counts_lenmax_${LEN_TAG}_${TAG}.txt"
  python3 /home/arxiv_code/tokenizers_rust/count_full_corpus_token_totals.py \
    --corpus "$CORPUS_TRAIN" \
    --len_tok_dir "$LEN_TOK_DIR" \
    --bpe_tok_dir "$BPE_TOK_DIR" \
    --batch_size 1024 \
    --report_every 200000 \
    > "$TOK_COUNTS" 2>&1 || true

  # Parse TPCs from token count output
  LEN_TPC=$(grep -E '^len_tokens' "$TOK_COUNTS" | sed -E 's/.*tpc=([0-9.]+).*/\1/' | tail -n 1)
  BPE_TPC=$(grep -E '^bpe_tokens' "$TOK_COUNTS" | sed -E 's/.*tpc=([0-9.]+).*/\1/' | tail -n 1)

  # Train model
  LEN_MODEL_LOG="/home/arxiv_code/tokenizers_rust/run_llama_lenmax_${LEN_TAG}_${TAG}.log"
  LEN_SAVE_DIR="/home/arxiv_code/tokenizers_rust/model_lenmax_${LEN_TAG}_${TAG}"
  run_model "$LEN_TOK_DIR" "$LEN_MODEL_LOG" "$LEN_SAVE_DIR" "$STEPS" "Length-MAX(${LEN_TAG})"

  # Curves + overlay
  LEN_CURVE="/home/arxiv_code/tokenizers_rust/loss_curve_lenmax_${LEN_TAG}_${TAG}.csv"
  python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$LEN_MODEL_LOG" --out_csv "$LEN_CURVE" | tee -a "$PIPE_LOG"

  python3 /home/arxiv_code/tokenizers_rust/plot_loss_overlay.py \
    --csv "$LEN_CURVE" --label "Length-MAX (${LEN_TAG})" \
    --csv "$BPE_CURVE" --label "BPE (baseline)" \
    --out_png "/home/arxiv_code/tokenizers_rust/loss_overlay_lenmax_vs_bpe_${LEN_TAG}_${TAG}.png" \
    --out_pdf "/home/arxiv_code/tokenizers_rust/loss_overlay_lenmax_vs_bpe_${LEN_TAG}_${TAG}.pdf" \
    --title "Loss vs step (${LEN_TAG}, eval=validation, pack=${PACK_CHARS})"

  python3 /home/arxiv_code/tokenizers_rust/plot_loss_overlay.py \
    --y_col bpc \
    --csv "$LEN_CURVE" --label "Length-MAX (${LEN_TAG})" \
    --csv "$BPE_CURVE" --label "BPE (baseline)" \
    --out_png "/home/arxiv_code/tokenizers_rust/bpc_overlay_lenmax_vs_bpe_${LEN_TAG}_${TAG}.png" \
    --out_pdf "/home/arxiv_code/tokenizers_rust/bpc_overlay_lenmax_vs_bpe_${LEN_TAG}_${TAG}.pdf" \
    --title "Train bpc proxy vs step (${LEN_TAG}; quality uses eval_bpc on validation)"

  LEN_BEST=$(best_eval_bpc "$LEN_MODEL_LOG")
  BPE_BEST=$(best_eval_bpc "$BPE_MODEL_LOG")
  echo "${max_chars},${LEN_TOK_DIR},${LEN_TPC},${BPE_TPC},${BPE_BEST},${LEN_BEST},${LEN_MODEL_LOG}" >> "$SUMMARY_CSV"
  echo "[pipeline] ${LEN_TAG}: tpc=${LEN_TPC} best_eval_bpc=${LEN_BEST} (BPE best=${BPE_BEST})" | tee -a "$PIPE_LOG"
done

echo "[pipeline] DONE $(date -Is)" | tee -a "$PIPE_LOG"
echo "[pipeline] summary_csv=$SUMMARY_CSV" | tee -a "$PIPE_LOG"


