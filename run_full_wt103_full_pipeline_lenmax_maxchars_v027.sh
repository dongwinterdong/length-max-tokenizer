#!/usr/bin/env bash
set -euo pipefail

# Full pipeline (v027): train a new Length-MAX tokenizer with max_token_chars constraint,
# then train matched Llama-style models (BPE vs Length-MAX) on full WikiText-103.
#
# Notes:
# - BPE tokenizer is reused (already trained).
# - Models use pack_chars=4000 + force_eos + contiguous packing to make token budget comparable.
# - Eval is on held-out validation.txt (best checkpoint tracked via eval_bpc).

CORPUS_TRAIN="/home/arxiv_code/datasets/wikitext103_raw_txt/train.txt"
CORPUS_EVAL="/home/arxiv_code/datasets/wikitext103_raw_txt/validation.txt"

# Tokenizers
LEN_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_wikitext103_32k_n9_full_maxchars64_v027"
BPE_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_bpe_wikitext103_32k_full_v019"

# Length-MAX tokenizer training params
LEN_NUM_MERGES=40000
LEN_AIM_VOCAB=32000
LEN_N_MAX=9
LEN_MAX_TOKEN_CHARS=64
LEN_WORKERS=64
LEN_MULTI_PROCESS=1
LEN_USE_HEAP=0

# Model training params (same for both)
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

TAG="lenmax_maxchars${LEN_MAX_TOKEN_CHARS}_full_tokfixed_pack${PACK_CHARS}_evalval_steps${STEPS}_v027"

# Logs / outputs
PIPE_LOG="/home/arxiv_code/tokenizers_rust/pipeline_${TAG}.log"
LEN_TOK_LOG="/home/arxiv_code/tokenizers_rust/train_wikitext103_32k_n9_full_maxchars${LEN_MAX_TOKEN_CHARS}_v027.log"

LEN_MODEL_LOG="/home/arxiv_code/tokenizers_rust/run_llama_full_lenmax_wt103_full_ddp2_h${HIDDEN_SIZE}l${NUM_LAYERS}_sl${SEQ_LEN}_bs${BATCH_SIZE}_steps${STEPS}_${PRECISION}_pack${PACK_CHARS}_${TAG}.log"
BPE_MODEL_LOG="/home/arxiv_code/tokenizers_rust/run_llama_full_bpe_wt103_full_ddp2_h${HIDDEN_SIZE}l${NUM_LAYERS}_sl${SEQ_LEN}_bs${BATCH_SIZE}_steps${STEPS}_${PRECISION}_pack${PACK_CHARS}_${TAG}.log"

LEN_SAVE_DIR="/home/arxiv_code/tokenizers_rust/model_lenmax_${TAG}"
BPE_SAVE_DIR="/home/arxiv_code/tokenizers_rust/model_bpe_${TAG}"

LEN_CURVE="/home/arxiv_code/tokenizers_rust/loss_curve_lenmax_${TAG}.csv"
BPE_CURVE="/home/arxiv_code/tokenizers_rust/loss_curve_bpe_${TAG}.csv"

LOSS_OVERLAY_PNG="/home/arxiv_code/tokenizers_rust/loss_overlay_lenmax_vs_bpe_${TAG}.png"
LOSS_OVERLAY_PDF="/home/arxiv_code/tokenizers_rust/loss_overlay_lenmax_vs_bpe_${TAG}.pdf"
BPC_OVERLAY_PNG="/home/arxiv_code/tokenizers_rust/bpc_overlay_lenmax_vs_bpe_${TAG}.png"
BPC_OVERLAY_PDF="/home/arxiv_code/tokenizers_rust/bpc_overlay_lenmax_vs_bpe_${TAG}.pdf"

SYNC_DIR="/home/arxiv_code/tokenizers_rust/loss_sync_${TAG}"
TOKEN_COUNTS="/home/arxiv_code/tokenizers_rust/token_counts_${TAG}.txt"

echo "[pipeline] start $(date -Is)" | tee -a "$PIPE_LOG"
echo "[pipeline] train=$CORPUS_TRAIN eval=$CORPUS_EVAL" | tee -a "$PIPE_LOG"
echo "[pipeline] len_tok_dir=$LEN_TOK_DIR" | tee -a "$PIPE_LOG"
echo "[pipeline] bpe_tok_dir=$BPE_TOK_DIR" | tee -a "$PIPE_LOG"
echo "[pipeline] len tokenizer: aim_vocab=$LEN_AIM_VOCAB n_max=$LEN_N_MAX max_token_chars=$LEN_MAX_TOKEN_CHARS workers=$LEN_WORKERS multi_process=$LEN_MULTI_PROCESS" | tee -a "$PIPE_LOG"

wait_for_file() {
  local f="$1"
  local name="$2"
  echo "[pipeline] waiting for $name: $f" | tee -a "$PIPE_LOG"
  while [[ ! -f "$f" ]]; do
    sleep 60
  done
  echo "[pipeline] ready: $name" | tee -a "$PIPE_LOG"
}

train_len_tokenizer() {
  if [[ -f "$LEN_TOK_DIR/vocab.json" ]]; then
    echo "[pipeline] skip Length-MAX tokenizer (already exists): $LEN_TOK_DIR" | tee -a "$PIPE_LOG"
    return 0
  fi
  echo "[pipeline] start Length-MAX tokenizer -> $LEN_TOK_LOG" | tee -a "$PIPE_LOG"
  mkdir -p "$LEN_TOK_DIR"
  python3 - <<PY > "$LEN_TOK_LOG" 2>&1
from pathlib import Path
import length_tokenizer_rs

print("length_tokenizer_rs", getattr(length_tokenizer_rs, "__version__", None))

corpus = Path("${CORPUS_TRAIN}")
out_dir = Path("${LEN_TOK_DIR}")
out_dir.mkdir(parents=True, exist_ok=True)

length_tokenizer_rs.train_to_hf(
    corpus_file=str(corpus),
    out_dir=str(out_dir),
    num_merges=int(${LEN_NUM_MERGES}),
    aim_token_num=int(${LEN_AIM_VOCAB}),
    n_max=int(${LEN_N_MAX}),
    max_token_chars=int(${LEN_MAX_TOKEN_CHARS}),
    num_workers=int(${LEN_WORKERS}),
    multi_process=bool(${LEN_MULTI_PROCESS}),
    use_heap=bool(${LEN_USE_HEAP}),
)
print("DONE vocab.json bytes=", (out_dir/"vocab.json").stat().st_size)
PY
  echo "[pipeline] done Length-MAX tokenizer" | tee -a "$PIPE_LOG"
}

run_model() {
  local tok_dir="$1"
  local out_log="$2"
  local save_dir="$3"
  local name="$4"
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
    --steps "$STEPS" \
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

# 0) ensure BPE tokenizer exists (reused)
wait_for_file "$BPE_TOK_DIR/tokenizer.json" "BPE HF tokenizer.json"

# 1) train Length-MAX tokenizer (new)
train_len_tokenizer
wait_for_file "$LEN_TOK_DIR/vocab.json" "Length-MAX HF vocab.json"

# 2) token totals (exact full-corpus TPC)
echo "[pipeline] count full-corpus token totals" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/count_full_corpus_token_totals.py \
  --corpus "$CORPUS_TRAIN" \
  --len_tok_dir "$LEN_TOK_DIR" \
  --bpe_tok_dir "$BPE_TOK_DIR" \
  --batch_size 1024 \
  --report_every 200000 \
  > "$TOKEN_COUNTS" 2>&1 || true
echo "[pipeline] token totals written: $TOKEN_COUNTS" | tee -a "$PIPE_LOG"

# 3) train models (BPE first, then Length-MAX)
run_model "$BPE_TOK_DIR" "$BPE_MODEL_LOG" "$BPE_SAVE_DIR" "BPE"
run_model "$LEN_TOK_DIR" "$LEN_MODEL_LOG" "$LEN_SAVE_DIR" "Length-MAX(max_token_chars)"

# 4) curves + overlay
echo "[pipeline] extract loss curves" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$LEN_MODEL_LOG" --out_csv "$LEN_CURVE" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$BPE_MODEL_LOG" --out_csv "$BPE_CURVE" | tee -a "$PIPE_LOG"

echo "[pipeline] plot overlays" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/plot_loss_overlay.py \
  --csv "$LEN_CURVE" --label "Length-MAX (max_chars=${LEN_MAX_TOKEN_CHARS})" \
  --csv "$BPE_CURVE" --label "BPE (full)" \
  --out_png "$LOSS_OVERLAY_PNG" \
  --out_pdf "$LOSS_OVERLAY_PDF" \
  --title "Loss vs step (full corpus, held-out eval=validation, pack=${PACK_CHARS})"

python3 /home/arxiv_code/tokenizers_rust/plot_loss_overlay.py \
  --y_col bpc \
  --csv "$LEN_CURVE" --label "Length-MAX (max_chars=${LEN_MAX_TOKEN_CHARS})" \
  --csv "$BPE_CURVE" --label "BPE (full)" \
  --out_png "$BPC_OVERLAY_PNG" \
  --out_pdf "$BPC_OVERLAY_PDF" \
  --title "Train bpc proxy vs step (quality metric uses eval_bpc on validation)"

echo "[pipeline] delta sync plots (may fail if step grids differ)" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/analyze_loss_sync.py \
  --a_csv "$LEN_CURVE" --b_csv "$BPE_CURVE" \
  --a_name "lenmax_maxchars${LEN_MAX_TOKEN_CHARS}" --b_name "bpe_full" \
  --tpc_a 0.176099 --tpc_b 0.211906 \
  --out_dir "$SYNC_DIR" | tee -a "$PIPE_LOG" || true

echo "[pipeline] DONE $(date -Is)" | tee -a "$PIPE_LOG"







