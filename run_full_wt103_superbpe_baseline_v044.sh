#!/usr/bin/env bash
set -euo pipefail

# v044: 训练 SuperBPE baseline 模型（用于替代旧 BPE baseline）。
#
# 说明：
# - SuperBPE tokenizer 由 v043 训练产出（ByteLevel(use_regex=False) 允许跨空格 merge）
# - 模型训练超参与我们现有 head-to-head 保持一致（Llama-style 58.5M, pack=4000, steps=10000）

CORPUS_TRAIN="/home/arxiv_code/datasets/wikitext103_raw_txt/train.txt"
CORPUS_EVAL="/home/arxiv_code/datasets/wikitext103_raw_txt/validation.txt"

SUPER_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_superbpe_wikitext103_32000_full_v043"

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

TAG="superbpe_pack${PACK_CHARS}_evalval_steps${STEPS}_v044"

PIPE_LOG="/home/arxiv_code/tokenizers_rust/pipeline_${TAG}.log"
MODEL_LOG="/home/arxiv_code/tokenizers_rust/run_llama_superbpe_${TAG}.log"
SAVE_DIR="/home/arxiv_code/tokenizers_rust/model_superbpe_${TAG}"
CURVE="/home/arxiv_code/tokenizers_rust/loss_curve_superbpe_${TAG}.csv"

echo "[pipeline] start $(date -Is)" | tee -a "$PIPE_LOG"
echo "[pipeline] superbpe_tok_dir=$SUPER_TOK_DIR" | tee -a "$PIPE_LOG"
echo "[pipeline] train=$CORPUS_TRAIN eval=$CORPUS_EVAL" | tee -a "$PIPE_LOG"

wait_for_file() {
  local f="$1"
  local name="$2"
  echo "[pipeline] waiting for $name: $f" | tee -a "$PIPE_LOG"
  while [[ ! -f "$f" ]]; do
    sleep 60
  done
  echo "[pipeline] ready: $name" | tee -a "$PIPE_LOG"
}

wait_for_file "$SUPER_TOK_DIR/tokenizer.json" "SuperBPE HF tokenizer.json"

if [[ -f "$MODEL_LOG" ]] && grep -q "^== saved ==$" "$MODEL_LOG"; then
  echo "[pipeline] skip SuperBPE model (already finished): $MODEL_LOG" | tee -a "$PIPE_LOG"
  echo "[pipeline] DONE $(date -Is)" | tee -a "$PIPE_LOG"
  exit 0
fi

echo "[pipeline] start SuperBPE model -> $MODEL_LOG" | tee -a "$PIPE_LOG"
rm -f "$MODEL_LOG"

PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 \
  /home/arxiv_code/tokenizers_rust/validate_modern_arch_llama.py \
  --tokenizer_dir "$SUPER_TOK_DIR" \
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
  --save_dir "$SAVE_DIR" \
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
  > "$MODEL_LOG" 2>&1

echo "[pipeline] done SuperBPE model" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$MODEL_LOG" --out_csv "$CURVE" | tee -a "$PIPE_LOG"
echo "[pipeline] curve_csv=$CURVE" | tee -a "$PIPE_LOG"
echo "[pipeline] DONE $(date -Is)" | tee -a "$PIPE_LOG"


