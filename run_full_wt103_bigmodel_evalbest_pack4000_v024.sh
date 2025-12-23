#!/usr/bin/env bash
set -euo pipefail

# Bigger-model experiment (token-budget equal) with eval-best checkpointing.
#
# Hypothesis: Length-MAX may benefit more from larger capacity (longer/rarer tokens),
# potentially narrowing loss and improving bpc enough to beat BPE.

CORPUS="/home/arxiv_code/datasets/wikitext103_raw_txt/train.txt"

# Tokenizers (already trained)
LEN_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_wikitext103_32k_n9_full_v019"
BPE_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_bpe_wikitext103_32k_full_v019"

# Token-budget equalization
PACK_CHARS=4000

# Training
STEPS=10000
SEQ_LEN=512
BATCH_SIZE=64
PRINT_EVERY=200
PRECISION=bf16

# Bigger Llama-style model
HIDDEN_SIZE=768
NUM_LAYERS=12
NUM_HEADS=12
NUM_KV_HEADS=12
INTERMEDIATE_SIZE=2048

# Optim
LR=3e-4
LR_SCHEDULE=cosine
WARMUP_STEPS=200
MIN_LR=3e-5
WEIGHT_DECAY=0.1
ADAM_BETA2=0.95
GRAD_CLIP=1.0

# Eval-best (rank0 only)
EVAL_EVERY=200
EVAL_BATCHES=8
EVAL_BATCH_SIZE=32
EVAL_SEED=12345
SAVE_BEST_METRIC=bpc
SAVE_BEST_ON=eval
SAVE_BEST_MIN_DELTA=0.0002

# Generation (from best)
GEN_FROM_BEST=1
GEN_PROMPT="= Valkyria Chronicles III ="
GEN_T=0.7
GEN_TOP_P=0.9
GEN_REP=1.12
GEN_NREP=3
MAX_NEW_TOKENS=160

TAG="big_h${HIDDEN_SIZE}l${NUM_LAYERS}_tokfixed_pack${PACK_CHARS}_evalbest_steps${STEPS}_v024"

# Logs
PIPE_LOG="/home/arxiv_code/tokenizers_rust/pipeline_${TAG}.log"
LEN_MODEL_LOG="/home/arxiv_code/tokenizers_rust/run_llama_big_lenmax_wt103_full_ddp2_h${HIDDEN_SIZE}l${NUM_LAYERS}_sl${SEQ_LEN}_bs${BATCH_SIZE}_steps${STEPS}_${PRECISION}_pack${PACK_CHARS}_${TAG}.log"
BPE_MODEL_LOG="/home/arxiv_code/tokenizers_rust/run_llama_big_bpe_wt103_full_ddp2_h${HIDDEN_SIZE}l${NUM_LAYERS}_sl${SEQ_LEN}_bs${BATCH_SIZE}_steps${STEPS}_${PRECISION}_pack${PACK_CHARS}_${TAG}.log"

# Model dirs
LEN_SAVE_DIR="/home/arxiv_code/tokenizers_rust/model_lenmax_full_${TAG}"
BPE_SAVE_DIR="/home/arxiv_code/tokenizers_rust/model_bpe_full_${TAG}"

# Curves / plots
LEN_CURVE="/home/arxiv_code/tokenizers_rust/loss_curve_lenmax_full_${TAG}.csv"
BPE_CURVE="/home/arxiv_code/tokenizers_rust/loss_curve_bpe_full_${TAG}.csv"
LOSS_OVERLAY_PNG="/home/arxiv_code/tokenizers_rust/loss_overlay_lenmax_vs_bpe_full_${TAG}.png"
LOSS_OVERLAY_PDF="/home/arxiv_code/tokenizers_rust/loss_overlay_lenmax_vs_bpe_full_${TAG}.pdf"
BPC_OVERLAY_PNG="/home/arxiv_code/tokenizers_rust/bpc_overlay_lenmax_vs_bpe_full_${TAG}.png"
BPC_OVERLAY_PDF="/home/arxiv_code/tokenizers_rust/bpc_overlay_lenmax_vs_bpe_full_${TAG}.pdf"

SYNC_DIR="/home/arxiv_code/tokenizers_rust/loss_sync_full_${TAG}"

echo "[pipeline] start $(date -Is)" | tee -a "$PIPE_LOG"
echo "[pipeline] corpus=$CORPUS" | tee -a "$PIPE_LOG"
echo "[pipeline] pack_chars=$PACK_CHARS" | tee -a "$PIPE_LOG"
echo "[pipeline] steps=$STEPS seq_len=$SEQ_LEN bs=$BATCH_SIZE precision=$PRECISION" | tee -a "$PIPE_LOG"
echo "[pipeline] model: h=$HIDDEN_SIZE layers=$NUM_LAYERS heads=$NUM_HEADS kv_heads=$NUM_KV_HEADS intermediate=$INTERMEDIATE_SIZE" | tee -a "$PIPE_LOG"
echo "[pipeline] len_tok_dir=$LEN_TOK_DIR" | tee -a "$PIPE_LOG"
echo "[pipeline] bpe_tok_dir=$BPE_TOK_DIR" | tee -a "$PIPE_LOG"
echo "[pipeline] tag=$TAG" | tee -a "$PIPE_LOG"

run_model() {
  local tok_dir="$1"
  local out_log="$2"
  local save_dir="$3"
  local name="$4"
  if [[ -f "$out_log" ]] && grep -q "^== saved ==$" "$out_log"; then
    echo "[pipeline] skip $name (already finished): $out_log" | tee -a "$PIPE_LOG"
    return 0
  fi
  echo "[pipeline] start $name -> $out_log" | tee -a "$PIPE_LOG"
  # Note: big models can be memory tight on 16GB GPUs; enable grad checkpointing.
  # Also set expandable_segments to reduce fragmentation risk.
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=2 \
    /home/arxiv_code/tokenizers_rust/validate_modern_arch_llama.py \
    --tokenizer_dir "$tok_dir" \
    --corpus_file "$CORPUS" \
    --max_lines 0 \
    --seed 42 \
    --device cuda \
    --precision "$PRECISION" \
    --grad_checkpointing \
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
  echo "[pipeline] done $name" | tee -a "$PIPE_LOG"
}

# Run Length-MAX first, then BPE
run_model "$LEN_TOK_DIR" "$LEN_MODEL_LOG" "$LEN_SAVE_DIR" "Length-MAX"
run_model "$BPE_TOK_DIR" "$BPE_MODEL_LOG" "$BPE_SAVE_DIR" "BPE"

echo "[pipeline] extract loss curves" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$LEN_MODEL_LOG" --out_csv "$LEN_CURVE" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$BPE_MODEL_LOG" --out_csv "$BPE_CURVE" | tee -a "$PIPE_LOG"

echo "[pipeline] plot overlays" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/plot_loss_overlay.py \
  --csv "$LEN_CURVE" --label "Length-MAX (big, pack=${PACK_CHARS})" \
  --csv "$BPE_CURVE" --label "BPE (big, pack=${PACK_CHARS})" \
  --out_png "$LOSS_OVERLAY_PNG" \
  --out_pdf "$LOSS_OVERLAY_PDF" \
  --title "Loss vs step (big model, token-budget equal, full corpus, DDP2, bf16, seq_len=${SEQ_LEN}, bs=${BATCH_SIZE})"

python3 /home/arxiv_code/tokenizers_rust/plot_loss_overlay.py \
  --y_col bpc \
  --csv "$LEN_CURVE" --label "Length-MAX (big, pack=${PACK_CHARS})" \
  --csv "$BPE_CURVE" --label "BPE (big, pack=${PACK_CHARS})" \
  --out_png "$BPC_OVERLAY_PNG" \
  --out_pdf "$BPC_OVERLAY_PDF" \
  --title "Bits/char proxy vs step (big model, bpc≈loss*TPC/ln2)"

echo "[pipeline] delta sync plots" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/analyze_loss_sync.py \
  --a_csv "$LEN_CURVE" --b_csv "$BPE_CURVE" \
  --a_name lenmax_big --b_name bpe_big \
  --tpc_a 0.176099 --tpc_b 0.211906 \
  --out_dir "$SYNC_DIR" | tee -a "$PIPE_LOG"

echo "[pipeline] DONE $(date -Is)" | tee -a "$PIPE_LOG"


