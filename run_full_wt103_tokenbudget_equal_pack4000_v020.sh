#!/usr/bin/env bash
set -euo pipefail

# Token-budget-equal Llama training:
# Goal: over 10k steps, make effective supervised tokens equal across tokenizers
# by making padding ~0 (active_frac≈1.0) for BOTH tokenizers.
#
# Strategy: use the SAME seq_len/batch_size/steps and set pack_chars high enough
# (here 4000) so both tokenizers almost always hit max_length=seq_len with truncation,
# hence no pad tokens (labels=-100) → equal active tokens.

CORPUS="/home/arxiv_code/datasets/wikitext103_raw_txt/train.txt"

# Tokenizers (already trained)
LEN_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_wikitext103_32k_n9_full_v019"
BPE_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_bpe_wikitext103_32k_full_v019"

# Experiment tag
PACK_CHARS=4000
TAG="tokfixed_pack${PACK_CHARS}_steps10000"

# Logs
PIPE_LOG="/home/arxiv_code/tokenizers_rust/pipeline_tokfixed_pack${PACK_CHARS}_v020.log"
LEN_MODEL_LOG="/home/arxiv_code/tokenizers_rust/run_llama_full_lenmax_wt103_full_ddp2_h512l8_sl512_bs64_steps10000_bf16_pack${PACK_CHARS}.log"
BPE_MODEL_LOG="/home/arxiv_code/tokenizers_rust/run_llama_full_bpe_wt103_full_ddp2_h512l8_sl512_bs64_steps10000_bf16_pack${PACK_CHARS}.log"

# Curves / plots
LEN_CURVE="/home/arxiv_code/tokenizers_rust/loss_curve_lenmax_full_${TAG}.csv"
BPE_CURVE="/home/arxiv_code/tokenizers_rust/loss_curve_bpe_full_${TAG}.csv"

LOSS_OVERLAY_PNG="/home/arxiv_code/tokenizers_rust/loss_overlay_lenmax_vs_bpe_full_${TAG}.png"
LOSS_OVERLAY_PDF="/home/arxiv_code/tokenizers_rust/loss_overlay_lenmax_vs_bpe_full_${TAG}.pdf"
BPC_OVERLAY_PNG="/home/arxiv_code/tokenizers_rust/bpc_overlay_lenmax_vs_bpe_full_${TAG}.png"
BPC_OVERLAY_PDF="/home/arxiv_code/tokenizers_rust/bpc_overlay_lenmax_vs_bpe_full_${TAG}.pdf"

echo "[pipeline] start $(date -Is)" | tee -a "$PIPE_LOG"
echo "[pipeline] corpus=$CORPUS" | tee -a "$PIPE_LOG"
echo "[pipeline] pack_chars=$PACK_CHARS" | tee -a "$PIPE_LOG"
echo "[pipeline] len_tok_dir=$LEN_TOK_DIR" | tee -a "$PIPE_LOG"
echo "[pipeline] bpe_tok_dir=$BPE_TOK_DIR" | tee -a "$PIPE_LOG"

run_model() {
  local tok_dir="$1"
  local out_log="$2"
  local tag="$3"
  if [[ -f "$out_log" ]] && grep -q "== generate ==" "$out_log"; then
    echo "[pipeline] skip $tag model (already finished): $out_log" | tee -a "$PIPE_LOG"
    return 0
  fi
  echo "[pipeline] start $tag model -> $out_log" | tee -a "$PIPE_LOG"
  PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 \
    /home/arxiv_code/tokenizers_rust/validate_modern_arch_llama.py \
    --tokenizer_dir "$tok_dir" \
    --corpus_file "$CORPUS" \
    --max_lines 0 \
    --seed 42 \
    --device cuda \
    --precision bf16 \
    --seq_len 512 \
    --batch_size 64 \
    --steps 10000 \
    --hidden_size 512 \
    --num_layers 8 \
    --num_heads 8 \
    --num_kv_heads 8 \
    --intermediate_size 1408 \
    --print_every 200 \
    --max_new_tokens 160 \
    --pack_chars "$PACK_CHARS" \
    --tpc_estimate_lines 20000 \
    > "$out_log" 2>&1
  echo "[pipeline] done $tag model" | tee -a "$PIPE_LOG"
}

# Run BPE first, then Length-MAX
run_model "$BPE_TOK_DIR" "$BPE_MODEL_LOG" "BPE"
run_model "$LEN_TOK_DIR" "$LEN_MODEL_LOG" "Length-MAX"

echo "[pipeline] extract loss curves" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$LEN_MODEL_LOG" --out_csv "$LEN_CURVE" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$BPE_MODEL_LOG" --out_csv "$BPE_CURVE" | tee -a "$PIPE_LOG"

echo "[pipeline] plot overlays" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/plot_loss_overlay.py \
  --csv "$LEN_CURVE" --label "Length-MAX (full, pack=${PACK_CHARS})" \
  --csv "$BPE_CURVE" --label "BPE (full, pack=${PACK_CHARS})" \
  --out_png "$LOSS_OVERLAY_PNG" \
  --out_pdf "$LOSS_OVERLAY_PDF" \
  --title "Loss vs step (token-budget equal, full corpus, Llama-style 58.5M, DDP 2x5060Ti, bf16, seq_len=512, bs=64)"

python3 /home/arxiv_code/tokenizers_rust/plot_loss_overlay.py \
  --y_col bpc \
  --csv "$LEN_CURVE" --label "Length-MAX (full, pack=${PACK_CHARS})" \
  --csv "$BPE_CURVE" --label "BPE (full, pack=${PACK_CHARS})" \
  --out_png "$BPC_OVERLAY_PNG" \
  --out_pdf "$BPC_OVERLAY_PDF" \
  --title "Bits/char proxy vs step (token-budget equal, bpc≈loss*TPC/ln2)"

echo "[pipeline] DONE $(date -Is)" | tee -a "$PIPE_LOG"








