#!/usr/bin/env bash
set -euo pipefail

# Full-corpus end-to-end pipeline:
# 1) Wait for Length-MAX full tokenizer training (already running) to finish and produce HF dir
# 2) Wait for BPE full tokenizer training to finish and produce HF dir
# 3) Train two matched Llama-style models on full corpus and export logs
# 4) Extract loss curves + plot overlay (loss + bpc)

CORPUS="/home/arxiv_code/datasets/wikitext103_raw_txt/train.txt"

# Tokenizers
LEN_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_wikitext103_32k_n9_full_v019"
BPE_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_bpe_wikitext103_32k_full_v019"

# Logs
PIPE_LOG="/home/arxiv_code/tokenizers_rust/pipeline_full_wt103_full_v019.log"

LEN_MODEL_LOG="/home/arxiv_code/tokenizers_rust/run_llama_full_lenmax_wt103_full_ddp2_h512l8_sl512_bs64_steps10000_bf16_pack3000.log"
BPE_MODEL_LOG="/home/arxiv_code/tokenizers_rust/run_llama_full_bpe_wt103_full_ddp2_h512l8_sl512_bs64_steps10000_bf16_pack3000.log"

LEN_CURVE="/home/arxiv_code/tokenizers_rust/loss_curve_lenmax_full_steps10000.csv"
BPE_CURVE="/home/arxiv_code/tokenizers_rust/loss_curve_bpe_full_steps10000.csv"

LOSS_OVERLAY_PNG="/home/arxiv_code/tokenizers_rust/loss_overlay_lenmax_vs_bpe_full_steps10000.png"
LOSS_OVERLAY_PDF="/home/arxiv_code/tokenizers_rust/loss_overlay_lenmax_vs_bpe_full_steps10000.pdf"
BPC_OVERLAY_PNG="/home/arxiv_code/tokenizers_rust/bpc_overlay_lenmax_vs_bpe_full_steps10000.png"
BPC_OVERLAY_PDF="/home/arxiv_code/tokenizers_rust/bpc_overlay_lenmax_vs_bpe_full_steps10000.pdf"

echo "[pipeline] start $(date -Is)" | tee -a "$PIPE_LOG"
echo "[pipeline] corpus=$CORPUS" | tee -a "$PIPE_LOG"
echo "[pipeline] len_tok_dir=$LEN_TOK_DIR" | tee -a "$PIPE_LOG"
echo "[pipeline] bpe_tok_dir=$BPE_TOK_DIR" | tee -a "$PIPE_LOG"

wait_for_file() {
  local f="$1"
  local name="$2"
  echo "[pipeline] waiting for $name: $f" | tee -a "$PIPE_LOG"
  while [[ ! -f "$f" ]]; do
    sleep 300
  done
  echo "[pipeline] ready: $name" | tee -a "$PIPE_LOG"
}

# 1) wait for BPE tokenizer first (usually faster), start BPE model as soon as it is ready
wait_for_file "$BPE_TOK_DIR/tokenizer.json" "BPE HF tokenizer.json"

# 2) train models (BPE first, then Length-MAX) – same hyperparams for head-to-head
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
    --pack_chars 3000 \
    --tpc_estimate_lines 20000 \
    > "$out_log" 2>&1
  echo "[pipeline] done $tag model" | tee -a "$PIPE_LOG"
}

run_model "$BPE_TOK_DIR" "$BPE_MODEL_LOG" "BPE"

# 3) wait for Length-MAX tokenizer, then train Length-MAX model
wait_for_file "$LEN_TOK_DIR/vocab.json" "Length-MAX HF vocab.json"
run_model "$LEN_TOK_DIR" "$LEN_MODEL_LOG" "Length-MAX"

# 4) extract curves + overlay
echo "[pipeline] extract loss curves" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$LEN_MODEL_LOG" --out_csv "$LEN_CURVE" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/extract_loss_curve.py --log "$BPE_MODEL_LOG" --out_csv "$BPE_CURVE" | tee -a "$PIPE_LOG"

echo "[pipeline] plot overlays" | tee -a "$PIPE_LOG"
python3 /home/arxiv_code/tokenizers_rust/plot_loss_overlay.py \
  --csv "$LEN_CURVE" --label "Length-MAX (full)" \
  --csv "$BPE_CURVE" --label "BPE (full)" \
  --out_png "$LOSS_OVERLAY_PNG" \
  --out_pdf "$LOSS_OVERLAY_PDF" \
  --title "Loss vs step (full corpus, Llama-style 58.5M, DDP 2x5060Ti, bf16, seq_len=512, bs=64, pack=3000)"

python3 /home/arxiv_code/tokenizers_rust/plot_loss_overlay.py \
  --y_col bpc \
  --csv "$LEN_CURVE" --label "Length-MAX (full)" \
  --csv "$BPE_CURVE" --label "BPE (full)" \
  --out_png "$BPC_OVERLAY_PNG" \
  --out_pdf "$BPC_OVERLAY_PDF" \
  --title "Bits/char proxy vs step (full corpus, bpc≈loss*TPC/ln2)"

echo "[pipeline] DONE $(date -Is)" | tee -a "$PIPE_LOG"


