### Length‑MAX Tokenizer (Rust + PyO3)

This repository provides a **Length‑MAX vocabulary trainer** and a **Rust DP tokenizer** (minimum‑token segmentation / low‑TPC), exported as a HuggingFace‑compatible local tokenizer directory.

Below is the **single recommended fast path**: **train vocab → train a modern Llama‑style LM → run inference + decode**.

---

## Fast path (end‑to‑end)

### 0) Install

```bash
python3 -m pip install -U length-tokenizer-rs torch transformers
```

### 1) Train a vocab (export an HF tokenizer directory)

Input: `corpus.txt` (one sample per line). Output: `./tokenizer_out/` (contains `vocab.json` and `tokenization_length_tokenizer.py`).

```python
from length_tokenizer_rs import train_to_hf

train_to_hf(
    corpus_file="corpus.txt",
    out_dir="./tokenizer_out",
    aim_token_num=32000,
    num_merges=40000,
    n_min=2,
    n_max=9,
    max_token_chars=48,
    num_workers=64,
    multi_process=True,
    use_heap=False,
)
```

### 2) Train a modern Llama‑style model (RoPE + RMSNorm + SwiGLU)

This repo includes `validate_modern_arch_llama.py`, which trains `transformers.LlamaForCausalLM` from scratch using your exported tokenizer dir.

Example (2×GPU DDP, bf16, packed training, saves best‑bpc checkpoint):

```bash
torchrun --standalone --nproc_per_node 2 validate_modern_arch_llama.py \
  --tokenizer_dir ./tokenizer_out \
  --corpus_file corpus.txt \
  --eval_corpus_file corpus.txt \
  --max_lines 0 \
  --eval_max_lines 0 \
  --device cuda \
  --precision bf16 \
  --seq_len 512 \
  --pack_chars 4000 \
  --pack_mode contiguous \
  --force_eos \
  --batch_size 64 \
  --steps 10000 \
  --hidden_size 512 \
  --num_layers 8 \
  --num_heads 8 \
  --num_kv_heads 8 \
  --intermediate_size 1408 \
  --lr 3e-4 \
  --lr_schedule cosine \
  --warmup_steps 200 \
  --min_lr 3e-5 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --grad_clip 1.0 \
  --eval_every 200 \
  --eval_batches 8 \
  --eval_batch_size 32 \
  --eval_seed 12345 \
  --save_dir ./model_out \
  --save_best_metric bpc \
  --save_best_on eval \
  --save_best_min_delta 0.0002 \
  --gen_from_best \
  --gen_prompt "= Valkyria Chronicles III =" \
  --max_new_tokens 160 \
  --gen_temperature 0.7 \
  --gen_top_p 0.9 \
  --gen_repetition_penalty 1.12 \
  --gen_no_repeat_ngram_size 3
```

The script prints a sample generation at the end and writes the best checkpoint to:

- `./model_out/best_bpc/`

### 3) Inference + decode (load best checkpoint)

```python
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

tok = AutoTokenizer.from_pretrained("./tokenizer_out", trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained("./model_out/best_bpc").to("cuda").eval()

prompt = "= Valkyria Chronicles III ="
inputs = tok(prompt, return_tensors="pt").to("cuda")
out = model.generate(
    **inputs,
    max_new_tokens=160,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.12,
    no_repeat_ngram_size=3,
)
print(tok.decode(out[0], skip_special_tokens=True))
```
