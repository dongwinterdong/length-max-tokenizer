### Length‑MAX Tokenizer (Rust core + Python extension)

This repo provides:

- **Vocabulary training** (Length‑MAX merges)
- **DP inference tokenizer** (`length_tokenizer_rs.DpTokenizer`, min‑token / low‑TPC segmentation)
- **HuggingFace export** (a local tokenizer directory compatible with `AutoTokenizer(..., trust_remote_code=True)`)

For the full reproduction pipeline (WT103 v047/v048, downstream eval, plots, PyPI release), see `OPERATIONS.md`.

#### Install

```bash
python3 -m pip install -U length-tokenizer-rs
```

If your corpus is Parquet / you want streaming reads:

```bash
python3 -m pip install -U pyarrow
```

#### Train a vocab (export an HF tokenizer directory)

After training, `out_dir/` contains: `vocab.json`, `tokenizer_config.json`, `special_tokens_map.json`, `tokenization_length_tokenizer.py`, etc.

Text corpus (one sample per line):

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
)
```

Parquet corpus (streaming via pyarrow):

```python
from length_tokenizer_rs import train_to_hf_parquet

train_to_hf_parquet(
    parquet_path="/path/to/parquet_dir_or_file",
    out_dir="./tokenizer_out",
    text_column="text",
    aim_token_num=32000,
    num_merges=40000,
    n_min=2,
    n_max=9,
    max_token_chars=48,
    num_workers=64,
    multi_process=True,
)
```

#### Tokenize (DP min‑token) and write ids

```python
from length_tokenizer_rs import DpTokenizer

dp = DpTokenizer("./tokenizer_out/vocab.json", "<unk>")

ids = dp.encode("hello world")
batch_ids = list(dp.encode_batch(["hello", "world"]))
```

#### Use with 🤗 Transformers (HF local dir)

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("./tokenizer_out", trust_remote_code=True)
```

#### Modern Llama‑style training sanity check

```bash
python3 validate_modern_arch_llama.py \
  --tokenizer_dir ./tokenizer_out \
  --corpus_file corpus.txt \
  --seq_len 256 \
  --batch_size 8 \
  --steps 100
```

#### Publish to PyPI (GitHub Actions)

Workflow: `.github/workflows/publish_pypi.yml` (tag‑triggered).

```bash
cd tokenizers_rust
git add -A
git commit -m "Release v0.1.10"
git tag v0.1.10
git push origin main --tags
```
