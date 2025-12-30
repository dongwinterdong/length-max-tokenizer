### Length-MAX Tokenizer

> 说明（中文补充）：本仓库同时提供
> - Rust 训练/统计实现（`length_tokenizer`）
> - Rust 推理分词（DP 最少 token / 最低 TPC）：Python 扩展 `length_tokenizer_rs.DpTokenizer`
> - HuggingFace remote code 导出（`train_to_hf*` 生成 `tokenizer_out/` 目录）

#### 1) Install

```bash
pip install length-tokenizer-rs
```

If your corpus is parquet (or you want streaming reads via pyarrow):

```bash
pip install pyarrow
```

#### 1.1 GitHub Actions / CI install

- **Option A (recommended)**: publish via GitHub Actions (this repo) and install from PyPI.
  - In GitHub repo settings → **Secrets and variables** → **Actions**, add:
    - `PYPI_API_TOKEN`: your PyPI token (project-scoped recommended).
  - Bump version in `pyproject.toml` / `Cargo.toml`, then create and push a tag like `v0.1.10`.
  - The workflow `/.github/workflows/publish_pypi.yml` will build wheels (Linux/macOS/Windows) + sdist and publish.
  - Then in any CI (or locally):

```bash
pip install length-tokenizer-rs==0.1.10
```

- **Option B**: build-from-source install inside GitHub Actions (no PyPI needed).
  - This compiles the Rust extension during CI.

```yaml
- uses: actions/checkout@v4
- uses: actions/setup-python@v5
  with:
    python-version: "3.10"
- uses: PyO3/maturin-action@v1
  with:
    command: develop
    args: --release
- run: python -c "import length_tokenizer_rs; print(length_tokenizer_rs.__version__)"
```

#### 2) Train a vocab from your corpus (export a local tokenizer directory)

After training you will get a directory (e.g. `./tokenizer_out/`) containing:
`vocab.json` / `tokenizer_config.json` / `special_tokens_map.json` / `tokenization_length_tokenizer.py` / `README.md`

##### 2.1 Text corpus (one sentence per line)

```python
from length_tokenizer_rs import train_to_hf

train_to_hf(
    corpus_file="corpus.txt",   # one sentence per line
    out_dir="./tokenizer_out",
    num_merges=50000,
    aim_token_num=20000,
    n_max=6,
    num_workers=8,
    multi_process=False,
    use_heap=False,  # 默认关闭：更省内存（推荐大语料/大 n_max）
)
```

##### 2.2 Parquet corpus (streaming read via pyarrow)

```python
from length_tokenizer_rs import train_to_hf_parquet

train_to_hf_parquet(
    parquet_path="/path/to/parquet_dir_or_file",
    out_dir="./tokenizer_out",
    text_column="text",
    max_docs=0,
    batch_size=8192,
    recursive=True,
    num_merges=50000,
    aim_token_num=20000,
    n_max=6,
    num_workers=8,
    multi_process=False,
    use_heap=False,  # 默认关闭：更省内存
    chunk_size=4096,
)
```

##### 2.3 推荐最高效设置（大语料 + n_max=9 + 32k vocab）

当你要对齐主流 Llama 系列（`vocab_size≈32k`）并且 n-gram 上限较大（如 `n_max=9`）时，**内存峰值通常是最大瓶颈**。
本实现新增了 `use_heap` 开关（默认关闭），用于避免候选堆复制一份 n-gram key 导致的额外内存。

- **推荐参数**（经验值，适用于 2×GPU 训练前的 vocab 训练阶段）：
  - `aim_token_num=32000`
  - `num_merges=40000`（上限，不等于最终词表大小）
  - `n_max=9`
  - `num_workers=64`（按 CPU 资源调整）
  - `use_heap=False`（默认，强烈建议保持）
  - `multi_process=True`（**推荐大语料**：更稳定，也便于利用增量 apply 优化）

- **默认高效行为（无需额外设置）**：当 `multi_process=True` 时，本实现会默认开启一组“高吞吐/低峰值”的策略：
  - 默认启用 **增量模式**（不再每步全量重算 stats）
  - 默认将 diff 临时文件优先写到 **`/dev/shm`**（Linux 内存盘；若不存在则回退到系统 temp 目录）
  - 默认给每个 worker 设置一个“不过量”的线程数（避免 64 个 worker 各自开满 128 线程导致抖动）
  - 默认使用较大的 `MP_BUCKET_BATCH` 以提升主进程合并桶文件的并行度

- **仍可覆盖**（仅在你需要 debug/保守模式时）：
  - `MP_FULL_RECOMPUTE=1` 或 `MP_NO_INCREMENTAL=1`：强制每步全量重算（更慢）
  - `WORKER_THREADS=1`：限制 worker 内部线程（更稳、但可能更慢）
  - `MP_BUCKET_BATCH=64`：降低主进程并行读桶批大小（降低峰值内存）

- **日志重定向（很重要）**：训练日志默认写到 **stderr**，如果你用 `tee` 记日志，需要把 stderr 合并到 stdout：

```bash
... 2>&1 | tee run.log
```

- **CLI 示例**（适合快速 bench / 复现实验）：

```bash
cd tokenizers_rust

cargo run --release --bin length_tokenizer -- \
  --corpus /path/to/train.txt \
  --corpus-format txt \
  --output token_table_32k.json \
  --num-merges 40000 \
  --aim-token-num 32000 \
  --n-max 9 \
  --num-workers 64 \
  2>&1 | tee run_32k_n9.log
```

如你确实有充足内存并希望加速“找 best n-gram”，可以显式开启 heap：

```bash
--use-heap
```

#### 3) Tokenize your corpus with the new vocab → write ids (data prep)

The examples below write `ids.txt`: one sample per line, space-separated token ids (high throughput: Rust `DpTokenizer.encode_batch()`).

##### 3.1 Text corpus (one sentence per line) → ids.txt

```python
import json
from pathlib import Path

from length_tokenizer_rs import DpTokenizer

TOKENIZER_DIR = Path("./tokenizer_out")
VOCAB = TOKENIZER_DIR / "vocab.json"
dp = DpTokenizer(str(VOCAB), "<unk>")

# Optional: add BOS/EOS (adjust to your training pipeline)
vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
bos = vocab.get("<s>")
eos = vocab.get("</s>")

IN_TXT = Path("corpus.txt")
OUT_IDS = Path("corpus.ids.txt")

BATCH = 256
buf = []
with IN_TXT.open("r", encoding="utf-8", errors="ignore") as r, OUT_IDS.open("w", encoding="utf-8") as w:
    for line in r:
        s = line.strip()
        if not s:
            continue
        buf.append(s)
        if len(buf) >= BATCH:
            for ids in dp.encode_batch(buf):
                if bos is not None:
                    w.write(str(int(bos)) + " ")
                w.write(" ".join(str(int(x)) for x in ids))
                if eos is not None:
                    w.write(" " + str(int(eos)))
                w.write("\n")
            buf.clear()
    if buf:
        for ids in dp.encode_batch(buf):
            if bos is not None:
                w.write(str(int(bos)) + " ")
            w.write(" ".join(str(int(x)) for x in ids))
            if eos is not None:
                w.write(" " + str(int(eos)))
            w.write("\n")
```

##### 3.2 Parquet corpus (streaming read) → ids.txt

```python
import json
from pathlib import Path

import pyarrow.dataset as ds
from length_tokenizer_rs import DpTokenizer

PARQUET = "/path/to/parquet_dir_or_file"
TEXT_COL = "text"

TOKENIZER_DIR = Path("./tokenizer_out")
VOCAB = TOKENIZER_DIR / "vocab.json"
dp = DpTokenizer(str(VOCAB), "<unk>")

vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
bos = vocab.get("<s>")
eos = vocab.get("</s>")

OUT_IDS = Path("parquet.ids.txt")
BATCH = 256
buf = []

dataset = ds.dataset(PARQUET, format="parquet")
scanner = dataset.scanner(columns=[TEXT_COL], batch_size=8192, use_threads=True)

with OUT_IDS.open("w", encoding="utf-8") as w:
    for batch in scanner.to_batches():
        col = batch.column(0)
        for s in col.to_pylist():
            if not s or not str(s).strip():
                continue
            buf.append(str(s))
            if len(buf) >= BATCH:
                for ids in dp.encode_batch(buf):
                    if bos is not None:
                        w.write(str(int(bos)) + " ")
                    w.write(" ".join(str(int(x)) for x in ids))
                    if eos is not None:
                        w.write(" " + str(int(eos)))
                    w.write("\n")
                buf.clear()
    if buf:
        for ids in dp.encode_batch(buf):
            if bos is not None:
                w.write(str(int(bos)) + " ")
            w.write(" ".join(str(int(x)) for x in ids))
            if eos is not None:
                w.write(" " + str(int(eos)))
            w.write("\n")
```

#### 4) Load the tokenizer for training (local directory)

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("./tokenizer_out", trust_remote_code=True)
assert getattr(tok, "_rust", None) is not None, "Rust extension not active"
```

#### 5) 现代架构验证（Llama-style: RoPE + RMSNorm + SwiGLU）

审稿/对外说明时，“现代架构验证”通常指：在 **非 GPT-2** 的 decoder-only Transformer 上，从零训练（或至少跑通训练环节）并复现同方向的效率收益。

本仓库提供一个最小可复用脚本 `validate_modern_arch_llama.py`，用你导出的 Length-MAX tokenizer 直接训练一个小的 `LlamaForCausalLM`（RoPE + RMSNorm + SwiGLU）若干步，验证流程可跑通：

```bash
pip install -U torch transformers

python validate_modern_arch_llama.py \
  --tokenizer_dir ./tokenizer_out \
  --corpus_file corpus.txt \
  --seq_len 256 \
  --batch_size 8 \
  --steps 100
```

建议用于 rebuttal 的正式实验：保持同语料/同 vocab size/同超参，只替换 tokenizer（BPE vs Length-MAX），并在该现代架构上报告 steps-to-target loss、latency/throughput、以及下游任务指标。

##### 5.1 2×GPU（例如 2×5060Ti 16GB）推荐跑法：torchrun DDP

脚本已支持 DDP。示例（两张卡）：

```bash
torchrun --standalone --nproc_per_node 2 validate_modern_arch_llama.py \
  --tokenizer_dir ./tokenizer_out \
  --corpus_file corpus.txt \
  --max_lines 0 \
  --device cuda \
  --precision bf16 \
  --grad_checkpointing \
  --seq_len 1024 \
  --batch_size 32 \
  --grad_accum 4 \
  --steps 2000 \
  --lr 3e-4 \
  --weight_decay 0.1 \
  --print_every 50 \
  --hidden_size 768 \
  --num_layers 12 \
  --num_heads 12 \
  --num_kv_heads 12 \
  --intermediate_size 2048
```

说明：
- `--batch_size` 是 **global batch size**（DDP 下会按 world size 自动切分到每张卡）。
- 显存不够时，优先开 `--grad_checkpointing`，然后减小 `--seq_len` 或 `--batch_size`，用 `--grad_accum` 把全局 batch 拉回去。

#### 6) Publish to PyPI（推荐用 GitHub Actions）

这个仓库原本就带了 GitHub Actions 发版流程：`tokenizers_rust/.github/workflows/publish_pypi.yml`。

- **GitHub 发版（推荐，跨平台 wheel）**：
  - workflow 触发条件是 **push tag**，例如 `v0.1.7`
  - 会构建 **Linux/macOS/Windows** 的 wheels（py3.10/3.11/3.12），并用 `PYPI_API_TOKEN` 发布到 PyPI

示例（在你自己的 git repo 里执行）：

```bash
git add tokenizers_rust
git commit -m "Release v0.1.7"
git tag v0.1.7
git push origin main --tags
```

- **本地发布（不推荐，通常只会上传当前平台 wheel）**：

注意：`maturin publish` **不需要也不支持** `--release`（默认就是 release 构建）。

```bash
cd tokenizers_rust
maturin publish
```


