### length_tokenizer（Rust + Python wheel）

这是一个 Rust 实现的 tokenizer（支持训练与推理），对外以一个 PyPI 包分发：`length-tokenizer-rs`。

推荐使用方式（从头训练词表）：
- `train_to_hf*` 从语料训练并导出一个**本地 tokenizer 输出目录**（包含 `vocab.json` / 配置 / remote code 文件）
- 训练模型时直接用本地路径 `AutoTokenizer.from_pretrained("/path/to/tokenizer_out", trust_remote_code=True)` 加载

### 0) 安装（推荐）

```bash
pip install length-tokenizer-rs
```

### 0.5) 从源码构建 wheel（开发者）

要求：你的机器上有 Python 与 Rust 工具链。

```bash
pip install maturin
cd tokenizers_rust
maturin build --release
```

开发安装（直接 `import length_tokenizer_rs`）：

```bash
pip install maturin
cd tokenizers_rust
maturin develop --release
python3 -c \"from length_tokenizer_rs import DpTokenizer; print(DpTokenizer)\"
```

### 1) 从头训练词表并导出本地 tokenizer 目录

安装后你可以直接在 Python 里从语料训练，并导出一套可被 Transformers 直接加载的本地目录：

```python
from length_tokenizer_rs import train_to_hf

train_to_hf(
    corpus_file="corpus.txt",   # 纯文本：每行一句
    out_dir="./tokenizer_out",
    num_merges=50000,
    aim_token_num=20000,
    n_max=6,
    num_workers=8,
    multi_process=False,
)
```

导出目录会包含：`vocab.json`、`tokenizer_config.json`、`special_tokens_map.json`、`tokenization_length_tokenizer.py`、`README.md`。

#### 流式训练（iterator / generator，不把整份语料常驻内存）

适合 HF datasets streaming、自己写的生成器等：

```python
from length_tokenizer_rs import train_to_hf_iter

def stream_texts():
    yield "hello world"
    yield "foo bar"

train_to_hf_iter(
    stream_texts(),
    out_dir="./tokenizer_out",
    num_merges=50000,
    aim_token_num=20000,
    n_max=6,
    num_workers=8,
    multi_process=False,
    chunk_size=4096,
)
```

#### Parquet 训练（流式读取，依赖 pyarrow）

不把 arrow/parquet 打进 wheel；需要用户自行安装 `pyarrow`：

```bash
pip install pyarrow
```

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
    chunk_size=4096,
)
```

### 2) 用新 tokenizer 对语料 tokenize（训练数据准备）

训练完成后，你会得到 `./tokenizer_out/vocab.json`。下面示例用同一份 `vocab.json` 把语料转成 token ids，并写到磁盘：
- 输出格式：`ids.txt`（每行一条样本，空格分隔的 token id）
- 分词实现：直接走 Rust 扩展 `DpTokenizer.encode_batch()`（高吞吐）

#### 2.1 纯文本语料（每行一句）→ ids.txt

```python
import json
from pathlib import Path

from length_tokenizer_rs import DpTokenizer

TOKENIZER_DIR = Path("./tokenizer_out")
VOCAB = TOKENIZER_DIR / "vocab.json"
dp = DpTokenizer(str(VOCAB), "<unk>")

# 如需加特殊 token（可按你的训练脚本需要调整）
vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
bos = vocab.get("<s>")
eos = vocab.get("</s>")

IN_TXT = Path("corpus.txt")          # 每行一句
OUT_IDS = Path("corpus.ids.txt")     # 每行: space-separated ids

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

#### 2.2 Parquet 语料（流式读取，依赖 pyarrow）→ ids.txt

```bash
pip install pyarrow
```

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

#### 2.3 常见：把 ids 拼接并按 block_size 切块（适合 causal LM）

如果你的训练是 GPT/causal LM 风格，一般会把所有 ids 串起来再切成固定长度 block（减少 padding、吞吐更高）：

```python
from pathlib import Path

IN_IDS = Path("corpus.ids.txt")
OUT_BLOCKS = Path("train.blocks.txt")  # 每行一个 block（空格分隔），示例写文本；你也可以改成二进制/memmap
BLOCK_SIZE = 4096

buf = []
with IN_IDS.open("r", encoding="utf-8") as r, OUT_BLOCKS.open("w", encoding="utf-8") as w:
    for line in r:
        ids = [int(x) for x in line.split()]
        buf.extend(ids)
        while len(buf) >= BLOCK_SIZE:
            block = buf[:BLOCK_SIZE]
            buf = buf[BLOCK_SIZE:]
            w.write(" ".join(map(str, block)))
            w.write("\n")
```

### 3) 训练模型时加载（本地路径）

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("./tokenizer_out", trust_remote_code=True)

# 建议用 batched 调用提高吞吐
batch = tok(["hello world", "foo bar"], padding=True, return_attention_mask=True)
print(batch["input_ids"])
```

### 3) 发布到 PyPI（让用户 `pip install length-tokenizer-rs`）

#### 推荐方式：GitHub Actions 自动构建并发布

仓库已包含工作流：`tokenizers_rust/.github/workflows/publish_pypi.yml`

你需要做：
- **在 PyPI 注册账号**，并创建一个 API Token（Account settings → API tokens）
- **在 GitHub 仓库里设置 secret**：`PYPI_API_TOKEN`
- **更新版本号**（建议 `Cargo.toml` 与 `pyproject.toml` 同步），提交并打 tag：

```bash
git tag v0.1.5
git push --tags
```

工作流会在 Linux/macOS/Windows 上构建 wheels 并上传到 PyPI。

#### 备选方式：本机直接发布（不推荐作为正式发布）

本机直接 `maturin publish` 会受你本地 glibc / 平台影响，可能生成兼容性较窄的 wheel。
正式发布建议用 CI（manylinux2014）来构建。


