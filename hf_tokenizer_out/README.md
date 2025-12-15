### LengthTokenizer（DP 最小 token / 最低 TPC）

这是一个用于 Hugging Face Transformers 的 **自定义 tokenizer**（remote code），分词策略为：
- 先按空白切词，并把每个词变换为：`词 + "Ġ"`（与本仓库 Rust 训练口径一致）
- 在给定 vocab 下，用 Trie + DP 找到 **token 数最少** 的全局最优切分（TPC 最低）

#### 使用方法（Transformers）

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("YOUR_USER/YOUR_REPO", trust_remote_code=True)
print(tok.tokenize("hello world"))
```

> 注意：这是 remote code，需要 `trust_remote_code=True`。

### 高性能（可选）：安装 Rust wheel 以加速分词

本 tokenizer 默认是 Python remote-code。若你希望分词速度接近原生 Rust，可额外安装 PyO3 wheel：

```bash
pip install maturin
# 在源码仓库 tokenizers_rust/ 下构建并安装（或你也可以从你发布的 wheel 安装）
maturin develop --release
```

安装后，`tokenization_length_tokenizer.py` 会自动优先 import `length_tokenizer_rs` 并使用 Rust 的 DP 分词。

如需禁用（对齐/排障）：

```bash
export LENGTH_TOKENIZER_DISABLE_RUST=1
```

### 训练（从指定语料重新训练词汇表）

本仓库也**打包了训练功能**，用于从你自己的语料重新训练并导出一份新的 HF tokenizer 目录。

要求：
- 本机安装 Rust 工具链（`cargo`）

用法：

```bash
python3 train_from_corpus.py \
  --corpus /path/to/new_corpus.txt \
  --out ./hf_tokenizer_new \
  --num-merges 50000 \
  --n-max 6
```

训练完成后，把 `./hf_tokenizer_new/` 上传到 HuggingFace Hub 即可（加载时依然需要 `trust_remote_code=True`）。
