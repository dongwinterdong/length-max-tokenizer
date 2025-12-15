### length_tokenizer（Rust + HuggingFace + Python wheel）

这是一个 Rust 实现的 tokenizer（支持训练与推理），并提供两种分发形态：

- **HuggingFace remote-code tokenizer**：在 `hf_tokenizer_out/` 目录，可直接上传到 HuggingFace Hub（加载需 `trust_remote_code=True`）
- **Python wheel（高性能）**：使用 PyO3+maturin 构建扩展模块 `length_tokenizer_rs`，供 remote-code tokenizer 优先调用（速度接近原生 Rust）

### 1) 构建 Python wheel（PyO3 / maturin）

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

### 2) HuggingFace 侧使用高性能扩展（可选）

在 HF 的 `tokenization_length_tokenizer.py` 中会优先尝试：

```python
from length_tokenizer_rs import DpTokenizer
```

如果你安装了 wheel，则 `_tokenize()` 会调用 Rust 扩展来做 **Trie+DP 最少 token** 分词；
如果未安装，会自动回退到纯 Python 实现（更慢，但可用）。

你可以设置环境变量禁用 Rust 扩展（用于对齐/排障）：

```bash
export LENGTH_TOKENIZER_DISABLE_RUST=1
```

### 3) 发布到 PyPI（让用户 `pip install length-tokenizer-rs`）

#### 推荐方式：GitHub Actions 自动构建并发布

仓库已包含工作流：`tokenizers_rust/.github/workflows/publish_pypi.yml`

你需要做：
- **在 PyPI 注册账号**，并创建一个 API Token（Account settings → API tokens）
- **在 GitHub 仓库里设置 secret**：`PYPI_API_TOKEN`
- **更新版本号**（建议 `Cargo.toml` 与 `pyproject.toml` 同步），提交并打 tag：

```bash
git tag v0.1.1
git push --tags
```

工作流会在 Linux/macOS/Windows 上构建 wheels 并上传到 PyPI。

#### 备选方式：本机直接发布（不推荐作为正式发布）

本机直接 `maturin publish` 会受你本地 glibc / 平台影响，可能生成兼容性较窄的 wheel。
正式发布建议用 CI（manylinux2014）来构建。


