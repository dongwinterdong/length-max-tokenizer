# Length‑MAX Tokenizer‑RS 操作说明（最新版本）

本说明面向 **`tokenizers_rust/`**：包含 Length‑MAX 的 Rust 核心、PyO3 Python 扩展（`length-tokenizer-rs`），以及本仓库用于复现实验/评测/画图的一组脚本。

> 约定：本仓库很多脚本默认使用作者机器上的路径（例如 `/home/arxiv_code/datasets/...`）。如果你的目录不同，请用脚本提供的 CLI 参数覆盖，或直接修改脚本头部的默认路径。

---

## 0) 环境准备

- **Python**: >= 3.8  
- **Rust**: stable toolchain（用于从源码构建 Python 扩展）  
- **可选 GPU**: 复现 Llama‑style 训练/评测需要 CUDA + PyTorch

推荐（可选）安装：

```bash
python3 -m pip install -U pip
python3 -m pip install -U torch transformers datasets matplotlib
```

---

## 1) 安装（推荐从 PyPI）

```bash
python3 -m pip install -U length-tokenizer-rs
python3 -c "import length_tokenizer_rs; print(length_tokenizer_rs.__version__)"
```

从源码构建（需要 Rust）：

```bash
cd /path/to/length-max-tokenizer/tokenizers_rust
python3 -m pip install -U maturin
maturin develop -r
python3 -c "import length_tokenizer_rs; print(length_tokenizer_rs.__version__)"
```

---

## 2) 训练 Length‑MAX 词表（WT103 v047 示例）

本仓库提供了可直接复现 v047 的训练脚本（开启受控的跨词标点混合 token）：

```bash
cd tokenizers_rust
bash run_train_wt103_lenmax_punctnorm_nostage_lowtpc_crossmix_vocab_v047.sh
```

成功后会生成一个 HuggingFace tokenizer 目录（里面最重要的是 `vocab.json`）：

- `tokenizer_out_wikitext103_32k_n9_full_maxchars48_punctnorm_nostage_lowtpc_crossmix_v047/`

---

## 3) 一键对比：Length‑MAX(v047) vs SuperBPE（v048 pipeline）

模板脚本：`run_full_wt103_lenmax_vs_superbpe_template.sh`  
它会：

- 等待 `LEN_TOK_DIR/vocab.json`
- 先统计 full‑corpus token totals（得到 DP TPC）
- （可选）根据 `MAX_TPC_GAP` gate 决定是否继续训练 10k steps 模型
- 输出 `summary_*.csv` 与 overlay 图（图文件默认被 `.gitignore` 忽略）

示例（以 v047 tokenizer 目录为输入）：

```bash
cd tokenizers_rust
LEN_TOK_DIR="/home/arxiv_code/tokenizers_rust/tokenizer_out_wikitext103_32k_n9_full_maxchars48_punctnorm_nostage_lowtpc_crossmix_v047" \
TAG="lenmax_punctnorm_nostage_crossmix_v047_vs_superbpe_pack4000_evalval_steps10000_v048" \
MAX_TPC_GAP="0.008" \
bash run_full_wt103_lenmax_vs_superbpe_template.sh
```

主要产物（文件名由 `TAG` 决定）：

- `token_counts_${TAG}.txt`（full‑corpus TPC）
- `summary_${TAG}.csv`（TPC + best eval_loss + best bpc 的汇总）
- `run_llama_lenmax_${TAG}.log` / `run_llama_superbpe_${TAG}.log`（训练日志）

---

## 4) WT103/WT2 的额外可比指标（ppl_char / bpb / throughput）

脚本：`eval_extra_metrics_lenmax_vs_superbpe.py`  
默认路径已指向本仓库当前“最佳对比”目录（v047 vs v044），你也可以用参数覆盖：

```bash
cd tokenizers_rust
python3 eval_extra_metrics_lenmax_vs_superbpe.py \
  --device cuda \
  --precision bf16 \
  --seq_len 512 \
  --pack_chars 4000 \
  --pack_mode contiguous \
  --force_eos \
  --batches 32 \
  --batch_size 16 \
  --warmup_batches 8
```

输出会打印一个 CSV 风格表（包含 `model_tok/s` 与 `model_char/s` 这类更公平的 model‑only 吞吐）。

---

## 5) 下游任务 + 长上下文（含适配性指标）

脚本：`eval_downstream_threeway.py`（BPE / SuperBPE / Length‑MAX 三模型对比）

特点：

- MC 任务支持 **长度归一化打分**（`--mc_score_norm mean_char` 推荐）
- 长上下文 needle/kv 检索会同时输出 `needle_present_rate`（信息是否仍在窗口中）

示例（默认会跑一组任务并写 CSV）：

```bash
cd tokenizers_rust
python3 eval_downstream_threeway.py \
  --device cuda \
  --precision bf16 \
  --seq_len 512 \
  --mc_score_norm mean_char \
  --max_examples 1000 \
  --out_csv downstream_threeway_results_v1.csv
```

画图：

```bash
cd tokenizers_rust
python3 plot_downstream_threeway.py --csv downstream_threeway_results_v1.csv --out_dir downstream_plots_v1
```

---

## 6) 论文/回应审稿人的“有利指标”汇总图

脚本：`plot_lenmax_favorable_metrics.py`

### 6.1 直接从本机实验产物生成

```bash
cd tokenizers_rust
python3 plot_lenmax_favorable_metrics.py --out_dir lenmax_favorable_plots
```

### 6.2 不跑训练也能复现关键图（使用打包的 plot data）

仓库内提供了一个小的 `lenmax_plot_data.tar.gz`（包含用于画图的若干 CSV）。

```bash
cd tokenizers_rust
tar -xzf lenmax_plot_data.tar.gz
python3 plot_lenmax_favorable_metrics.py \
  --len_vs_super_csv lenmax_plot_data/summary_lenmax_punctnorm_nostage_crossmix_v047_vs_superbpe_pack4000_evalval_steps10000_v048.csv \
  --bpe_baseline_csv lenmax_plot_data/summary_lenmax_punctwhitelist_stage30000_maxw3_pack4000_evalval_steps10000_v042.csv \
  --downstream_csv lenmax_plot_data/downstream_threeway_results_v1.csv \
  --longctx_csv lenmax_plot_data/longctx_fair_results_v3.csv \
  --dist_csv_in lenmax_plot_data/lenmax_token_distribution_wt103_valid.csv \
  --out_dir lenmax_favorable_plots
```

---

## 7) 发布到 PyPI（GitHub Actions）

workflow：`.github/workflows/publish_pypi.yml`  
触发方式：push tag（tag 需与 `pyproject.toml` / `Cargo.toml` 版本一致）

发布前：

- bump `tokenizers_rust/pyproject.toml` 的 `version`
- bump `tokenizers_rust/Cargo.toml` 的 `version`
- GitHub repo 配置 secret：`PYPI_API_TOKEN`

示例（以当前版本为例）：

```bash
cd tokenizers_rust
git add -A
git commit -m "Release v0.1.10"
git tag v0.1.10
git push origin main --tags
```


