//! PyO3 扩展模块：`length_tokenizer_rs`
//!
//! 目标：
//! - 把“DP 最少 token（最低 TPC）”的应用分词放到 Rust 里跑
//! - HuggingFace 的 `tokenization_length_tokenizer.py` 会优先 import 本模块
//!
//! 构建 wheel（示例）：
//! ```bash
//! # 安装 maturin（用户环境）
//! pip install maturin
//! cd tokenizers_rust
//! maturin build --release
//! # 或开发安装：
//! maturin develop --release
//! ```

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use rayon::prelude::*;

use crate::{LengthTokenizer, TokenTrie};

const END_TOKEN: &str = "Ġ";
const UNK: &str = "<unk>";

#[pyclass]
struct DpTokenizer {
    trie: TokenTrie,
    unk_id: u32,
}

#[pymethods]
impl DpTokenizer {
    /// 从 vocab.json 初始化（token -> id）
    #[new]
    #[pyo3(signature = (vocab_file, unk_token=None))]
    fn new(vocab_file: &str, unk_token: Option<String>) -> PyResult<Self> {
        let unk = unk_token.unwrap_or_else(|| UNK.to_string());
        let p = Path::new(vocab_file);
        let f = File::open(p).map_err(|e| PyValueError::new_err(format!("open vocab failed: {e}")))?;
        let reader = BufReader::new(f);
        let vocab: std::collections::HashMap<String, u32> =
            serde_json::from_reader(reader).map_err(|e| PyValueError::new_err(format!("parse vocab.json failed: {e}")))?;

        let unk_id = *vocab
            .get(&unk)
            .ok_or_else(|| PyValueError::new_err(format!("unk_token {unk:?} not found in vocab")))?;

        // 构建 trie（term_id = token id）
        let mut trie = TokenTrie::new();
        // 注意：HashMap 遍历无序，但这里只依赖 term_id，插入顺序无关
        for (tok, &id) in &vocab {
            trie.insert(tok, id);
        }

        Ok(Self { trie, unk_id })
    }

    /// encode：返回 token id 列表（DP 最少 token，无法匹配时用 unk 兜底）
    fn encode(&self, text: &str) -> Vec<u32> {
        // 复刻 Rust 训练口径：split_whitespace + 每词追加 END_TOKEN
        let chars = LengthTokenizer::normalize_chars(text);
        self.trie.dp_min_ids_allow_unk(&chars, self.unk_id)
    }

    /// 批量 encode：释放 GIL，适合大吞吐
    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        let trie = &self.trie;
        let unk_id = self.unk_id;
        py.allow_threads(|| {
            Ok(texts
                // rayon：并行处理 batch（collect 保持原顺序）
                .into_par_iter()
                .map(|t| {
                    let chars = LengthTokenizer::normalize_chars(&t);
                    trie.dp_min_ids_allow_unk(&chars, unk_id)
                })
                .collect())
        })
    }
}

#[pymodule]
fn length_tokenizer_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("END_TOKEN", END_TOKEN)?;
    m.add_class::<DpTokenizer>()?;
    Ok(())
}


