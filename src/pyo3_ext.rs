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
use pyo3::types::PyDict;
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use ahash::RandomState;
use hashbrown::HashMap;
use crate::{hf_export, LengthTokenizer, TokenTrie, TokenizerConfig};

const END_TOKEN: &str = "Ġ";
const UNK: &str = "<unk>";

fn forbid_end_inner_enabled() -> bool {
    // If set, we forbid tokens that contain END_TOKEN ('Ġ') anywhere except the last char.
    // This effectively prevents cross-word tokens in our whitespace+END normalization scheme.
    env::var("LENGTH_TOKENIZER_FORBID_END_INNER").is_ok()
}

fn cross_word_whole_only_enabled() -> bool {
    // If set, we allow cross-word tokens only if they cover whole words:
    // - token ends with END_TOKEN
    // - token starts at a word start position (enforced in DP by requiring word-start)
    env::var("LENGTH_TOKENIZER_CROSS_WORD_WHOLE_ONLY").is_ok()
}

fn forbid_punct_tokens_enabled() -> bool {
    // If set, we forbid using any *multi-character* token that contains hard punctuation.
    // (Single-character punctuation tokens are still allowed, otherwise punctuation would become <unk>.)
    env::var("LENGTH_TOKENIZER_FORBID_PUNCT_TOKENS").is_ok()
}

#[inline]
fn has_end_before_last(tok: &str) -> bool {
    // True if END_TOKEN appears and is NOT the last character.
    // Use char iteration (UTF-8 safe): 'Ġ' is multi-byte.
    let mut it = tok.chars().peekable();
    while let Some(ch) = it.next() {
        if ch == 'Ġ' && it.peek().is_some() {
            return true;
        }
    }
    false
}

#[inline]
fn has_hard_punct(tok: &str) -> bool {
    for ch in tok.chars() {
        if ch == 'Ġ' {
            continue;
        }
        if !ch.is_alphanumeric() {
            return true;
        }
    }
    false
}

#[pyclass]
struct DpTokenizer {
    trie: TokenTrie,
    unk_id: u32,
    require_word_start: Vec<u8>,
    cross_word_whole_only: bool,
}

fn build_counts_chunk(
    pool: &rayon::ThreadPool,
    texts: &[String],
) -> HashMap<Vec<String>, u32, RandomState> {
    pool.install(|| {
        texts
            .par_iter()
            .map(|s| {
                let mut local: HashMap<Vec<String>, u32, RandomState> = HashMap::with_hasher(RandomState::new());
                let encoded = LengthTokenizer::encode_sentence_str(s);
                *local.entry(encoded).or_insert(0) += 1;
                local
            })
            .reduce(
                || HashMap::with_hasher(RandomState::new()),
                |mut acc, m| {
                    for (k, v) in m {
                        *acc.entry(k).or_insert(0) += v;
                    }
                    acc
                },
            )
    })
}

fn merge_counts_into(
    dst: &mut HashMap<Vec<String>, u32, RandomState>,
    src: HashMap<Vec<String>, u32, RandomState>,
) {
    for (k, v) in src {
        let e = dst.entry(k).or_insert(0);
        *e = e.saturating_add(v);
    }
}

#[pyfunction]
#[pyo3(signature = (
    corpus_file,
    out_dir,
    num_merges=50000,
    aim_token_num=20000,
    n_max=6,
    max_token_chars=0,
    num_workers=0,
    multi_process=false,
    use_heap=false,
    n_min=2,
    forbid_punct_word_mix=false,
    forbid_punct_tokens=false,
    allow_space_punct_tokens=false,
    allow_abbrev_tokens=false,
    allow_hyphen_tokens=false,
    allow_word_final_punct_tokens=false,
    allow_apostrophe_tokens=false,
    allow_cross_word_punct_word_mix_tokens=false,
    cross_word_start_vocab=0,
    max_token_words=0,
    forbid_incomplete_cross_word=false
))]
fn train_to_hf(
    corpus_file: &str,
    out_dir: &str,
    num_merges: usize,
    aim_token_num: usize,
    n_max: usize,
    max_token_chars: usize,
    num_workers: usize,
    multi_process: bool,
    use_heap: bool,
    n_min: usize,
    forbid_punct_word_mix: bool,
    forbid_punct_tokens: bool,
    allow_space_punct_tokens: bool,
    allow_abbrev_tokens: bool,
    allow_hyphen_tokens: bool,
    allow_word_final_punct_tokens: bool,
    allow_apostrophe_tokens: bool,
    allow_cross_word_punct_word_mix_tokens: bool,
    cross_word_start_vocab: usize,
    max_token_words: usize,
    forbid_incomplete_cross_word: bool,
) -> PyResult<()> {
    if n_min < 2 {
        return Err(PyValueError::new_err("n_min must be >= 2"));
    }
    if n_max < n_min {
        return Err(PyValueError::new_err("n_max must be >= n_min"));
    }

    // multi-process 在 Python 进程内需要额外的 worker 启动方式；通过 MP_PY_WORKER 切换。
    let prev_mp = env::var("MP_PY_WORKER").ok();
    if multi_process {
        env::set_var("MP_PY_WORKER", "1");
    } else {
        env::remove_var("MP_PY_WORKER");
    }

    let res = (|| -> anyhow::Result<()> {
        let corpus = LengthTokenizer::load_corpus(corpus_file)?;
        if corpus.is_empty() {
            anyhow::bail!("corpus is empty: {corpus_file}");
        }
        let n_values: Vec<usize> = (n_min..=n_max).collect();
        let cfg = TokenizerConfig {
            num_merges,
            n_values,
            aim_token_num,
            max_token_chars,
            forbid_punct_word_mix,
            forbid_punct_tokens,
            allow_space_punct_tokens,
            allow_abbrev_tokens,
            allow_hyphen_tokens,
            allow_word_final_punct_tokens,
            allow_apostrophe_tokens,
            allow_cross_word_punct_word_mix_tokens,
            cross_word_start_vocab,
            max_token_words,
            forbid_incomplete_cross_word,
            recompute_each_step: false,
            use_heap,
            num_workers,
            use_multiprocess: multi_process,
        };
        let tk = LengthTokenizer::new(&corpus, cfg)?;
        hf_export::export_from_trained(&tk, Path::new(out_dir))?;
        Ok(())
    })();

    // 恢复环境变量，避免影响用户后续逻辑
    match prev_mp {
        Some(v) => env::set_var("MP_PY_WORKER", v),
        None => env::remove_var("MP_PY_WORKER"),
    }

    res.map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
    texts,
    out_dir,
    num_merges=50000,
    aim_token_num=20000,
    n_max=6,
    max_token_chars=0,
    num_workers=0,
    multi_process=false,
    use_heap=false,
    max_docs=0,
    chunk_size=4096,
    n_min=2,
    forbid_punct_word_mix=false,
    forbid_punct_tokens=false,
    allow_space_punct_tokens=false,
    allow_abbrev_tokens=false,
    allow_hyphen_tokens=false,
    allow_word_final_punct_tokens=false,
    allow_apostrophe_tokens=false,
    allow_cross_word_punct_word_mix_tokens=false,
    cross_word_start_vocab=0,
    max_token_words=0,
    forbid_incomplete_cross_word=false
))]
fn train_to_hf_iter(
    py: Python<'_>,
    texts: &Bound<'_, PyAny>,
    out_dir: &str,
    num_merges: usize,
    aim_token_num: usize,
    n_max: usize,
    max_token_chars: usize,
    num_workers: usize,
    multi_process: bool,
    use_heap: bool,
    max_docs: usize,
    chunk_size: usize,
    n_min: usize,
    forbid_punct_word_mix: bool,
    forbid_punct_tokens: bool,
    allow_space_punct_tokens: bool,
    allow_abbrev_tokens: bool,
    allow_hyphen_tokens: bool,
    allow_word_final_punct_tokens: bool,
    allow_apostrophe_tokens: bool,
    allow_cross_word_punct_word_mix_tokens: bool,
    cross_word_start_vocab: usize,
    max_token_words: usize,
    forbid_incomplete_cross_word: bool,
) -> PyResult<()> {
    if n_min < 2 {
        return Err(PyValueError::new_err("n_min must be >= 2"));
    }
    if n_max < n_min {
        return Err(PyValueError::new_err("n_max must be >= n_min"));
    }
    if chunk_size == 0 {
        return Err(PyValueError::new_err("chunk_size must be >= 1"));
    }

    let workers = if num_workers == 0 { num_cpus::get().max(1) } else { num_workers.max(1) };
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| PyValueError::new_err(format!("build rayon pool failed: {e}")))?;

    // multi-process 在 Python 进程内需要额外的 worker 启动方式；通过 MP_PY_WORKER 切换。
    let prev_mp = env::var("MP_PY_WORKER").ok();
    if multi_process {
        env::set_var("MP_PY_WORKER", "1");
    } else {
        env::remove_var("MP_PY_WORKER");
    }

    let mut counts: HashMap<Vec<String>, u32, RandomState> = HashMap::with_hasher(RandomState::new());
    let mut corpus_chars: u64 = 0;
    let mut seen: usize = 0;
    let mut buf: Vec<String> = Vec::with_capacity(chunk_size.min(1 << 20));

    let res = (|| -> anyhow::Result<()> {
        for item in texts.iter()? {
            let s: String = item?.extract()?;
            if s.trim().is_empty() {
                continue;
            }
            corpus_chars = corpus_chars.saturating_add(s.chars().count() as u64);
            buf.push(s);
            seen += 1;
            if max_docs > 0 && seen >= max_docs {
                break;
            }
            if buf.len() >= chunk_size {
                let chunk = std::mem::take(&mut buf);
                let local = py.allow_threads(|| build_counts_chunk(&pool, &chunk));
                merge_counts_into(&mut counts, local);
            }
        }

        if !buf.is_empty() {
            let chunk = std::mem::take(&mut buf);
            let local = py.allow_threads(|| build_counts_chunk(&pool, &chunk));
            merge_counts_into(&mut counts, local);
        }

        if counts.is_empty() {
            anyhow::bail!("no training texts (empty iterator?)");
        }

        let n_values: Vec<usize> = (n_min..=n_max).collect();
        let cfg = TokenizerConfig {
            num_merges,
            n_values,
            aim_token_num,
            max_token_chars,
            forbid_punct_word_mix,
            forbid_punct_tokens,
            allow_space_punct_tokens,
            allow_abbrev_tokens,
            allow_hyphen_tokens,
            allow_word_final_punct_tokens,
            allow_apostrophe_tokens,
            allow_cross_word_punct_word_mix_tokens,
            cross_word_start_vocab,
            max_token_words,
            forbid_incomplete_cross_word,
            recompute_each_step: false,
            use_heap,
            num_workers,
            use_multiprocess: multi_process,
        };

        let tk = LengthTokenizer::new_from_counts(counts, corpus_chars, cfg)?;
        hf_export::export_from_trained(&tk, Path::new(out_dir))?;
        Ok(())
    })();

    // 恢复环境变量，避免影响用户后续逻辑
    match prev_mp {
        Some(v) => env::set_var("MP_PY_WORKER", v),
        None => env::remove_var("MP_PY_WORKER"),
    }

    res.map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
    parquet_path,
    out_dir,
    text_column="text",
    max_docs=0,
    batch_size=8192,
    recursive=true,
    num_merges=50000,
    aim_token_num=20000,
    n_max=6,
    max_token_chars=0,
    num_workers=0,
    multi_process=false,
    use_heap=false,
    chunk_size=4096,
    n_min=2,
    forbid_punct_word_mix=false,
    forbid_punct_tokens=false,
    allow_space_punct_tokens=false,
    allow_abbrev_tokens=false,
    allow_hyphen_tokens=false,
    allow_word_final_punct_tokens=false,
    allow_apostrophe_tokens=false,
    allow_cross_word_punct_word_mix_tokens=false,
    cross_word_start_vocab=0,
    max_token_words=0,
    forbid_incomplete_cross_word=false
))]
fn train_to_hf_parquet(
    py: Python<'_>,
    parquet_path: &str,
    out_dir: &str,
    text_column: &str,
    max_docs: usize,
    batch_size: usize,
    recursive: bool,
    num_merges: usize,
    aim_token_num: usize,
    n_max: usize,
    max_token_chars: usize,
    num_workers: usize,
    multi_process: bool,
    use_heap: bool,
    chunk_size: usize,
    n_min: usize,
    forbid_punct_word_mix: bool,
    forbid_punct_tokens: bool,
    allow_space_punct_tokens: bool,
    allow_abbrev_tokens: bool,
    allow_hyphen_tokens: bool,
    allow_word_final_punct_tokens: bool,
    allow_apostrophe_tokens: bool,
    allow_cross_word_punct_word_mix_tokens: bool,
    cross_word_start_vocab: usize,
    max_token_words: usize,
    forbid_incomplete_cross_word: bool,
) -> PyResult<()> {
    // 依赖 Python 侧 pyarrow 来读取 parquet（不把 arrow/parquet 打进 wheel）
    let ds = py
        .import_bound("pyarrow.dataset")
        .map_err(|_| PyValueError::new_err("pyarrow is required for parquet: pip install pyarrow"))?;

    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("format", "parquet")?;

    // recursive=false 时只取目录下一层 parquet 文件（避免深层扫描）
    let src_obj = if !recursive && Path::new(parquet_path).is_dir() {
        let mut files: Vec<String> = Vec::new();
        for ent in std::fs::read_dir(parquet_path)
            .map_err(|e| PyValueError::new_err(format!("read_dir failed: {e}")))? {
            let ent = ent.map_err(|e| PyValueError::new_err(format!("read_dir entry failed: {e}")))?;
            let p = ent.path();
            if p.is_file()
                && p.extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s.eq_ignore_ascii_case("parquet"))
                    .unwrap_or(false)
            {
                files.push(p.to_string_lossy().to_string());
            }
        }
        if files.is_empty() {
            return Err(PyValueError::new_err("no .parquet files found (recursive=false)"));
        }
        files.into_py(py)
    } else {
        parquet_path.into_py(py)
    };

    let dataset = ds.getattr("dataset")?.call((src_obj,), Some(&kwargs))?;

    let scan_kwargs = PyDict::new_bound(py);
    scan_kwargs.set_item("columns", vec![text_column])?;
    scan_kwargs.set_item("batch_size", batch_size.max(1))?;
    scan_kwargs.set_item("use_threads", true)?;
    let scanner = dataset.call_method("scanner", (), Some(&scan_kwargs))?;
    let batches = scanner.call_method0("to_batches")?;

    // 把 parquet batch 展开成 string iterator，再走 train_to_hf_iter 的流式路径
    // 这里为了减少 Python/Rust 往返，把每个 batch 的列一次性 to_pylist() 再抽取为 Vec<Option<String>>。
    let mut buf: Vec<String> = Vec::with_capacity(chunk_size.max(1));
    let mut seen: usize = 0;

    // 先准备训练参数配置（复用 train_to_hf_iter 的实现思路，但 parquet 这边我们直接喂 Rust Strings）
    if n_min < 2 {
        return Err(PyValueError::new_err("n_min must be >= 2"));
    }
    if n_max < n_min {
        return Err(PyValueError::new_err("n_max must be >= n_min"));
    }
    if chunk_size == 0 {
        return Err(PyValueError::new_err("chunk_size must be >= 1"));
    }

    let workers = if num_workers == 0 { num_cpus::get().max(1) } else { num_workers.max(1) };
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| PyValueError::new_err(format!("build rayon pool failed: {e}")))?;

    let prev_mp = env::var("MP_PY_WORKER").ok();
    if multi_process {
        env::set_var("MP_PY_WORKER", "1");
    } else {
        env::remove_var("MP_PY_WORKER");
    }

    let mut counts: HashMap<Vec<String>, u32, RandomState> = HashMap::with_hasher(RandomState::new());
    let mut corpus_chars: u64 = 0;

    let res = (|| -> anyhow::Result<()> {
        for b in batches.iter()? {
            let b = b?;
            let col = b.call_method1("column", (0,))?;
            let pylist = col.call_method0("to_pylist")?;
            let rows: Vec<Option<String>> = pylist.extract()?;
            for s in rows.into_iter().flatten() {
                if s.trim().is_empty() {
                    continue;
                }
                corpus_chars = corpus_chars.saturating_add(s.chars().count() as u64);
                buf.push(s);
                seen += 1;
                if max_docs > 0 && seen >= max_docs {
                    break;
                }
                if buf.len() >= chunk_size {
                    let chunk = std::mem::take(&mut buf);
                    let local = py.allow_threads(|| build_counts_chunk(&pool, &chunk));
                    merge_counts_into(&mut counts, local);
                }
            }
            if max_docs > 0 && seen >= max_docs {
                break;
            }
        }

        if !buf.is_empty() {
            let chunk = std::mem::take(&mut buf);
            let local = py.allow_threads(|| build_counts_chunk(&pool, &chunk));
            merge_counts_into(&mut counts, local);
        }
        if counts.is_empty() {
            anyhow::bail!("no training texts read from parquet");
        }

        let n_values: Vec<usize> = (n_min..=n_max).collect();
        let cfg = TokenizerConfig {
            num_merges,
            n_values,
            aim_token_num,
            max_token_chars,
            forbid_punct_word_mix,
            forbid_punct_tokens,
            allow_space_punct_tokens,
            allow_abbrev_tokens,
            allow_hyphen_tokens,
            allow_word_final_punct_tokens,
            allow_apostrophe_tokens,
            allow_cross_word_punct_word_mix_tokens,
            cross_word_start_vocab,
            max_token_words,
            forbid_incomplete_cross_word,
            recompute_each_step: false,
            use_heap,
            num_workers,
            use_multiprocess: multi_process,
        };
        let tk = LengthTokenizer::new_from_counts(counts, corpus_chars, cfg)?;
        hf_export::export_from_trained(&tk, Path::new(out_dir))?;
        Ok(())
    })();

    match prev_mp {
        Some(v) => env::set_var("MP_PY_WORKER", v),
        None => env::remove_var("MP_PY_WORKER"),
    }

    res.map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
    Ok(())
}

/// worker 入口：供 multi-process 训练时子进程调用（不要在普通代码里直接用）
#[pyfunction]
fn _run_worker() -> PyResult<()> {
    crate::run_worker().map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
    Ok(())
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
        let forbid_end_inner = forbid_end_inner_enabled();
        let cross_word_whole_only = (!forbid_end_inner) && cross_word_whole_only_enabled();
        let forbid_punct_tokens = forbid_punct_tokens_enabled();
        let mut trie = TokenTrie::new();
        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        let mut require_word_start: Vec<u8> = vec![0u8; max_id.saturating_add(1)];
        // 注意：HashMap 遍历无序，但这里只依赖 term_id，插入顺序无关
        for (tok, &id) in &vocab {
            if forbid_punct_tokens && tok.chars().count() > 1 && has_hard_punct(tok) {
                continue;
            }
            let end_inner = has_end_before_last(tok);
            if forbid_end_inner && end_inner {
                continue;
            }
            if cross_word_whole_only && end_inner {
                // Cross-word token must end with END_TOKEN, otherwise it would end mid-word.
                if !tok.ends_with(END_TOKEN) {
                    continue;
                }
                // Also require token itself begins with a non-END char, i.e. it starts with a word.
                // (This matches the notion "token covers complete words".)
                if tok.starts_with(END_TOKEN) {
                    continue;
                }
                // Mark as requiring word-start position in DP.
                if (id as usize) < require_word_start.len() {
                    require_word_start[id as usize] = 1;
                }
            }
            trie.insert(tok, id);
        }

        Ok(Self {
            trie,
            unk_id,
            require_word_start,
            cross_word_whole_only,
        })
    }

    /// encode：返回 token id 列表（DP 最少 token，无法匹配时用 unk 兜底）
    fn encode(&self, text: &str) -> Vec<u32> {
        // 复刻 Rust 训练口径：split_whitespace + 每词追加 END_TOKEN
        let chars = LengthTokenizer::normalize_chars(text);
        if self.cross_word_whole_only {
            let end_ch = END_TOKEN.chars().next().unwrap_or('Ġ');
            self.trie
                .dp_min_ids_allow_unk_require_word_start(&chars, self.unk_id, &self.require_word_start, end_ch)
        } else {
            self.trie.dp_min_ids_allow_unk(&chars, self.unk_id)
        }
    }

    /// 批量 encode：释放 GIL，适合大吞吐
    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        let trie = &self.trie;
        let unk_id = self.unk_id;
        let cross_word_whole_only = self.cross_word_whole_only;
        let require_word_start = &self.require_word_start;
        let end_ch = END_TOKEN.chars().next().unwrap_or('Ġ');
        py.allow_threads(|| {
            Ok(texts
                // rayon：并行处理 batch（collect 保持原顺序）
                .into_par_iter()
                .map(|t| {
                    let chars = LengthTokenizer::normalize_chars(&t);
                    if cross_word_whole_only {
                        trie.dp_min_ids_allow_unk_require_word_start(&chars, unk_id, require_word_start, end_ch)
                    } else {
                        trie.dp_min_ids_allow_unk(&chars, unk_id)
                    }
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
    m.add_function(wrap_pyfunction!(train_to_hf, m)?)?;
    m.add_function(wrap_pyfunction!(train_to_hf_iter, m)?)?;
    m.add_function(wrap_pyfunction!(train_to_hf_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(_run_worker, m)?)?;
    Ok(())
}


