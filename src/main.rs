use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use length_tokenizer::{LengthTokenizer, TokenizerConfig};
use humantime::format_rfc3339_millis;
#[cfg(feature = "parquet")]
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::path::PathBuf;
use std::time::SystemTime;

fn log_main(tag: &str, msg: impl AsRef<str>) {
    let ts = format_rfc3339_millis(SystemTime::now());
    let pid = std::process::id();
    eprintln!("[{ts}][pid={pid}][{tag}] {}", msg.as_ref());
}

fn log_main_debug(tag: &str, msg: impl AsRef<str>) {
    let lvl = std::env::var("LOG_LEVEL").ok().unwrap_or_else(|| "info".to_string());
    let dbg = lvl.trim().eq_ignore_ascii_case("debug") || std::env::var("LOG_DEBUG").is_ok();
    if dbg {
        log_main(tag, msg);
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum CorpusFormat {
    /// 自动判断：目录或 .parquet -> parquet；否则当作每行一句的纯文本
    Auto,
    /// 纯文本，每行一句
    Txt,
    /// Parquet（FineWeb / FineWeb-Edu）：读取 text 列
    Parquet,
}

#[derive(Debug, Parser)]
#[command(name = "length-tokenizer")]
#[command(about = "Rust port of length.py (BPE-style, multi-gram)", long_about = None)]
struct Args {
    /// 语料文件（每行一句）
    #[arg(short, long, default_value = "corpus_py.txt")]
    corpus: PathBuf,

    /// 语料格式（auto/txt/parquet）
    #[arg(long, value_enum, default_value_t = CorpusFormat::Auto)]
    corpus_format: CorpusFormat,

    /// Parquet 模式：读取的文本列名（FineWeb-Edu 默认是 `text`）
    #[arg(long, default_value = "text")]
    text_column: String,

    /// 仅读取前 N 条样本（0=不限制）。用于在超大语料上做快速试跑/消融。
    #[arg(long, default_value_t = 0)]
    max_docs: usize,

    /// Parquet 模式：batch size（rows/RecordBatch）
    #[arg(long, default_value_t = 8192)]
    parquet_batch_size: usize,

    /// Parquet 模式：递归扫描子目录下的 .parquet
    #[arg(long, default_value_t = false)]
    parquet_recursive: bool,

    /// 训练输出文件
    #[arg(short, long, default_value = "token_table.json")]
    output: PathBuf,

    /// 仅训练/计时，不写出 token_table.json（避免大文件序列化耗时与占用）
    #[arg(long, default_value_t = false)]
    no_save: bool,

    /// 合并次数
    #[arg(long, default_value_t = 500)]
    num_merges: usize,

    /// 目标词表上限
    #[arg(long, default_value_t = 15_000)]
    aim_token_num: usize,

    /// 最小 n 值（会生成 [n_min..=n_max]）
    #[arg(long, default_value_t = 2)]
    n_min: usize,

    /// 最大 n 值（会生成 [2..=n]）
    #[arg(long, default_value_t = 6)]
    n_max: usize,

    /// 限制新生成 token 的最大字符长度（0 表示不限制）
    #[arg(long, default_value_t = 0)]
    max_token_chars: usize,

    /// 禁止生成“混合硬标点 + 词元(字母数字)”的新 token（训练阶段过滤候选 merge）
    #[arg(long, default_value_t = false)]
    forbid_punct_word_mix: bool,

    /// 禁止生成“包含硬标点（ASCII 标点，排除 ' 与 -）”的新 token（训练阶段过滤候选 merge）
    #[arg(long, default_value_t = false)]
    forbid_punct_tokens: bool,

    /// 当启用 --forbid-punct-tokens 时，仍允许生成“空格+标点”token（在本实现中表现为标点+'Ġ'，例如 ",Ġ" ".Ġ" "@-@Ġ"）
    #[arg(long, default_value_t = false)]
    allow_space_punct_tokens: bool,

    /// 当启用 --forbid-punct-tokens 时，仍允许生成“缩写/点分模式”token（例如 "U.S.Ġ", "e.g.Ġ"）
    #[arg(long, default_value_t = false)]
    allow_abbrev_tokens: bool,

    /// 当启用 --forbid-punct-tokens 时，仍允许生成“连字符模式”token（例如 "well-knownĠ"）
    #[arg(long, default_value_t = false)]
    allow_hyphen_tokens: bool,

    /// 当启用 --forbid-punct-tokens 时，仍允许生成“词尾标点”token（例如 "word,Ġ"）
    #[arg(long, default_value_t = false)]
    allow_word_final_punct_tokens: bool,

    /// 当启用 --forbid-punct-tokens 时，仍允许生成“撇号缩写/所有格”token（例如 "don'tĠ", "John'sĠ"）
    #[arg(long, default_value_t = false)]
    allow_apostrophe_tokens: bool,

    /// 当启用 --forbid-punct-tokens 时，仍允许生成“跨词 + 标点混合”的 token（受控白名单）。
    ///
    /// 主要用于吸收 SuperBPE 的低 TPC 优势（例如 ",ĠtheĠ" ".ĠAtĠtheĠ" 等模式）。
    #[arg(long, default_value_t = false)]
    allow_cross_word_punct_word_mix_tokens: bool,

    /// SuperBPE 风格：当词表大小达到该阈值后，才允许生成跨词(superword) token（>=2 个 'Ġ'）
    ///
    /// 0 表示不限制（从一开始就允许）。
    #[arg(long, default_value_t = 0)]
    cross_word_start_vocab: usize,

    /// 限制单个 token 最多包含多少个词（按 'Ġ' 计数）；0 表示不限制
    #[arg(long, default_value_t = 0)]
    max_token_words: usize,

    /// 禁止生成“跨词但不完整”的 token：包含 'Ġ' 但不以 'Ġ' 结尾（训练阶段过滤候选 merge）
    #[arg(long, default_value_t = false)]
    forbid_incomplete_cross_word: bool,

    /// 每步全量重算统计（调试/验证用，默认关闭）
    #[arg(long, default_value_t = false)]
    recompute_each_step: bool,

    /// 是否启用候选堆（BinaryHeap）来加速“找 best n-gram”。
    ///
    /// 注意：heap 会复制一份 n-gram key，带来显著额外内存；
    /// 在大语料/大 n_max（例如 9）时非常容易触发 OOM。默认关闭。
    #[arg(long, default_value_t = false)]
    use_heap: bool,

    /// 线程/进程分片数量（0 表示自动=CPU核数）
    #[arg(long, default_value_t = 0)]
    num_workers: usize,

    /// 启用多进程模式（默认单进程+多线程）
    #[arg(long, default_value_t = false)]
    multi_process: bool,

    /// 内部使用：以 worker 进程身份启动
    #[arg(long, hide = true, default_value_t = false)]
    as_worker: bool,
}

fn load_txt_corpus(path: &Path, max_docs: Option<usize>) -> Result<Vec<String>> {
    let f = File::open(path).with_context(|| format!("open corpus txt failed: {path:?}"))?;
    let reader = BufReader::new(f);
    let mut corpus: Vec<String> = Vec::new();
    for line in reader.lines() {
        let l = line?;
        if l.trim().is_empty() {
            continue;
        }
        corpus.push(l);
        if let Some(m) = max_docs {
            if corpus.len() >= m {
                break;
            }
        }
    }
    Ok(corpus)
}

fn detect_format(path: &Path, fmt: CorpusFormat) -> CorpusFormat {
    match fmt {
        CorpusFormat::Auto => {
            if path.is_dir() {
                return CorpusFormat::Parquet;
            }
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if ext.eq_ignore_ascii_case("parquet") {
                    return CorpusFormat::Parquet;
                }
            }
            CorpusFormat::Txt
        }
        _ => fmt,
    }
}

#[cfg(feature = "parquet")]
fn collect_parquet_files(root: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    fn walk(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) -> Result<()> {
        for ent in fs::read_dir(dir).with_context(|| format!("read_dir failed: {dir:?}"))? {
            let ent = ent?;
            let path = ent.path();
            if path.is_dir() {
                if recursive {
                    walk(&path, true, out)?;
                }
                continue;
            }
            if path
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("parquet"))
                .unwrap_or(false)
            {
                out.push(path);
            }
        }
        Ok(())
    }

    let mut files: Vec<PathBuf> = Vec::new();
    if root.is_file() {
        files.push(root.to_path_buf());
    } else {
        walk(root, recursive, &mut files)?;
    }
    files.sort();
    Ok(files)
}

#[cfg(feature = "parquet")]
fn load_parquet_corpus(
    path: &Path,
    text_column: &str,
    max_docs: Option<usize>,
    batch_size: usize,
    recursive: bool,
) -> Result<Vec<String>> {
    use arrow::array::{Array, LargeStringArray, StringArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let files = collect_parquet_files(path, recursive)?;
    if files.is_empty() {
        bail!("no parquet files found under {:?}", path);
    }
    let mut out: Vec<String> = Vec::with_capacity(max_docs.unwrap_or(0).min(1_000_000));

    for p in files {
        let f = File::open(&p).with_context(|| format!("open parquet failed: {p:?}"))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(f)
            .with_context(|| format!("parquet reader init failed: {p:?}"))?;
        let mut reader = builder
            .with_batch_size(batch_size.max(1))
            .build()
            .with_context(|| format!("parquet reader build failed: {p:?}"))?;

        while let Some(batch) = reader.next() {
            let batch = batch.with_context(|| format!("read parquet batch failed: {p:?}"))?;
            let Some(arr) = batch.column_by_name(text_column) else {
                bail!("parquet file {:?} missing text column {:?}", p, text_column);
            };
            if let Some(col) = arr.as_any().downcast_ref::<StringArray>() {
                for i in 0..col.len() {
                    if col.is_null(i) {
                        continue;
                    }
                    out.push(col.value(i).to_owned());
                    if let Some(m) = max_docs {
                        if out.len() >= m {
                            return Ok(out);
                        }
                    }
                }
            } else if let Some(col) = arr.as_any().downcast_ref::<LargeStringArray>() {
                for i in 0..col.len() {
                    if col.is_null(i) {
                        continue;
                    }
                    out.push(col.value(i).to_owned());
                    if let Some(m) = max_docs {
                        if out.len() >= m {
                            return Ok(out);
                        }
                    }
                }
            } else {
                bail!(
                    "parquet text column {:?} has unsupported type (expected String/LargeString)",
                    text_column
                );
            }
        }
    }
    Ok(out)
}

#[cfg(not(feature = "parquet"))]
fn load_parquet_corpus(
    _path: &Path,
    _text_column: &str,
    _max_docs: Option<usize>,
    _batch_size: usize,
    _recursive: bool,
) -> Result<Vec<String>> {
    bail!("parquet support is not compiled. Rebuild with `--features parquet` (or enable default features).")
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.as_worker {
        // Worker 模式：仅处理主进程通过 stdin 发来的指令
        return length_tokenizer::run_worker();
    }
    let max_docs = if args.max_docs == 0 { None } else { Some(args.max_docs) };
    let fmt = detect_format(&args.corpus, args.corpus_format);
    let corpus = match fmt {
        CorpusFormat::Txt => load_txt_corpus(&args.corpus, max_docs)?,
        CorpusFormat::Parquet => load_parquet_corpus(
            &args.corpus,
            &args.text_column,
            max_docs,
            args.parquet_batch_size,
            args.parquet_recursive,
        )?,
        CorpusFormat::Auto => unreachable!("auto resolved in detect_format"),
    };

    if args.n_min < 2 {
        bail!("--n-min must be >= 2");
    }
    if args.n_max < args.n_min {
        bail!("--n-max must be >= --n-min");
    }
    let n_values: Vec<usize> = (args.n_min..=args.n_max).collect();
    let cfg = TokenizerConfig {
        num_merges: args.num_merges,
        n_values,
        aim_token_num: args.aim_token_num,
        max_token_chars: args.max_token_chars,
        forbid_punct_word_mix: args.forbid_punct_word_mix,
        forbid_punct_tokens: args.forbid_punct_tokens,
        allow_space_punct_tokens: args.allow_space_punct_tokens,
        allow_abbrev_tokens: args.allow_abbrev_tokens,
        allow_hyphen_tokens: args.allow_hyphen_tokens,
        allow_word_final_punct_tokens: args.allow_word_final_punct_tokens,
        allow_apostrophe_tokens: args.allow_apostrophe_tokens,
        allow_cross_word_punct_word_mix_tokens: args.allow_cross_word_punct_word_mix_tokens,
        cross_word_start_vocab: args.cross_word_start_vocab,
        max_token_words: args.max_token_words,
        forbid_incomplete_cross_word: args.forbid_incomplete_cross_word,
        recompute_each_step: args.recompute_each_step,
        use_heap: args.use_heap,
        num_workers: args.num_workers,
        use_multiprocess: args.multi_process,
    };

    log_main(
        "main",
        format!(
            "start corpus={:?} format={:?} max_docs={:?} output={:?} merges={} aim_token_num={} n_min={} n_max={} max_token_chars={} forbid_punct_word_mix={} forbid_punct_tokens={} allow_space_punct_tokens={} allow_abbrev_tokens={} allow_hyphen_tokens={} allow_word_final_punct_tokens={} allow_apostrophe_tokens={} allow_cross_word_punct_word_mix_tokens={} cross_word_start_vocab={} max_token_words={} forbid_incomplete_cross_word={} recompute_each_step={} use_heap={} multi_process={} num_workers={}",
            args.corpus,
            fmt,
            max_docs,
            args.output,
            args.num_merges,
            args.aim_token_num,
            args.n_min,
            args.n_max,
            args.max_token_chars,
            args.forbid_punct_word_mix,
            args.forbid_punct_tokens,
            args.allow_space_punct_tokens,
            args.allow_abbrev_tokens,
            args.allow_hyphen_tokens,
            args.allow_word_final_punct_tokens,
            args.allow_apostrophe_tokens,
            args.allow_cross_word_punct_word_mix_tokens,
            args.cross_word_start_vocab,
            args.max_token_words,
            args.forbid_incomplete_cross_word,
            args.recompute_each_step,
            args.use_heap,
            args.multi_process,
            args.num_workers
        ),
    );

    let tokenizer = LengthTokenizer::new(&corpus, cfg)?;
    if args.no_save {
        log_main("main", "no_save=true, skip writing token table");
    } else {
    tokenizer.save(&args.output)?;
    log_main("main", format!("saved token table to {:?}", args.output));
    }

    // 简单示例：对第一行做分词
    if let Some(sample) = corpus.first() {
        let toks = tokenizer.tokenize(sample);
        // 调试信息：默认不打印，避免污染关键训练日志
        log_main_debug("sample", format!("tokens(first line)={:?}", toks));
    }

    Ok(())
}

