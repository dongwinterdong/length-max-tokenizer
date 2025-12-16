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

    /// 最大 n 值（会生成 [2..=n]）
    #[arg(long, default_value_t = 6)]
    n_max: usize,

    /// 每步全量重算统计（调试/验证用，默认关闭）
    #[arg(long, default_value_t = false)]
    recompute_each_step: bool,

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

    let n_values: Vec<usize> = (2..=args.n_max).collect();
    let cfg = TokenizerConfig {
        num_merges: args.num_merges,
        n_values,
        aim_token_num: args.aim_token_num,
        recompute_each_step: args.recompute_each_step,
        num_workers: args.num_workers,
        use_multiprocess: args.multi_process,
    };

    log_main(
        "main",
        format!(
            "start corpus={:?} format={:?} max_docs={:?} output={:?} merges={} aim_token_num={} n_max={} recompute_each_step={} multi_process={} num_workers={}",
            args.corpus,
            fmt,
            max_docs,
            args.output,
            args.num_merges,
            args.aim_token_num,
            args.n_max,
            args.recompute_each_step,
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

