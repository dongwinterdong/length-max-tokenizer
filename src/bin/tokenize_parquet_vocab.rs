use anyhow::{bail, Context, Result};
use clap::Parser;
use std::collections::HashMap as StdHashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use arrow::array::{Array, LargeStringArray, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const END_TOKEN: char = 'Ġ';

#[derive(Default)]
struct TrieNode {
    next: StdHashMap<char, usize>,
    term_id: Option<u32>,
}

#[derive(Default)]
struct TokenTrie {
    nodes: Vec<TrieNode>,
}

impl TokenTrie {
    fn new() -> Self {
        Self {
            nodes: vec![TrieNode::default()],
        }
    }

    fn insert(&mut self, token: &str, id: u32) {
        let mut cur = 0usize;
        for ch in token.chars() {
            let nxt = match self.nodes[cur].next.get(&ch) {
                Some(&idx) => idx,
                None => {
                    let idx = self.nodes.len();
                    self.nodes.push(TrieNode::default());
                    self.nodes[cur].next.insert(ch, idx);
                    idx
                }
            };
            cur = nxt;
        }
        self.nodes[cur].term_id = Some(id);
    }

    /// DP min-token with unk fallback (consume 1 char -> unk_id when no token matches).
    fn dp_min_ids_allow_unk(&self, s: &[char], unk_id: u32) -> Vec<u32> {
        let n = s.len();
        if n == 0 {
            return Vec::new();
        }
        let inf = u32::MAX / 8;
        let mut dp: Vec<u32> = vec![inf; n + 1];
        let mut back: Vec<Option<(usize, u32)>> = vec![None; n + 1];
        dp[n] = 0;

        for i in (0..n).rev() {
            let mut node = 0usize;
            for j in i..n {
                let Some(&child) = self.nodes[node].next.get(&s[j]) else {
                    break;
                };
                node = child;
                if let Some(tok_id) = self.nodes[node].term_id {
                    let cand = 1u32.saturating_add(dp[j + 1]);
                    let better = if cand < dp[i] {
                        true
                    } else if cand == dp[i] {
                        // tie-break: prefer longer token
                        match back[i] {
                            Some((best_j, _)) => (j + 1 - i) > (best_j - i),
                            None => true,
                        }
                    } else {
                        false
                    };
                    if better {
                        dp[i] = cand;
                        back[i] = Some((j + 1, tok_id));
                    }
                }
            }
            if back[i].is_none() {
                dp[i] = 1u32.saturating_add(dp[i + 1]);
                back[i] = Some((i + 1, unk_id));
            }
        }

        let mut out: Vec<u32> = Vec::new();
        let mut i = 0usize;
        while i < n {
            let Some((j, id)) = back[i] else {
                out.push(unk_id);
                i += 1;
                continue;
            };
            out.push(id);
            i = j;
        }
        out
    }
}

fn normalize_chars(text: &str) -> Vec<char> {
    let mut out: Vec<char> = Vec::new();
    for w in text.split_whitespace() {
        out.extend(w.chars());
        out.push(END_TOKEN);
    }
    out
}

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

#[derive(Debug, Parser)]
#[command(name = "tokenize_parquet_vocab", about = "Tokenize FineWeb(-Edu) parquet text column with vocab.json (DP min-token)")]
struct Args {
    /// vocab.json (token -> id)
    #[arg(long)]
    vocab: PathBuf,

    /// unk token string (must exist in vocab)
    #[arg(long, default_value = "<unk>")]
    unk_token: String,

    /// Parquet file or directory
    #[arg(long)]
    parquet: PathBuf,

    /// Recursively scan parquet directory
    #[arg(long, default_value_t = false)]
    recursive: bool,

    /// Text column name (FineWeb-Edu: text)
    #[arg(long, default_value = "text")]
    text_column: String,

    /// Parquet batch size (rows)
    #[arg(long, default_value_t = 8192)]
    batch_size: usize,

    /// Max docs to tokenize (0 = unlimited)
    #[arg(long, default_value_t = 0)]
    max_docs: usize,

    /// Output ids file (ltbin format). If omitted, only prints stats.
    #[arg(long, default_value = "")]
    out: String,

    /// Print progress every N docs
    #[arg(long, default_value_t = 10000)]
    progress_every: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load vocab.json
    let f = File::open(&args.vocab).with_context(|| format!("open vocab failed: {:?}", args.vocab))?;
    let vocab: StdHashMap<String, u32> =
        serde_json::from_reader(f).with_context(|| format!("parse vocab.json failed: {:?}", args.vocab))?;
    let unk_id = *vocab
        .get(&args.unk_token)
        .ok_or_else(|| anyhow::anyhow!("unk_token {:?} not found in vocab", args.unk_token))?;

    // Build trie
    let mut trie = TokenTrie::new();
    for (tok, &id) in &vocab {
        trie.insert(tok, id);
    }

    let files = collect_parquet_files(&args.parquet, args.recursive)?;
    if files.is_empty() {
        bail!("no parquet files found under {:?}", args.parquet);
    }

    let mut writer: Option<BufWriter<File>> = None;
    if !args.out.is_empty() {
        let out_path = PathBuf::from(&args.out);
        let f = File::create(&out_path).with_context(|| format!("create out failed: {out_path:?}"))?;
        writer = Some(BufWriter::new(f));
    }

    let mut docs: u64 = 0;
    let mut total_tokens: u64 = 0;
    let mut total_chars: u64 = 0;

    for p in files {
        let f = File::open(&p).with_context(|| format!("open parquet failed: {p:?}"))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(f)
            .with_context(|| format!("parquet reader init failed: {p:?}"))?;
        let mut reader = builder
            .with_batch_size(args.batch_size.max(1))
            .build()
            .with_context(|| format!("parquet reader build failed: {p:?}"))?;

        while let Some(batch) = reader.next() {
            let batch = batch.with_context(|| format!("read parquet batch failed: {p:?}"))?;
            let Some(arr) = batch.column_by_name(&args.text_column) else {
                bail!("parquet file {:?} missing text column {:?}", p, args.text_column);
            };

            // Iterate rows
            if let Some(col) = arr.as_any().downcast_ref::<StringArray>() {
                for i in 0..col.len() {
                    if col.is_null(i) {
                        continue;
                    }
                    let s = col.value(i);
                    let chars = normalize_chars(s);
                    total_chars += s.chars().count() as u64;
                    let ids = trie.dp_min_ids_allow_unk(&chars, unk_id);
                    total_tokens += ids.len() as u64;
                    docs += 1;

                    if let Some(w) = writer.as_mut() {
                        // ltbin: u32 len + u32 ids (LE)
                        let len = ids.len() as u32;
                        w.write_all(&len.to_le_bytes())?;
                        for id in ids {
                            w.write_all(&id.to_le_bytes())?;
                        }
                    }

                    if args.progress_every > 0 && (docs as usize) % args.progress_every == 0 {
                        eprintln!(
                            "[tok] docs={} total_tokens={} tpc={:.6}",
                            docs,
                            total_tokens,
                            if total_chars == 0 { 0.0 } else { total_tokens as f64 / total_chars as f64 }
                        );
                    }
                    if args.max_docs > 0 && (docs as usize) >= args.max_docs {
                        break;
                    }
                }
            } else if let Some(col) = arr.as_any().downcast_ref::<LargeStringArray>() {
                for i in 0..col.len() {
                    if col.is_null(i) {
                        continue;
                    }
                    let s = col.value(i);
                    let chars = normalize_chars(s);
                    total_chars += s.chars().count() as u64;
                    let ids = trie.dp_min_ids_allow_unk(&chars, unk_id);
                    total_tokens += ids.len() as u64;
                    docs += 1;

                    if let Some(w) = writer.as_mut() {
                        let len = ids.len() as u32;
                        w.write_all(&len.to_le_bytes())?;
                        for id in ids {
                            w.write_all(&id.to_le_bytes())?;
                        }
                    }

                    if args.progress_every > 0 && (docs as usize) % args.progress_every == 0 {
                        eprintln!(
                            "[tok] docs={} total_tokens={} tpc={:.6}",
                            docs,
                            total_tokens,
                            if total_chars == 0 { 0.0 } else { total_tokens as f64 / total_chars as f64 }
                        );
                    }
                    if args.max_docs > 0 && (docs as usize) >= args.max_docs {
                        break;
                    }
                }
            } else {
                bail!(
                    "text column {:?} has unsupported type (expected String/LargeString)",
                    args.text_column
                );
            }

            if args.max_docs > 0 && (docs as usize) >= args.max_docs {
                break;
            }
        }
        if args.max_docs > 0 && (docs as usize) >= args.max_docs {
            break;
        }
    }

    if let Some(mut w) = writer {
        w.flush()?;
    }

    let tpc = if total_chars == 0 { 0.0 } else { total_tokens as f64 / total_chars as f64 };
    println!(
        "{{\"docs\":{},\"total_chars\":{},\"total_tokens\":{},\"tpc\":{:.6}}}",
        docs, total_chars, total_tokens, tpc
    );
    Ok(())
}


