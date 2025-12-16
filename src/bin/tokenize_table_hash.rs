use anyhow::{Context, Result};
use clap::Parser;
use length_tokenizer::{LengthTokenizer, TokenizerConfig};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

/// Deterministic FNV-1a 64-bit hash (stable across languages and runs).
#[inline]
fn fnv1a64_update(mut h: u64, bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    if h == 0 {
        h = FNV_OFFSET;
    }
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[derive(Debug, Parser)]
#[command(
    name = "tokenize_table_hash",
    about = "Tokenize a text corpus with a token_table.json (DP min-token) and output a deterministic hash"
)]
struct Args {
    /// token_table.json path (output of training)
    #[arg(long)]
    table: PathBuf,

    /// corpus text file (one sentence per line)
    #[arg(long)]
    corpus: PathBuf,

    /// Max lines to process (0 = unlimited)
    #[arg(long, default_value_t = 1000)]
    max_lines: usize,

    /// Max n for config (only used for load/rebuild; tokenization itself is DP over final vocab)
    #[arg(long, default_value_t = 6)]
    n_max: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let cfg = TokenizerConfig {
        num_merges: 0,
        n_values: (2..=args.n_max).collect(),
        aim_token_num: 0,
        recompute_each_step: false,
        num_workers: 0,
        use_multiprocess: false,
    };

    let tokenizer = LengthTokenizer::load(&args.table, cfg)
        .with_context(|| format!("load token table failed: {:?}", args.table))?;

    let f = File::open(&args.corpus).with_context(|| format!("open corpus failed: {:?}", args.corpus))?;
    let reader = BufReader::new(f);

    let mut lines: u64 = 0;
    let mut total_tokens: u64 = 0;
    let mut h: u64 = 0; // 0 means "uninitialized" for fnv

    for line in reader.lines() {
        let l = line?;
        if l.trim().is_empty() {
            continue;
        }
        lines += 1;
        let toks = tokenizer.tokenize(&l);
        total_tokens += toks.len() as u64;
        for t in toks {
            h = fnv1a64_update(h, t.as_bytes());
            h = fnv1a64_update(h, b"\0");
        }
        h = fnv1a64_update(h, b"\n");

        if args.max_lines > 0 && (lines as usize) >= args.max_lines {
            break;
        }
    }

    // JSON to stdout (easy to parse in Python)
    println!(
        "{{\"lines\":{},\"total_tokens\":{},\"fnv64\":\"0x{:016x}\"}}",
        lines, total_tokens, h
    );
    Ok(())
}




