use anyhow::Result;
use length_tokenizer::{LengthTokenizer, TokenizerConfig};
use std::fs::File;
use std::io::{BufRead, BufReader};

/// 用途：检查“训练完词表后的应用分词（tokenize）”是否真的在降低 TPC（tokens per character）。
///
/// 说明：
/// - 这里的“baseline token 数”按当前实现的基础编码规则计算：对每个 word 拆成字符 + 追加一个 END_TOKEN(Ġ)。
/// - “tokenize 后 token 数”使用 `LengthTokenizer::tokenize()` 的输出 token 数（Vec<String> 长度）。
/// - 字符数用 Rust `chars().count()`（与训练里 `total_chars()` 的口径一致）。
///
/// 用法：
/// ```bash
/// cargo run --release --bin tpc_corpus -- <token_table.json> <corpus.txt> [max_lines]
/// ```
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("用法: {} <token_table.json> <corpus.txt> [max_lines]", args[0]);
        eprintln!("示例: {} token_table_smoke.json corpus_smoke.txt 1000", args[0]);
        std::process::exit(2);
    }

    let table_path = &args[1];
    let corpus_path = &args[2];
    let max_lines: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);

    // 注意：当前库的 load 会重建 global_stats（大表可能很慢/很吃内存）。
    // 这里用于“正确性/指标”检查，建议先用小表（smoke/sample）验证。
    let cfg = TokenizerConfig::default();
    let tk = LengthTokenizer::load(table_path, cfg)?;

    let f = File::open(corpus_path)?;
    let reader = BufReader::new(f);

    let mut total_chars: u64 = 0;
    let mut baseline_tokens: u64 = 0;
    let mut tokenized_tokens: u64 = 0;
    let mut seen_lines: u64 = 0;

    for line in reader.lines() {
        if seen_lines as usize >= max_lines {
            break;
        }
        let s = line?;
        let s = s.trim_end_matches('\r');
        if s.trim().is_empty() {
            continue;
        }

        seen_lines += 1;
        total_chars += s.chars().count() as u64;

        // baseline：字符 + 每个 word 一个 END_TOKEN
        let mut words = 0u64;
        let mut chars_in_words = 0u64;
        for w in s.split_whitespace() {
            words += 1;
            chars_in_words += w.chars().count() as u64;
        }
        baseline_tokens += chars_in_words + words;

        // tokenize：应用 merges 后的 token 数
        let toks = tk.tokenize(s);
        tokenized_tokens += toks.len() as u64;
    }

    let tpc_base = if total_chars == 0 {
        0.0
    } else {
        baseline_tokens as f64 / total_chars as f64
    };
    let tpc_tok = if total_chars == 0 {
        0.0
    } else {
        tokenized_tokens as f64 / total_chars as f64
    };

    println!("lines={}", seen_lines);
    println!("total_chars={}", total_chars);
    println!("baseline_tokens={} tpc_base={:.6} tok/char", baseline_tokens, tpc_base);
    println!(
        "tokenized_tokens={} tpc_tokenized={:.6} tok/char",
        tokenized_tokens, tpc_tok
    );
    println!(
        "reduction: tokens {} -> {} ({:.2}%)",
        baseline_tokens,
        tokenized_tokens,
        if baseline_tokens == 0 {
            0.0
        } else {
            100.0 * (baseline_tokens.saturating_sub(tokenized_tokens)) as f64 / baseline_tokens as f64
        }
    );

    Ok(())
}


