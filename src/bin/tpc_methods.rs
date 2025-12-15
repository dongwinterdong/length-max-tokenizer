use anyhow::Result;
use hashbrown::HashMap;
use serde::Deserialize;
use serde::de::{self, DeserializeSeed, MapAccess, Visitor};
use std::cmp::min;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// 目标：在“当前训练方法产出的 token_table（merges）”上，对比不同应用分词策略的 TPC（tokens per character）。
///
/// 对比策略：
/// - **merge**：按 merges 顺序依次扫描并做替换（与你现在 `tokenize()` 的语义一致）
/// - **dp_min**：给定 token 词表（由 merges 的 parts+replacement + 语料中出现的单字符 + END_TOKEN 构成），用 DP 找到全局最少 token 切分
/// - **greedy_longest**：贪心“最长匹配”切分（max-munch），通常比 merge 快，但不保证最优
///
/// 注意：
/// - 为避免 2GB+ 的 `vocab` 导致 OOM，本工具只解析 JSON 根对象中的 `merges`，一旦遇到 `vocab` key 立刻停止（不读取 value）。
/// - TPC 分母口径与当前训练一致：用原始行的 `chars().count()`（包括空白），分子是分词后的 token 数。

const END_TOKEN: &str = "Ġ";
const STOP_PREFIX: &str = "__STOP_AFTER_MERGES__";

#[derive(Debug, Clone, Deserialize)]
struct MergeRuleLite {
    parts: Vec<String>,
    replacement: String,
    #[serde(default)]
    freq: u32,
    #[serde(default)]
    score: u64,
}

/// 只从 token_table.json 中读取 merges（不读 vocab）
fn load_merges_only(path: &Path) -> Result<Vec<MergeRuleLite>> {
    let f = File::open(path)?;
    let reader = BufReader::with_capacity(16 * 1024 * 1024, f);
    let mut de = serde_json::Deserializer::from_reader(reader);

    let mut merges: Vec<MergeRuleLite> = Vec::new();
    let seed = RootSeed { merges: &mut merges };

    match seed.deserialize(&mut de) {
        Ok(()) => Ok(merges),
        Err(e) => {
            let msg = e.to_string();
            if msg.starts_with(STOP_PREFIX) {
                Ok(merges)
            } else {
                Err(e.into())
            }
        }
    }
}

struct RootSeed<'a> {
    merges: &'a mut Vec<MergeRuleLite>,
}

impl<'de, 'a> DeserializeSeed<'de> for RootSeed<'a> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(RootVisitor { merges: self.merges })
    }
}

struct RootVisitor<'a> {
    merges: &'a mut Vec<MergeRuleLite>,
}

impl<'de, 'a> Visitor<'de> for RootVisitor<'a> {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "token table root object")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "merges" => {
                    let v: Vec<MergeRuleLite> = map.next_value()?;
                    *self.merges = v;
                }
                "vocab" => {
                    // 不读取 value，直接停止解析（避免扫描 2GB+ vocab）
                    return Err(de::Error::custom(STOP_PREFIX));
                }
                _ => {
                    let _ = map.next_value::<de::IgnoredAny>()?;
                }
            }
        }
        Ok(())
    }
}

/// 一个很轻量的 interner：把 token String 映射为 u32 id
#[derive(Default)]
struct Interner {
    map: HashMap<String, u32>,
    next: u32,
}

impl Interner {
    fn id(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.next;
        self.next = self.next.wrapping_add(1);
        self.map.insert(s.to_string(), id);
        id
    }
}

#[derive(Default)]
struct TrieNode {
    next: HashMap<char, usize>,
    terminal: bool,
}

struct Trie {
    nodes: Vec<TrieNode>,
}

impl Trie {
    fn new() -> Self {
        Self {
            nodes: vec![TrieNode::default()],
        }
    }

    fn insert(&mut self, token: &str) {
        let mut cur = 0usize;
        for ch in token.chars() {
            let nxt = if let Some(&idx) = self.nodes[cur].next.get(&ch) {
                idx
            } else {
                let idx = self.nodes.len();
                self.nodes.push(TrieNode::default());
                self.nodes[cur].next.insert(ch, idx);
                idx
            };
            cur = nxt;
        }
        self.nodes[cur].terminal = true;
    }

    /// DP：全局最少 token 数（只计 token 数，不输出具体切分）
    fn dp_min_tokens(&self, s: &[char]) -> u32 {
        let n = s.len();
        let inf = u32::MAX / 8;
        let mut dp = vec![inf; n + 1];
        dp[n] = 0;

        for i in (0..n).rev() {
            let mut node = 0usize;
            for j in i..n {
                let Some(&child) = self.nodes[node].next.get(&s[j]) else {
                    break;
                };
                node = child;
                if self.nodes[node].terminal {
                    let cand = 1u32.saturating_add(dp[j + 1]);
                    if cand < dp[i] {
                        dp[i] = cand;
                    }
                }
            }
            // 理论上不应发生：我们会把语料里出现的单字符都塞进 trie，当兜底。
            if dp[i] == inf {
                dp[i] = 1u32.saturating_add(dp[i + 1]);
            }
        }
        dp[0]
    }

    /// 贪心：每步取能匹配到的最长 token（max-munch）
    fn greedy_longest_tokens(&self, s: &[char]) -> u32 {
        let mut i = 0usize;
        let n = s.len();
        let mut cnt = 0u32;
        while i < n {
            let mut node = 0usize;
            let mut best_end: Option<usize> = None;
            for j in i..n {
                let Some(&child) = self.nodes[node].next.get(&s[j]) else {
                    break;
                };
                node = child;
                if self.nodes[node].terminal {
                    best_end = Some(j + 1);
                }
            }
            let end = best_end.unwrap_or(i + 1);
            i = min(end, n);
            cnt = cnt.saturating_add(1);
        }
        cnt
    }
}

/// 将一行按当前训练的基础规则归一化为 char 序列：
/// - split_whitespace()
/// - 每个 word 拆 char
/// - 每个 word 末尾加 END_TOKEN
fn normalize_to_chars(line: &str) -> Vec<char> {
    let mut out: Vec<char> = Vec::new();
    for w in line.split_whitespace() {
        out.extend(w.chars());
        out.extend(END_TOKEN.chars());
    }
    out
}

fn baseline_tokens_count(line: &str) -> u64 {
    // baseline token 数 = Σ chars(word) + word_count（每词一个 END_TOKEN）
    let mut words = 0u64;
    let mut chars_in_words = 0u64;
    for w in line.split_whitespace() {
        words += 1;
        chars_in_words += w.chars().count() as u64;
    }
    chars_in_words + words
}

/// merge 应用：对 token id 序列按顺序做“非重叠”替换
fn apply_merge_ids(tokens: &mut Vec<u32>, parts: &[u32], replacement: u32) {
    let plen = parts.len();
    if plen == 0 || tokens.len() < plen {
        return;
    }
    let mut out: Vec<u32> = Vec::with_capacity(tokens.len());
    let mut i = 0usize;
    while i < tokens.len() {
        if i + plen <= tokens.len() && tokens[i..i + plen] == parts[..] {
            out.push(replacement);
            i += plen;
        } else {
            out.push(tokens[i]);
            i += 1;
        }
    }
    *tokens = out;
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("用法: {} <token_table.json> <corpus.txt> [max_lines]", args[0]);
        eprintln!("示例: {} token_table_smoke.json corpus_smoke.txt 10000", args[0]);
        std::process::exit(2);
    }
    let table_path = Path::new(&args[1]);
    let corpus_path = Path::new(&args[2]);
    let max_lines: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10_000);

    eprintln!("[load] 只读取 merges: {:?}", table_path);
    let merges = load_merges_only(table_path)?;
    eprintln!("[load] merges 条数={}", merges.len());

    // 第一遍：收集语料中出现的单字符 token（作为 DP/merge 的兜底），以及统计字符数/行数
    eprintln!("[scan] 收集字符集 + 统计 chars（前 {} 行）: {:?}", max_lines, corpus_path);
    let f = File::open(corpus_path)?;
    let reader = BufReader::new(f);

    let mut total_chars: u64 = 0;
    let mut seen_lines: u64 = 0;
    let mut char_tokens: HashMap<String, ()> = HashMap::new();
    // END_TOKEN 一定要有
    char_tokens.insert(END_TOKEN.to_string(), ());

    let mut baseline_tokens: u64 = 0;
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
        baseline_tokens += baseline_tokens_count(s);

        for w in s.split_whitespace() {
            for ch in w.chars() {
                let mut buf = [0u8; 4];
                let ss = ch.encode_utf8(&mut buf);
                char_tokens.insert(ss.to_string(), ());
            }
        }
    }

    eprintln!(
        "[scan] lines={} total_chars={} baseline_tokens={} baseline_tpc={:.6}",
        seen_lines,
        total_chars,
        baseline_tokens,
        if total_chars == 0 { 0.0 } else { baseline_tokens as f64 / total_chars as f64 }
    );

    // 构建 token 词表（merges 的 parts+replacement + 单字符兜底）
    let mut token_set: HashMap<String, ()> = HashMap::new();
    for m in &merges {
        for p in &m.parts {
            token_set.insert(p.clone(), ());
        }
        token_set.insert(m.replacement.clone(), ());
    }
    for (k, _) in &char_tokens {
        token_set.insert(k.clone(), ());
    }
    eprintln!("[vocab] token_set_size={}", token_set.len());

    // 构建 trie（用于 DP/greedy）
    eprintln!("[build] 构建 trie…");
    let mut trie = Trie::new();
    for (tok, _) in &token_set {
        trie.insert(tok);
    }
    eprintln!("[build] trie_nodes={}", trie.nodes.len());

    // 构建 interner + merges id 版（用于 merge 应用）
    eprintln!("[build] 构建 interner + merges_id…");
    let mut interner = Interner::default();
    // 先把 token_set 全部分配 id（避免 apply 时临时创建 string key）
    for (tok, _) in &token_set {
        let _ = interner.id(tok);
    }

    let mut merges_id: Vec<(Vec<u32>, u32)> = Vec::with_capacity(merges.len());
    for m in &merges {
        let parts: Vec<u32> = m.parts.iter().map(|s| interner.id(s)).collect();
        let rep = interner.id(&m.replacement);
        merges_id.push((parts, rep));
    }

    // 第二遍：计算各方法 token 数
    eprintln!("[eval] 计算各方法 tokens（前 {} 行）…", max_lines);
    let f = File::open(corpus_path)?;
    let reader = BufReader::new(f);

    let mut dp_tokens: u64 = 0;
    let mut greedy_tokens: u64 = 0;
    let mut merge_tokens: u64 = 0;

    let mut used_lines: u64 = 0;
    for line in reader.lines() {
        if used_lines as usize >= max_lines {
            break;
        }
        let s = line?;
        let s = s.trim_end_matches('\r');
        if s.trim().is_empty() {
            continue;
        }
        used_lines += 1;

        let norm = normalize_to_chars(s);
        // DP / greedy
        dp_tokens += trie.dp_min_tokens(&norm) as u64;
        greedy_tokens += trie.greedy_longest_tokens(&norm) as u64;

        // merge：先编码为 token id 序列，再逐条应用 merges
        let mut toks: Vec<u32> = Vec::with_capacity(norm.len());
        for ch in &norm {
            let mut buf = [0u8; 4];
            let ss = ch.encode_utf8(&mut buf);
            toks.push(interner.id(ss));
        }
        for (parts, rep) in &merges_id {
            apply_merge_ids(&mut toks, parts, *rep);
        }
        merge_tokens += toks.len() as u64;

        if used_lines % 100 == 0 {
            eprintln!(
                "[eval] lines={} dp_tpc={:.4} greedy_tpc={:.4} merge_tpc={:.4}",
                used_lines,
                if total_chars == 0 { 0.0 } else { dp_tokens as f64 / total_chars as f64 },
                if total_chars == 0 { 0.0 } else { greedy_tokens as f64 / total_chars as f64 },
                if total_chars == 0 { 0.0 } else { merge_tokens as f64 / total_chars as f64 },
            );
        }
    }

    let tpc_dp = if total_chars == 0 { 0.0 } else { dp_tokens as f64 / total_chars as f64 };
    let tpc_g = if total_chars == 0 { 0.0 } else { greedy_tokens as f64 / total_chars as f64 };
    let tpc_m = if total_chars == 0 { 0.0 } else { merge_tokens as f64 / total_chars as f64 };

    println!("lines={}", used_lines);
    println!("total_chars={}", total_chars);
    println!("baseline_tokens={} baseline_tpc={:.6}", baseline_tokens, if total_chars==0 {0.0} else {baseline_tokens as f64 / total_chars as f64});
    println!("dp_min_tokens={} tpc_dp_min={:.6}", dp_tokens, tpc_dp);
    println!("greedy_longest_tokens={} tpc_greedy={:.6}", greedy_tokens, tpc_g);
    println!("merge_tokens={} tpc_merge={:.6}", merge_tokens, tpc_m);

    // 给一个清晰的“谁最小”
    let mut best = ("dp_min", tpc_dp);
    if tpc_g < best.1 { best = ("greedy_longest", tpc_g); }
    if tpc_m < best.1 { best = ("merge", tpc_m); }
    println!("best_method={} best_tpc={:.6}", best.0, best.1);

    Ok(())
}


