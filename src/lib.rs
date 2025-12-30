use anyhow::Result;
use hashbrown::{HashMap, HashSet};
use humantime::format_rfc3339_millis;
use smallvec::{smallvec, SmallVec};
use ahash::RandomState;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hash::Hasher;
use std::fs::{self, File, OpenOptions};
use std::time::{Instant, SystemTime};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, Arc, Mutex, atomic::{AtomicUsize, Ordering as AtomicOrdering}};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

// 默认使用 jemalloc 以降低分配开销（仅非 MSVC 平台）。
// 但在构建 Python 扩展（PyO3）时不要覆盖全局 allocator：
// - 可能触发 glibc 的 “cannot allocate memory in static TLS block”
// - 也避免与 Python 运行时/其它扩展的分配器行为冲突
#[cfg(all(not(target_env = "msvc"), not(feature = "python")))]
use tikv_jemallocator::Jemalloc;
#[cfg(all(not(target_env = "msvc"), not(feature = "python")))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

static LOG_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
static LOG_LEVEL: OnceLock<LogLevel> = OnceLock::new();

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
}

fn parse_log_level(s: &str) -> Option<LogLevel> {
    match s.trim().to_ascii_lowercase().as_str() {
        "error" | "err" | "e" | "0" => Some(LogLevel::Error),
        "warn" | "warning" | "w" | "1" => Some(LogLevel::Warn),
        "info" | "i" | "2" => Some(LogLevel::Info),
        "debug" | "dbg" | "d" | "3" => Some(LogLevel::Debug),
        _ => None,
    }
}

#[inline]
fn current_log_level() -> LogLevel {
    *LOG_LEVEL.get_or_init(|| {
        if let Ok(v) = std::env::var("LOG_LEVEL") {
            parse_log_level(&v).unwrap_or(LogLevel::Info)
        } else if std::env::var("LOG_DEBUG").is_ok() {
            LogLevel::Debug
        } else {
            LogLevel::Info
        }
    })
}

#[inline]
fn log_enabled(level: LogLevel) -> bool {
    level <= current_log_level()
}

fn log_file_path() -> Option<&'static PathBuf> {
    LOG_PATH.get_or_init(|| {
        // 优先环境变量 LOG_FILE，否则默认写当前目录 length_tokenizer.log
        if let Ok(p) = std::env::var("LOG_FILE") {
            Some(PathBuf::from(p))
        } else {
            Some(PathBuf::from("length_tokenizer.log"))
        }
    }).as_ref()
}

#[inline]
fn log_at(level: LogLevel, tag: &str, msg: impl AsRef<str>) {
    if !log_enabled(level) {
        return;
    }
    let ts = format_rfc3339_millis(SystemTime::now());
    let pid = std::process::id();
    let line = format!("[{ts}][pid={pid}][{tag}] {}", msg.as_ref());
    eprintln!("{}", line);
    if let Some(path) = log_file_path() {
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "{}", line);
        }
    }
}

#[inline]
fn log_line(tag: &str, msg: impl AsRef<str>) {
    log_at(LogLevel::Info, tag, msg);
}

#[inline]
fn log_warn(tag: &str, msg: impl AsRef<str>) {
    log_at(LogLevel::Warn, tag, msg);
}

#[inline]
fn log_debug(tag: &str, msg: impl AsRef<str>) {
    log_at(LogLevel::Debug, tag, msg);
}

fn log_resources(tag: &str, start: Option<Instant>) {
    // 资源日志属于调试信息：默认不打，避免读 /proc 与字符串拼接开销。
    if !log_enabled(LogLevel::Debug) {
        return;
    }
    let rss_mb = fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| {
            let mut it = s.split_whitespace();
            let _size = it.next()?;
            let rss_pages = it.next()?.parse::<u64>().ok()?;
            Some(rss_pages * 4096 / (1024 * 1024)) // 假设 4K 页
        });
    let loadavg = fs::read_to_string("/proc/loadavg").ok();
    let elapsed = start.map(|t| t.elapsed().as_secs_f32());
    log_debug(
        "res",
        format!(
            "{} rss_mb={:?} loadavg={:?} elapsed={:?}",
        tag,
        rss_mb,
        loadavg.as_deref().map(str::trim),
        elapsed
        ),
    );
}

// ===== diff 桶文件 / manifest：用于多进程 apply 阶段增量更新 =====

const DIFF_MANIFEST_FILE: &str = "diff_manifest.bin";
const DIFF_MANIFEST_MAGIC: [u8; 4] = *b"LTMF";

#[inline]
fn diff_bits_len(bucket_cnt: usize) -> usize {
    (bucket_cnt + 7) / 8
}

#[inline]
fn diff_bit_set(bits: &mut [u8], idx: usize) {
    bits[idx >> 3] |= 1u8 << (idx & 7);
}

#[inline]
fn diff_bit_get(bits: &[u8], idx: usize) -> bool {
    (bits[idx >> 3] & (1u8 << (idx & 7))) != 0
}

fn diff_bits_to_indices(bits: &[u8], bucket_cnt: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for (byte_i, &b) in bits.iter().enumerate() {
        if b == 0 {
            continue;
        }
        for bit in 0..8usize {
            if (b >> bit) & 1 == 1 {
                let idx = byte_i * 8 + bit;
                if idx < bucket_cnt {
                    out.push(idx);
                }
            }
        }
    }
    out
}

#[inline]
fn diff_manifest_path(dir: &Path) -> PathBuf {
    dir.join(DIFF_MANIFEST_FILE)
}

fn write_diff_manifest(
    dir: &Path,
    bucket_cnt: usize,
    old_bits: &[u8],
    new_bits: &[u8],
) -> Result<()> {
    let bits_len = diff_bits_len(bucket_cnt);
    if old_bits.len() != bits_len || new_bits.len() != bits_len {
        return Err(anyhow::anyhow!(
            "diff manifest bits len mismatch: expect {} got old={} new={}",
            bits_len,
            old_bits.len(),
            new_bits.len()
        ));
    }
    let mut f = BufWriter::new(File::create(diff_manifest_path(dir))?);
    f.write_all(&DIFF_MANIFEST_MAGIC)?;
    f.write_all(&(bucket_cnt as u32).to_le_bytes())?;
    f.write_all(old_bits)?;
    f.write_all(new_bits)?;
    f.flush()?;
    Ok(())
}

fn read_diff_manifest(dir: &Path, bucket_cnt: usize) -> Option<(Vec<u8>, Vec<u8>)> {
    let p = diff_manifest_path(dir);
    let file = File::open(&p).ok()?;
    let mut r = BufReader::new(file);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).ok()?;
    if magic != DIFF_MANIFEST_MAGIC {
        return None;
    }
    let mut cnt_buf = [0u8; 4];
    r.read_exact(&mut cnt_buf).ok()?;
    let cnt = u32::from_le_bytes(cnt_buf) as usize;
    if cnt != bucket_cnt {
        return None;
    }
    let bits_len = diff_bits_len(bucket_cnt);
    let mut old_bits = vec![0u8; bits_len];
    let mut new_bits = vec![0u8; bits_len];
    r.read_exact(&mut old_bits).ok()?;
    r.read_exact(&mut new_bits).ok()?;
    Some((old_bits, new_bits))
}

/// 读取单个桶文件并合并到 `acc`，返回是否读取到任意记录。
/// 文件格式：`u64 record_count` + `record_count` 个 bincode 序列化的 `(Ngram, Stat)`。
fn read_bucket_file_into(
    path: &Path,
    acc: &mut HashMap<Ngram, Stat, RandomState>,
) -> Result<bool> {
    let file = File::open(path)?;
    let mut r = BufReader::new(file);
    let mut cnt_buf = [0u8; 8];
    r.read_exact(&mut cnt_buf)?;
    let cnt = u64::from_le_bytes(cnt_buf);
    let mut any = cnt > 0;
    for _ in 0..cnt {
        match bincode::deserialize_from::<_, (Ngram, Stat)>(&mut r) {
            Ok((ng, st)) => {
                let e: &mut Stat = acc.entry(ng).or_default();
                e.freq += st.freq;
                e.score += st.score;
            }
            Err(_) => {
                // 若出现损坏/截断，停止读取该文件，避免死循环
                any = true;
                break;
            }
        }
    }
    Ok(any)
}

/// 单条合并规则
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeRule {
    pub parts: Vec<String>,
    pub replacement: String,
    #[serde(default)]
    pub freq: u32,
    #[serde(default)]
    pub score: u64,
}

/// 序列统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Stat {
    freq: u32,
    score: u64,
}

#[derive(Clone)]
struct Candidate {
    score: u64,
    len: usize,
    ngram: Ngram,
}

impl Eq for Candidate {}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.len == other.len && self.ngram.as_slice() == other.ngram.as_slice()
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .cmp(&other.score)
            .then(self.len.cmp(&other.len))
            .then(self.ngram.as_slice().cmp(other.ngram.as_slice()))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 配置
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    pub num_merges: usize,
    pub n_values: Vec<usize>,
    pub aim_token_num: usize,
    /// 限制新生成 token 的最大字符长度（0 表示不限制）。
    ///
    /// 目的：避免出现极长、极稀疏的“句子级 token”。这些 token 往往会降低 LM 的可组合性与泛化，
    /// 同时也会让 embedding 更新更稀疏，影响 bpc 收敛。
    pub max_token_chars: usize,
    /// 禁止生成“混合硬标点 + 词元(字母数字)”的新 token（训练阶段过滤候选 merge）。
    ///
    /// 目的：避免出现诸如 `"Ġ,ĠandĠ`、`")ĠandĠ` 这类“标点+词”拼在一起的 token，
    /// 这些 token 可能提高 token-level 的可预测性难度，并损害 bpc。
    pub forbid_punct_word_mix: bool,
    /// 禁止生成“包含硬标点（ASCII 标点，排除 ' 与 -）”的新 token（训练阶段过滤候选 merge）。
    ///
    /// 注意：这不会删除/禁用 **单字符** 的标点 token（否则无法编码标点字符），
    /// 但会阻止把标点与其他字符（含 END_TOKEN）合并成更长 token（例如 ",Ġ", ".Ġ", "\"Ġ" 等）。
    pub forbid_punct_tokens: bool,
    /// 当启用 `forbid_punct_tokens` 时，仍允许生成“空格+标点”的 token，用于吸收 BPE 的优势。
    ///
    /// 注意：本实现的 END_TOKEN 是 **词尾标记**（token 以 'Ġ' 结尾表示后面跟着空白），
    /// 因而 “空格+标点（BPE 里的 `Ġ,` / `Ġ.`）” 在这里对应的是 **“标点 + 'Ġ'”**（例如 `,Ġ`、`.Ġ`、`@-@Ġ`）。
    ///
    /// 允许规则（对候选 merge 生效）：
    /// - 合并后的 token 只包含 **标点/符号**，且 **只包含 1 个 'Ġ' 并且在末尾**。
    pub allow_space_punct_tokens: bool,
    /// 当启用 `forbid_punct_tokens` 时，仍允许生成“常见缩写/点分模式”的 token（例如 `U.S.Ġ`, `e.g.Ġ`）。
    ///
    /// 允许规则（对候选 merge 生效）：
    /// - 合并后的 token **只包含 1 个 'Ġ' 并且在末尾**
    /// - 去掉末尾 'Ġ' 后，只包含 **字母数字** 与 **'.'**
    /// - 且至少包含 1 个 '.' 与 1 个字母数字
    pub allow_abbrev_tokens: bool,
    /// 当启用 `forbid_punct_tokens` 时，仍允许生成“连字符/负号模式”的 token（例如 `well-knownĠ`）。
    ///
    /// 允许规则（对候选 merge 生效）：
    /// - 合并后的 token **只包含 1 个 'Ġ' 并且在末尾**
    /// - 去掉末尾 'Ġ' 后，只包含 **字母数字** 与 **'-'**
    /// - 且至少包含 1 个 '-' 与 1 个字母数字，并且首尾为字母数字（避免 `-foo` / `bar-` 这类噪声）
    pub allow_hyphen_tokens: bool,
    /// 当启用 `forbid_punct_tokens` 时，仍允许生成“词尾标点”token（例如 `word,Ġ`, `word.Ġ`, `word)Ġ`）。
    ///
    /// 允许规则（对候选 merge 生效）：
    /// - token **只包含 1 个 'Ġ' 且在末尾**（单词级）
    /// - 去掉末尾 'Ġ' 后：前缀为字母数字，末尾为 1~3 个“允许的标点符号”
    /// - 标点仅允许出现在末尾（避免把标点塞进词干里）
    pub allow_word_final_punct_tokens: bool,
    /// 当启用 `forbid_punct_tokens` 时，仍允许生成“撇号缩写/所有格”token（例如 `don'tĠ`, `John'sĠ`, `I'mĠ`）。
    ///
    /// 允许规则（对候选 merge 生效）：
    /// - token **只包含 1 个 'Ġ' 且在末尾**（单词级）
    /// - 去掉末尾 'Ġ' 后：只包含字母数字与 `'`（撇号），并且至少包含 1 个字母数字与 1 个 `'`
    pub allow_apostrophe_tokens: bool,
    /// 当启用 `forbid_punct_tokens` 时，仍允许生成“跨词 + 标点混合”的 token（SuperBPE 的核心优势之一）。
    ///
    /// 这里的“跨词”以 token 中包含多个 END_TOKEN('Ġ') 为准（即 token 覆盖多个词尾）。
    /// 为避免失控地生成噪声 token，我们做**强约束**：
    /// - token 必须以 'Ġ' 结尾（完整跨词）
    /// - 以 'Ġ' 为分隔，每个 segment（不含 'Ġ'）必须满足下列之一：
    ///   - 纯字母数字（word）
    ///   - 纯标点（punct-only，长度受限）
    ///   - 缩写点分（abbrev，比如 U.S.）
    ///   - 连字符（hyphen，比如 well-known）
    ///   - 撇号（apostrophe，比如 don't）
    ///   - 词尾标点（word-final punct，比如 time, / word)）
    ///
    /// 目的：吸收 SuperBPE 的低 TPC 优势（例如 `,ĠtheĠ`, `.ĠAtĠtheĠ` 这类模式），同时尽量维持 token 可预测性。
    pub allow_cross_word_punct_word_mix_tokens: bool,
    /// SuperBPE 风格：控制“跨词(superword) token”的生成时机。
    ///
    /// 语义：当 `cross_word_start_vocab>0` 时，只有当当前词表大小（含 special tokens）达到该阈值后，
    /// 才允许生成 **包含 >=2 个 'Ġ'** 的 token（即跨越多个词的短语 token）。
    ///
    /// 直觉：先把预算用在“词内 subword + 高频标点模式”，等词表接近目标规模后再引入 superwords，
    /// 避免过早生成大量长尾短语 token 导致 token 分布变平、模型更难学。
    pub cross_word_start_vocab: usize,
    /// 限制单个 token 内最多包含多少个词（用 'Ġ' 的计数近似，即 token 中 'Ġ' 的个数）。
    ///
    /// - 0 表示不限制
    /// - 1 表示只允许单词级 token（含词尾 'Ġ'）与词内 subword（无 'Ġ'），禁止任何多词短语 token
    /// - 2/3/... 允许最多 2/3/... 词的短语 token
    pub max_token_words: usize,
    /// 禁止生成“跨词但不完整”的 token：即 token **包含 END_TOKEN('Ġ') 但不是以 END_TOKEN 结尾**。
    ///
    /// 直觉：这种 token 会跨越词边界却在末尾切进词内部（例如 `000Ġkg`, `'sĠdeath`），
    /// 往往更难预测，容易伤 bpc。
    pub forbid_incomplete_cross_word: bool,
    pub recompute_each_step: bool, // 测试/调试用，全量重算统计
    /// 是否使用候选堆（BinaryHeap）加速 argmax 选择。
    ///
    /// 重要：heap 会额外复制一份 n-gram key，带来显著的内存开销；
    /// 在大语料/大 n_max 时非常容易触发 OOM。
    /// 因此默认关闭（改为每步扫描 global_stats 选择 best，通常仍然可接受）。
    pub use_heap: bool,
    pub num_workers: usize,        // 多线程/多进程分片数量（阶段性预留）
    pub use_multiprocess: bool,    // 是否启用多进程 map-reduce
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            num_merges: 20000,
            n_values: vec![2, 3, 4, 5, 6],
            aim_token_num: 15_000,
            max_token_chars: 0,
            forbid_punct_word_mix: false,
            forbid_punct_tokens: false,
            allow_space_punct_tokens: false,
            allow_abbrev_tokens: false,
            allow_hyphen_tokens: false,
            allow_word_final_punct_tokens: false,
            allow_apostrophe_tokens: false,
            allow_cross_word_punct_word_mix_tokens: false,
            cross_word_start_vocab: 0,
            max_token_words: 0,
            forbid_incomplete_cross_word: false,
            recompute_each_step: false,
            use_heap: false,
            num_workers: 0,
            use_multiprocess: false,
        }
    }
}

/// 训练后的保存格式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTable {
    pub merges: Vec<MergeRule>,
    pub vocab: HashMap<String, u32>,
}

/// N-gram Key。
///
/// 说明：之前使用 `SmallVec<[u32; 16]>` 会导致每个 key 都内联携带 16 个 `u32`，
/// 即便 n=2 也会占用很大的内存；在大语料 + 大 n\_max（例如 9）时，
/// `global_stats`（以及可选 heap）会出现巨大的内存峰值并触发 OOM。
///
/// 这里把内联容量调整为 9（覆盖常用的 n\_max<=9 场景）；若用户设置 n\_max>9，
/// SmallVec 会自动退化为堆分配（功能正确，但可能更慢）。
type Ngram = SmallVec<[u32; 9]>;

/// 词尾空格占位符，使用单字符与 BPE 风格对齐（原 "</w>"）
const END_TOKEN: &str = "Ġ";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SeqEntry {
    tokens: Vec<u32>,
    freq: u32,
}

/// 字符串驻留池，分配稳定的 token id
#[derive(Debug, Clone)]
struct Interner {
    token_to_id: HashMap<String, u32, RandomState>,
    id_to_token: Vec<String>,
    id_to_chars: Vec<u32>,
    id_to_flags: Vec<u8>,
    id_to_end_count: Vec<u8>,
}

impl Interner {
    fn new() -> Self {
        Self {
            token_to_id: HashMap::with_hasher(RandomState::new()),
            id_to_token: Vec::new(),
            id_to_chars: Vec::new(),
            id_to_flags: Vec::new(),
            id_to_end_count: Vec::new(),
        }
    }

    #[inline]
    fn token_flags(s: &str) -> u8 {
        // Flags used to cheaply filter candidate merges.
        // Goal: detect tokens that contain punctuation/symbols, and tokens that mix
        // "word chars" with punctuation/symbols.
        //
        // - word char: Unicode alnum
        // - punct: any non-alphanumeric char (excluding END_TOKEN 'Ġ')
        //
        // Note: END_TOKEN ('Ġ') is treated as neutral for WORD/PUNCT, but we still record its presence.
        const WORD: u8 = 1;
        const PUNCT: u8 = 2;
        const HAS_END: u8 = 4;
        const END_AT_END: u8 = 8;
        let mut f: u8 = 0;
        let mut has_end = false;
        for ch in s.chars() {
            if ch == 'Ġ' {
                has_end = true;
                continue;
            }
            if ch.is_alphanumeric() {
                f |= WORD;
                continue;
            }
            // Any non-alnum (punctuation/symbol) counts as PUNCT for our merge filters.
            f |= PUNCT;
        }
        if has_end {
            f |= HAS_END;
        }
        if s.ends_with('Ġ') {
            f |= END_AT_END;
        }
        f
    }

    #[inline]
    fn token_end_count(s: &str) -> u8 {
        let mut n: u8 = 0;
        for ch in s.chars() {
            if ch == 'Ġ' {
                n = n.saturating_add(1);
            }
        }
        n
    }

    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.token_to_id.get(s) {
            return id;
        }
        let id = self.id_to_token.len() as u32;
        self.id_to_token.push(s.to_string());
        self.token_to_id.insert(s.to_string(), id);
        self.id_to_chars.push(s.chars().count() as u32);
        self.id_to_flags.push(Self::token_flags(s));
        self.id_to_end_count.push(Self::token_end_count(s));
        id
    }

    fn get(&self, id: u32) -> &str {
        &self.id_to_token[id as usize]
    }

    #[inline]
    fn chars_len(&self, id: u32) -> usize {
        self.id_to_chars[id as usize] as usize
    }

    fn rebuild_char_lens(&mut self) {
        self.id_to_chars = self.id_to_token.iter().map(|s| s.chars().count() as u32).collect();
    }

    fn rebuild_flags(&mut self) {
        self.id_to_flags = self.id_to_token.iter().map(|s| Self::token_flags(s)).collect();
    }

    fn rebuild_end_counts(&mut self) {
        self.id_to_end_count = self.id_to_token.iter().map(|s| Self::token_end_count(s)).collect();
    }
}

#[inline]
fn ngram_char_len(ng: &Ngram, id_to_chars: &[u32]) -> usize {
    let mut sum: usize = 0;
    for &id in ng.iter() {
        sum += id_to_chars[id as usize] as usize;
    }
    sum
}

#[inline]
fn ngram_flags_or(ng: &Ngram, id_to_flags: &[u8]) -> u8 {
    let mut f: u8 = 0;
    for &id in ng.iter() {
        f |= id_to_flags[id as usize];
    }
    f
}

#[inline]
fn ngram_end_count(ng: &Ngram, id_to_end_count: &[u8]) -> u8 {
    let mut sum: u8 = 0;
    for &id in ng.iter() {
        sum = sum.saturating_add(id_to_end_count[id as usize]);
    }
    sum
}

#[inline]
fn ngram_cross_word_blocked(
    ng: &Ngram,
    cfg: &TokenizerConfig,
    id_to_end_count: &[u8],
    vocab_size_with_specials: usize,
) -> bool {
    let words = ngram_end_count(ng, id_to_end_count) as usize;
    if cfg.max_token_words > 0 && words > cfg.max_token_words {
        return true;
    }
    if cfg.cross_word_start_vocab > 0 && words >= 2 && vocab_size_with_specials < cfg.cross_word_start_vocab {
        return true;
    }
    false
}

#[inline]
fn ngram_mixes_punct_and_word(ng: &Ngram, id_to_flags: &[u8]) -> bool {
    // Must match Interner::token_flags bits.
    const WORD: u8 = 1;
    const PUNCT: u8 = 2;
    let f = ngram_flags_or(ng, id_to_flags);
    (f & WORD) != 0 && (f & PUNCT) != 0
}

#[inline]
fn ngram_has_hard_punct(ng: &Ngram, id_to_flags: &[u8]) -> bool {
    // Must match Interner::token_flags bits.
    const PUNCT: u8 = 2;
    let f = ngram_flags_or(ng, id_to_flags);
    (f & PUNCT) != 0
}

#[inline]
fn ngram_has_single_trailing_end(ng: &Ngram, id_to_flags: &[u8], id_to_token: &[String]) -> bool {
    // Exactly one 'Ġ' in the merged token, and it must be the last character.
    //
    // We approximate this cheaply by enforcing:
    // - only the last part may contain END_TOKEN
    // - the last part must end with END_TOKEN and contain it exactly once
    const HAS_END: u8 = 4;
    const END_AT_END: u8 = 8;
    if ng.is_empty() {
        return false;
    }
    let last = *ng.last().unwrap();
    if (id_to_flags[last as usize] & END_AT_END) == 0 {
        return false;
    }
    // no END in prefix parts
    if ng.len() >= 2 {
        for &id in &ng[..ng.len() - 1] {
            if (id_to_flags[id as usize] & HAS_END) != 0 {
                return false;
            }
        }
    }
    let s_last = &id_to_token[last as usize];
    if !s_last.ends_with('Ġ') {
        return false;
    }
    // avoid allowing cross-word tokens like "theĠofĠ"
    s_last.chars().filter(|&c| c == 'Ġ').count() == 1
}

#[inline]
fn ngram_is_allowed_space_punct_token(ng: &Ngram, id_to_flags: &[u8], id_to_token: &[String]) -> bool {
    // Allow punctuation-only tokens that end with a single trailing END_TOKEN (e.g. ",Ġ", ".Ġ", "@-@Ġ").
    //
    // Criteria:
    // - merged token has PUNCT and has NO WORD chars
    // - merged token has exactly one trailing END_TOKEN
    const WORD: u8 = 1;
    const PUNCT: u8 = 2;
    let f = ngram_flags_or(ng, id_to_flags);
    if (f & PUNCT) == 0 {
        return false;
    }
    if (f & WORD) != 0 {
        return false;
    }
    ngram_has_single_trailing_end(ng, id_to_flags, id_to_token)
}

#[inline]
fn ngram_is_allowed_abbrev_token(ng: &Ngram, id_to_flags: &[u8], id_to_token: &[String]) -> bool {
    // Allow abbreviation/dot patterns like "U.S.Ġ", "e.g.Ġ", "etc.Ġ".
    //
    // Restrict to single-word tokens (exactly one trailing END_TOKEN) to avoid merging across words.
    if !ngram_has_single_trailing_end(ng, id_to_flags, id_to_token) {
        return false;
    }
    // Scan all characters excluding the last 'Ġ', without allocating the merged string.
    let mut has_dot = false;
    let mut has_alnum = false;
    let mut len_chars = 0usize;
    for (idx, &id) in ng.iter().enumerate() {
        let s = &id_to_token[id as usize];
        if idx + 1 == ng.len() {
            // last token: strip trailing 'Ġ'
            let core = match s.strip_suffix('Ġ') {
                Some(c) => c,
                None => return false,
            };
            for ch in core.chars() {
                len_chars += 1;
                if len_chars > 24 {
                    return false;
                }
                if ch.is_alphanumeric() {
                    has_alnum = true;
                } else if ch == '.' {
                    has_dot = true;
                } else {
                    return false;
                }
            }
        } else {
            // prefix tokens must not contain 'Ġ' (already ensured by ngram_has_single_trailing_end), but
            // we still enforce allowed char set here.
            for ch in s.chars() {
                if ch == 'Ġ' {
                    return false;
                }
                len_chars += 1;
                if len_chars > 24 {
                    return false;
                }
                if ch.is_alphanumeric() {
                    has_alnum = true;
                } else if ch == '.' {
                    has_dot = true;
                } else {
                    return false;
                }
            }
        }
    }
    has_dot && has_alnum
}

#[inline]
fn ngram_is_allowed_hyphen_token(ng: &Ngram, id_to_flags: &[u8], id_to_token: &[String]) -> bool {
    // Allow hyphenated-word patterns like "well-knownĠ".
    //
    // Restrict to single-word tokens (exactly one trailing END_TOKEN) to avoid merging across words.
    if !ngram_has_single_trailing_end(ng, id_to_flags, id_to_token) {
        return false;
    }
    let mut has_hyphen = false;
    let mut has_alnum = false;
    let mut len_chars = 0usize;
    let mut first: Option<char> = None;
    let mut last: Option<char> = None;
    for (idx, &id) in ng.iter().enumerate() {
        let s = &id_to_token[id as usize];
        if idx + 1 == ng.len() {
            let core = match s.strip_suffix('Ġ') {
                Some(c) => c,
                None => return false,
            };
            for ch in core.chars() {
                if ch == 'Ġ' {
                    return false;
                }
                len_chars += 1;
                if len_chars > 32 {
                    return false;
                }
                if first.is_none() {
                    first = Some(ch);
                }
                last = Some(ch);
                if ch.is_alphanumeric() {
                    has_alnum = true;
                } else if ch == '-' {
                    has_hyphen = true;
                } else {
                    return false;
                }
            }
        } else {
            for ch in s.chars() {
            if ch == 'Ġ' {
                return false;
            }
            len_chars += 1;
            if len_chars > 32 {
                return false;
            }
            if first.is_none() {
                first = Some(ch);
            }
            last = Some(ch);
            if ch.is_alphanumeric() {
                has_alnum = true;
            } else if ch == '-' {
                has_hyphen = true;
            } else {
                return false;
            }
        }
        }
    }
    if !has_hyphen || !has_alnum {
        return false;
    }
    // ensure token does not start/end with '-'
    match (first, last) {
        (Some(f), Some(l)) if f.is_alphanumeric() && l.is_alphanumeric() => true,
        _ => false,
    }
}

#[inline]
fn ngram_is_allowed_apostrophe_token(ng: &Ngram, id_to_flags: &[u8], id_to_token: &[String]) -> bool {
    // Allow single-word tokens containing apostrophe, e.g. "don'tĠ", "John'sĠ", "I'mĠ".
    if !ngram_has_single_trailing_end(ng, id_to_flags, id_to_token) {
        return false;
    }
    let mut has_apos = false;
    let mut has_alnum = false;
    let mut len_chars = 0usize;
    for (idx, &id) in ng.iter().enumerate() {
        let s = &id_to_token[id as usize];
        let core = if idx + 1 == ng.len() {
            match s.strip_suffix('Ġ') {
                Some(c) => c,
                None => return false,
            }
        } else {
            s.as_str()
        };
        for ch in core.chars() {
            if ch == 'Ġ' {
                return false;
            }
            len_chars += 1;
            if len_chars > 32 {
                return false;
            }
            if ch.is_alphanumeric() {
                has_alnum = true;
                continue;
            }
            if ch == '\'' {
                has_apos = true;
                continue;
            }
            return false;
        }
    }
    has_apos && has_alnum
}

#[inline]
fn is_word_final_punct_allowed(ch: char) -> bool {
    matches!(ch, ',' | '.' | '!' | '?' | ':' | ';' | ')' | ']' | '}' | '"' | '\'')
}

#[inline]
fn segment_is_punct_only(seg: &[char]) -> bool {
    if seg.is_empty() {
        return false;
    }
    if seg.len() > 6 {
        // 防止出现超长标点串（通常会让 token 非常稀疏且难学）
        return false;
    }
    seg.iter().all(|c| !c.is_alphanumeric())
}

#[inline]
fn segment_is_alnum_only(seg: &[char]) -> bool {
    !seg.is_empty() && seg.iter().all(|c| c.is_alphanumeric())
}

#[inline]
fn segment_is_abbrev(seg: &[char]) -> bool {
    if seg.is_empty() || seg.len() > 24 {
        return false;
    }
    let mut has_dot = false;
    let mut has_alnum = false;
    for &ch in seg {
        if ch.is_alphanumeric() {
            has_alnum = true;
        } else if ch == '.' {
            has_dot = true;
        } else {
            return false;
        }
    }
    has_dot && has_alnum
}

#[inline]
fn segment_is_hyphen(seg: &[char]) -> bool {
    if seg.is_empty() || seg.len() > 32 {
        return false;
    }
    if !seg[0].is_alphanumeric() || !seg[seg.len() - 1].is_alphanumeric() {
        return false;
    }
    let mut has_hyphen = false;
    let mut has_alnum = false;
    for &ch in seg {
        if ch.is_alphanumeric() {
            has_alnum = true;
        } else if ch == '-' {
            has_hyphen = true;
        } else {
            return false;
        }
    }
    has_hyphen && has_alnum
}

#[inline]
fn segment_is_apostrophe(seg: &[char]) -> bool {
    if seg.is_empty() || seg.len() > 32 {
        return false;
    }
    let mut has_apos = false;
    let mut has_alnum = false;
    for &ch in seg {
        if ch.is_alphanumeric() {
            has_alnum = true;
        } else if ch == '\'' {
            has_apos = true;
        } else {
            return false;
        }
    }
    has_apos && has_alnum
}

#[inline]
fn segment_is_word_final_punct(seg: &[char]) -> bool {
    if seg.is_empty() || seg.len() > 48 {
        return false;
    }
    // count trailing punct
    let mut punct_len = 0usize;
    for &ch in seg.iter().rev() {
        if is_word_final_punct_allowed(ch) {
            punct_len += 1;
            if punct_len > 3 {
                return false;
            }
        } else {
            break;
        }
    }
    if punct_len == 0 {
        return false;
    }
    let word_len = seg.len() - punct_len;
    if word_len == 0 {
        return false;
    }
    seg[..word_len].iter().all(|c| c.is_alphanumeric())
}

#[inline]
fn segment_allowed_for_cross_word_mix(seg: &[char]) -> bool {
    segment_is_alnum_only(seg)
        || segment_is_punct_only(seg)
        || segment_is_abbrev(seg)
        || segment_is_hyphen(seg)
        || segment_is_apostrophe(seg)
        || segment_is_word_final_punct(seg)
}

#[inline]
fn ngram_is_allowed_cross_word_punct_word_mix_token(ng: &Ngram, id_to_token: &[String]) -> bool {
    // Parse merged token by scanning characters and splitting on 'Ġ'.
    // Requirements:
    // - ends with 'Ġ' (so we end on a boundary)
    // - has at least 2 segments (i.e., cross-word)
    // - each segment conforms to allowed patterns
    use smallvec::SmallVec;

    // NOTE: smallvec::Array is only implemented for a bounded set of N; keep this <=32.
    // We still cap segment length at 48 chars; if it exceeds 32, SmallVec will spill to heap.
    let mut seg: SmallVec<[char; 32]> = SmallVec::new();
    let mut seg_cnt: usize = 0;
    for &id in ng.iter() {
        for ch in id_to_token[id as usize].chars() {
            if ch == 'Ġ' {
                if seg.is_empty() {
                    return false;
                }
                if !segment_allowed_for_cross_word_mix(seg.as_slice()) {
                    return false;
                }
                seg_cnt += 1;
                seg.clear();
            } else {
                if seg.len() >= 48 {
                    return false;
                }
                seg.push(ch);
            }
        }
    }
    // must end with 'Ġ'
    if !seg.is_empty() {
        return false;
    }
    seg_cnt >= 2
}

#[inline]
fn ngram_is_allowed_word_final_punct_token(ng: &Ngram, id_to_flags: &[u8], id_to_token: &[String]) -> bool {
    // Allow single-word tokens with trailing punctuation, e.g. "word,Ġ", "word)Ġ", "word.\"Ġ".
    if !ngram_has_single_trailing_end(ng, id_to_flags, id_to_token) {
        return false;
    }
    // Build a small buffer of chars (bounded) to check that punctuation only appears at end.
    let mut chars: Vec<char> = Vec::with_capacity(32);
    for (idx, &id) in ng.iter().enumerate() {
        let s = &id_to_token[id as usize];
        let core = if idx + 1 == ng.len() {
            match s.strip_suffix('Ġ') {
                Some(c) => c,
                None => return false,
            }
        } else {
            s.as_str()
        };
        for ch in core.chars() {
            if ch == 'Ġ' {
                return false;
            }
            if chars.len() >= 32 {
                return false;
            }
            chars.push(ch);
        }
    }
    if chars.is_empty() {
        return false;
    }
    // Find first trailing-punct boundary (scan from end)
    let mut punct_len = 0usize;
    for &ch in chars.iter().rev() {
        if is_word_final_punct_allowed(ch) {
            punct_len += 1;
            if punct_len > 3 {
                return false;
            }
        } else {
            break;
        }
    }
    if punct_len == 0 {
        return false;
    }
    let word_len = chars.len() - punct_len;
    if word_len == 0 {
        return false;
    }
    // Prefix must be alnum-only
    if !chars[..word_len].iter().all(|c| c.is_alphanumeric()) {
        return false;
    }
    true
}

#[inline]
fn ngram_has_forbidden_punct(ng: &Ngram, cfg: &TokenizerConfig, id_to_flags: &[u8], id_to_token: &[String]) -> bool {
    // Training-time filter for punctuation-bearing candidates, with optional allow-list rules.
    if !cfg.forbid_punct_tokens {
        return false;
    }
    if !ngram_has_hard_punct(ng, id_to_flags) {
        return false;
    }
    if cfg.allow_space_punct_tokens && ngram_is_allowed_space_punct_token(ng, id_to_flags, id_to_token) {
        return false;
    }
    if cfg.allow_abbrev_tokens && ngram_is_allowed_abbrev_token(ng, id_to_flags, id_to_token) {
        return false;
    }
    if cfg.allow_hyphen_tokens && ngram_is_allowed_hyphen_token(ng, id_to_flags, id_to_token) {
        return false;
    }
    if cfg.allow_apostrophe_tokens && ngram_is_allowed_apostrophe_token(ng, id_to_flags, id_to_token) {
        return false;
    }
    if cfg.allow_word_final_punct_tokens && ngram_is_allowed_word_final_punct_token(ng, id_to_flags, id_to_token) {
        return false;
    }
    if cfg.allow_cross_word_punct_word_mix_tokens && ngram_is_allowed_cross_word_punct_word_mix_token(ng, id_to_token) {
        return false;
    }
    true
}

#[inline]
fn ngram_is_incomplete_cross_word(ng: &Ngram, id_to_flags: &[u8]) -> bool {
    // Incomplete cross-word token: contains END_TOKEN somewhere, but DOES NOT end with END_TOKEN.
    // This can be detected purely from flags:
    // - n-gram OR-flags contains HAS_END
    // - last token does NOT have END_AT_END
    const HAS_END: u8 = 4;
    const END_AT_END: u8 = 8;
    if ng.is_empty() {
        return false;
    }
    let f = ngram_flags_or(ng, id_to_flags);
    if (f & HAS_END) == 0 {
        return false;
    }
    let last = *ng.last().unwrap();
    (id_to_flags[last as usize] & END_AT_END) == 0
}

// ===== 应用分词：Trie + DP（全局最少 token 数）=====

#[derive(Default)]
struct TrieNode {
    next: HashMap<char, usize, RandomState>,
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
        self.nodes[cur].term_id = Some(id);
    }

    /// DP：返回全局 token 数最少的切分结果（token id 列表）。
    /// - 代价函数：每个 token 代价 = 1
    /// - tie-break：同样 token 数时，优先选择更长的 token（更“max-munch”）
    fn dp_min_ids(&self, s: &[char]) -> Option<Vec<u32>> {
        let n = s.len();
        if n == 0 {
            return Some(Vec::new());
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
                        // tie-break：更长 token 优先
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
        }

        if back[0].is_none() {
            return None;
        }

        let mut out: Vec<u32> = Vec::with_capacity(dp[0] as usize);
        let mut i = 0usize;
        while i < n {
            let Some((j, id)) = back[i] else {
                return None;
            };
            out.push(id);
            i = j;
        }
        Some(out)
    }

    /// DP：最少 token 数 + 兜底 unk（用于 Python 扩展/线上推理）
    ///
    /// - 当某个位置无法匹配任何 token 时，消费 1 个字符并输出 unk_id
    /// - tie-break：同 token 数时优先更长 token（更接近 max-munch）
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

        let mut out: Vec<u32> = Vec::with_capacity(dp[0] as usize);
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

    /// DP：最少 token 数 + 兜底 unk，并支持“某些 token 必须从词首开始”这一约束。
    ///
    /// 约束由 `require_word_start[token_id]` 指定：
    /// - 0：无限制
    /// - 1：仅允许在词首位置使用（i==0 或 s[i-1]==end_char）
    ///
    /// 用途：允许“跨词 token（包含多个 END_TOKEN）”只在词首使用，从而避免它在词内起始导致
    /// “不完整跨词 token”（例如把 'Oxford' 拆成 'fordĠCambridgeĠ'）。
    fn dp_min_ids_allow_unk_require_word_start(
        &self,
        s: &[char],
        unk_id: u32,
        require_word_start: &[u8],
        end_char: char,
    ) -> Vec<u32> {
        let n = s.len();
        if n == 0 {
            return Vec::new();
        }

        let inf = u32::MAX / 8;
        let mut dp: Vec<u32> = vec![inf; n + 1];
        let mut back: Vec<Option<(usize, u32)>> = vec![None; n + 1];
        dp[n] = 0;

        for i in (0..n).rev() {
            let at_word_start = i == 0 || s[i - 1] == end_char;
            let mut node = 0usize;
            for j in i..n {
                let Some(&child) = self.nodes[node].next.get(&s[j]) else {
                    break;
                };
                node = child;
                if let Some(tok_id) = self.nodes[node].term_id {
                    let idx = tok_id as usize;
                    if idx < require_word_start.len() && require_word_start[idx] != 0 && !at_word_start {
                        continue;
                    }
                    let cand = 1u32.saturating_add(dp[j + 1]);
                    let better = if cand < dp[i] {
                        true
                    } else if cand == dp[i] {
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

        let mut out: Vec<u32> = Vec::with_capacity(dp[0] as usize);
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

pub struct LengthTokenizer {
    entries: Vec<SeqEntry>, // 常驻分词后的序列（token id）
    merges: Vec<MergeRule>,
    cfg: TokenizerConfig,
    global_stats: HashMap<Ngram, Stat, RandomState>,
    interner: Interner,
    corpus_chars: u64, // 原始语料总字符数（用于 TPC 统计）
    heap: BinaryHeap<Candidate>, // 最大堆维护候选，避免每步全表扫描
    /// 供“应用分词（DP 全局最少 token）”使用的 Trie（懒初始化）
    token_trie: OnceLock<TokenTrie>,
}

impl LengthTokenizer {
    /// 当前 tokenizer 的词表大小（包含 HF 导出时会加入的 special tokens）。
    ///
    /// 说明：
    /// - 训练过程中 `Interner` 只包含“实际合并产生/语料中出现”的 token（不含 `<unk>` 等 special tokens）
    /// - 但导出 HF tokenizer 目录时会额外注入固定 5 个 special tokens
    /// - 因此 `aim_token_num` 更符合“导出后最终 vocab size”的语义
    fn vocab_size_with_specials(&self) -> usize {
        let mut n = self.interner.id_to_token.len();
        for s in ["<unk>", "<pad>", "<s>", "</s>", "<mask>"] {
            if !self.interner.token_to_id.contains_key(s) {
                n += 1;
            }
        }
        n
    }

    /// 预估窗口总数，用于 HashMap 预分配，减少扩容开销
    #[inline]
    fn estimate_windows(len: usize, n_vals: &[usize]) -> usize {
        let mut total = 0usize;
        for &n in n_vals {
            if len >= n {
                total = total.saturating_add(len - n + 1);
            }
        }
        total
    }

    #[inline]
    fn total_tokens(entries: &[SeqEntry]) -> u64 {
        entries
            .iter()
            .map(|e| (e.tokens.len() as u64) * (e.freq as u64))
            .sum()
    }

    #[inline]
    fn total_chars(corpus: &[String]) -> u64 {
        corpus.iter().map(|s| s.chars().count() as u64).sum()
    }

    /// 归一化文本为“字符序列”（与训练 encode_sentence* 的口径一致）：
    /// - 按空白 split
    /// - 词内按字符展开
    /// - 词尾追加 END_TOKEN（Ġ）
    fn normalize_chars(text: &str) -> Vec<char> {
        let mut out: Vec<char> = Vec::new();
        for word in text.split_whitespace() {
            out.extend(word.chars());
            out.extend(END_TOKEN.chars());
        }
        out
    }

    /// 构建（或获取）token trie，用于 DP 全局最少 token 的应用分词。
    fn token_trie(&self) -> &TokenTrie {
        self.token_trie.get_or_init(|| {
            let mut trie = TokenTrie::new();
            for (id, tok) in self.interner.id_to_token.iter().enumerate() {
                trie.insert(tok, id as u32);
            }
            trie
        })
    }

    /// 应用分词（TPC 最小）：给定“词表=当前 interner 中的所有 token”，使用 DP 找到 token 数最少的切分。
    ///
    /// 返回 token id 序列（更适合用于后续训练数据构造）。
    pub fn tokenize_ids_min_tpc(&self, text: &str) -> Vec<u32> {
        let chars = Self::normalize_chars(text);
        if chars.is_empty() {
            return Vec::new();
        }
        let trie = self.token_trie();
        match trie.dp_min_ids(&chars) {
            Some(ids) => ids,
            None => {
                // 与当前 tokenize 行为一致：遇到未登录字符，直接报错。
                // 这里尽量给出定位信息，便于排查语料/词表不一致。
                panic!("dp tokenize failed: input contains char(s) not in token trie")
            }
        }
    }

    /// 保留原来的“按 merges 顺序应用”的分词（BPE 风格），便于对比与回归。
    pub fn tokenize_bpe(&self, text: &str) -> Vec<String> {
        // 先按基础字符+END_TOKEN 编码为 id
        let mut tokens: Vec<u32> = Vec::new();
        for word in text.split_whitespace() {
            for ch in word.chars() {
                let id = self.interner.token_to_id[&ch.to_string()];
                tokens.push(id);
            }
            tokens.push(self.interner.token_to_id[END_TOKEN]);
        }

        // 按训练顺序依次应用 merges（BPE 风格），保证能产出最长/已学习的 token
        let merges: Vec<(Vec<u32>, u32)> = self
            .merges
            .iter()
            .map(|m| {
                let parts: Vec<u32> = m
                    .parts
                    .iter()
                    .map(|p| self.interner.token_to_id[p])
                    .collect();
                let rep = self.interner.token_to_id[&m.replacement];
                (parts, rep)
            })
            .collect();

        for (parts, replacement) in &merges {
            let plen = parts.len();
            let mut i = 0;
            while i + plen <= tokens.len() {
                if tokens[i..i + plen] == parts[..] {
                    tokens.splice(i..i + plen, [*replacement].into_iter());
                    // 往前回退一格，允许新形成的上下文被后续 merge 捕获
                    i = i.saturating_sub(plen.saturating_sub(1));
                } else {
                    i += 1;
                }
            }
        }

        tokens
            .into_iter()
            .map(|id| self.interner.get(id).to_string())
            .collect()
    }

    /// 重建候选堆（仅保留 freq>1 的项）
    fn rebuild_heap(&mut self) {
        self.heap.clear();
        let max_chars = self.cfg.max_token_chars;
        let id_to_chars = &self.interner.id_to_chars;
        let id_to_flags = &self.interner.id_to_flags;
        let id_to_token = &self.interner.id_to_token;
        let id_to_end_count = &self.interner.id_to_end_count;
        let vocab_sz = self.vocab_size_with_specials();
        let forbid_mix = self.cfg.forbid_punct_word_mix;
        let forbid_punct = self.cfg.forbid_punct_tokens;
        let forbid_incomplete = self.cfg.forbid_incomplete_cross_word;
        self.heap.extend(
            self.global_stats
                .iter()
                .filter(|(_, st)| st.freq > 1)
                .filter(|(ng, _)| max_chars == 0 || ngram_char_len(ng, id_to_chars) <= max_chars)
                .filter(|(ng, _)| !forbid_mix || !ngram_mixes_punct_and_word(ng, id_to_flags))
                .filter(|(ng, _)| !forbid_punct || !ngram_has_forbidden_punct(ng, &self.cfg, id_to_flags, id_to_token))
                .filter(|(ng, _)| !ngram_cross_word_blocked(ng, &self.cfg, id_to_end_count, vocab_sz))
                .filter(|(ng, _)| !forbid_incomplete || !ngram_is_incomplete_cross_word(ng, id_to_flags))
                .map(|(ng, st)| Candidate {
                    score: st.score,
                    len: ng.len(),
                    ngram: ng.clone(),
                }),
        );
    }

    /// 将某个 n-gram 更新入堆（若 freq>1）
    fn push_candidate(&mut self, ng: &Ngram) {
        let max_chars = self.cfg.max_token_chars;
        if max_chars > 0 && ngram_char_len(ng, &self.interner.id_to_chars) > max_chars {
            return;
        }
        if self.cfg.forbid_punct_word_mix && ngram_mixes_punct_and_word(ng, &self.interner.id_to_flags) {
            return;
        }
        if self.cfg.forbid_punct_tokens
            && ngram_has_forbidden_punct(ng, &self.cfg, &self.interner.id_to_flags, &self.interner.id_to_token)
        {
            return;
        }
        let vocab_sz = self.vocab_size_with_specials();
        if ngram_cross_word_blocked(ng, &self.cfg, &self.interner.id_to_end_count, vocab_sz) {
            return;
        }
        if self.cfg.forbid_incomplete_cross_word && ngram_is_incomplete_cross_word(ng, &self.interner.id_to_flags) {
            return;
        }
        if let Some(st) = self.global_stats.get(ng) {
            if st.freq > 1 {
                self.heap.push(Candidate {
                    score: st.score,
                    len: ng.len(),
                    ngram: ng.clone(),
                });
            }
        }
    }

    /// 从堆中弹出当前有效的全局最优（懒惰删除过期项）
    fn pop_best_from_heap(&mut self) -> Option<(Ngram, Stat)> {
        let max_chars = self.cfg.max_token_chars;
        let id_to_chars = &self.interner.id_to_chars;
        let id_to_flags = &self.interner.id_to_flags;
        let id_to_token = &self.interner.id_to_token;
        let id_to_end_count = &self.interner.id_to_end_count;
        let vocab_sz = self.vocab_size_with_specials();
        let forbid_mix = self.cfg.forbid_punct_word_mix;
        let forbid_punct = self.cfg.forbid_punct_tokens;
        let forbid_incomplete = self.cfg.forbid_incomplete_cross_word;
        while let Some(cand) = self.heap.pop() {
            if let Some(st) = self.global_stats.get(&cand.ngram) {
                if st.freq <= 1 {
                    continue;
                }
                if st.score != cand.score || cand.len != cand.ngram.len() {
                    // 分值已变化，推入新值后继续
                    self.heap.push(Candidate {
                        score: st.score,
                        len: cand.ngram.len(),
                        ngram: cand.ngram,
                    });
                    continue;
                }
                if max_chars > 0 && ngram_char_len(&cand.ngram, id_to_chars) > max_chars {
                    continue;
                }
                if forbid_mix && ngram_mixes_punct_and_word(&cand.ngram, id_to_flags) {
                    continue;
                }
                if forbid_punct && ngram_has_forbidden_punct(&cand.ngram, &self.cfg, id_to_flags, id_to_token) {
                    continue;
                }
                if ngram_cross_word_blocked(&cand.ngram, &self.cfg, id_to_end_count, vocab_sz) {
                    continue;
                }
                if forbid_incomplete && ngram_is_incomplete_cross_word(&cand.ngram, id_to_flags) {
                    continue;
                }
                return Some((cand.ngram, st.clone()));
            }
        }
        None
    }

    /// 查看当前有效的全局最优分数（不弹出有效项；会清理堆顶过期项）
    fn peek_best_score(&mut self) -> Option<u64> {
        let vocab_sz = self.vocab_size_with_specials();
        loop {
            let cand = self.heap.peek()?.clone();
            match self.global_stats.get(&cand.ngram) {
                Some(st) if st.freq > 1 && st.score == cand.score && cand.len == cand.ngram.len() => {
                    if self.cfg.forbid_punct_word_mix
                        && ngram_mixes_punct_and_word(&cand.ngram, &self.interner.id_to_flags)
                    {
                        let _ = self.heap.pop();
                        continue;
                    }
                    if self.cfg.forbid_punct_tokens
                        && ngram_has_forbidden_punct(
                            &cand.ngram,
                            &self.cfg,
                            &self.interner.id_to_flags,
                            &self.interner.id_to_token,
                        )
                    {
                        let _ = self.heap.pop();
                        continue;
                    }
                    if ngram_cross_word_blocked(&cand.ngram, &self.cfg, &self.interner.id_to_end_count, vocab_sz) {
                        let _ = self.heap.pop();
                        continue;
                    }
                    if self.cfg.forbid_incomplete_cross_word
                        && ngram_is_incomplete_cross_word(&cand.ngram, &self.interner.id_to_flags)
                    {
                        let _ = self.heap.pop();
                        continue;
                    }
                    return Some(st.score);
                }
                Some(st) if st.freq > 1 => {
                    // 分值已变化：弹出旧的，推入新值
                    let _ = self.heap.pop();
                    self.heap.push(Candidate {
                        score: st.score,
                        len: cand.ngram.len(),
                        ngram: cand.ngram,
                    });
                }
                _ => {
                    let _ = self.heap.pop();
                }
            }
        }
    }

    /// 按需构建 rayon 线程池执行闭包，支持 num_workers
    fn with_pool<T: Send>(&self, f: impl FnOnce() -> T + Send) -> T {
        if self.cfg.num_workers > 0 {
            ThreadPoolBuilder::new()
                .num_threads(self.cfg.num_workers)
                .build()
                .expect("build pool")
                .install(f)
        } else {
            f()
        }
    }

    pub fn new(corpus: &[String], cfg: TokenizerConfig) -> Result<Self> {
        let corpus_chars = Self::total_chars(corpus);
        let mut tk = Self {
            entries: Vec::new(),
            merges: Vec::new(),
            cfg,
            global_stats: HashMap::with_hasher(RandomState::new()),
            interner: Interner::new(),
            corpus_chars,
            heap: BinaryHeap::new(),
            token_trie: OnceLock::new(),
        };
        tk.build_vocab(corpus)?;
        if tk.cfg.use_multiprocess {
            log_line("init", "multiprocess path: skip single-thread rebuild_global_stats");
            tk.train()?;
        } else {
            tk.rebuild_global_stats();
            tk.train()?;
        }
        Ok(tk)
    }

    // 调试用：全量重算统计（与原 Python 行为一致）
    fn compute_stats_full(&self) -> HashMap<Ngram, Stat, RandomState> {
        let n_vals = self.cfg.n_values.clone();
        let entries = &self.entries;
        self.with_pool(|| {
            entries
                .par_iter()
                .map(|entry| {
                    let mut local: HashMap<Ngram, Stat, RandomState> =
                        HashMap::with_capacity_and_hasher(
                            Self::estimate_windows(entry.tokens.len(), &n_vals),
                            RandomState::new(),
                        );
                    let tokens = &entry.tokens;
                    let freq = entry.freq;
                    for &n in &n_vals {
                        if tokens.len() < n {
                            continue;
                        }
                        for i in 0..=tokens.len() - n {
                            let mut ng = SmallVec::<[u32; 9]>::with_capacity(n);
                            ng.extend_from_slice(&tokens[i..i + n]);
                            let e = local.entry(ng).or_default();
                            e.freq += freq;
                            e.score += (n.saturating_sub(1) as u64) * freq as u64;
                        }
                    }
                    local
                })
                .reduce(
                    || HashMap::with_hasher(RandomState::new()),
                    |mut acc, m| {
                        for (k, v) in m {
                            let e = acc.entry(k).or_default();
                            e.freq += v.freq;
                            e.score += v.score;
                        }
                        acc
                    },
                )
        })
    }

    #[allow(dead_code)]
    fn compute_stats_threaded(&self, workers: usize) -> HashMap<Ngram, Stat, RandomState> {
        // 简化：直接使用 rayon 的线程池指定线程数，以保持稳定性
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers.max(1))
            .build()
            .expect("build pool");
        pool.install(|| self.compute_stats_full())
    }

    /// 读取文件语料（每行一句）
    pub fn load_corpus<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
        let f = File::open(path)?;
        let reader = BufReader::new(f);
        let mut corpus = Vec::new();
        for line in reader.lines() {
            let l = line?;
            if !l.trim().is_empty() {
                corpus.push(l);
            }
        }
        Ok(corpus)
    }

    fn build_vocab(&mut self, corpus: &[String]) -> Result<()> {
        if self.cfg.use_multiprocess {
            self.build_vocab_multiprocess(corpus)
        } else {
            self.build_vocab_threaded(corpus);
            Ok(())
        }
    }

    /// 单机多线程版本（Rayon），用于默认路径
    fn build_vocab_threaded(&mut self, corpus: &[String]) {
        // 第一阶段：并行对语料分片，构建局部词表（键仍为 Vec<String>），避免串行扫描瓶颈。
        let workers = if self.cfg.num_workers == 0 {
            num_cpus::get().max(1)
        } else {
            self.cfg.num_workers
        };

        let tmp: HashMap<Vec<String>, u32, RandomState> = ThreadPoolBuilder::new()
            .num_threads(workers.max(1))
            .build()
            .expect("build rayon pool for vocab")
            .install(|| {
                corpus
                    .par_iter()
                    .map(|sentence| {
                        let mut local: HashMap<Vec<String>, u32, RandomState> =
                            HashMap::with_hasher(RandomState::new());
                        let encoded = Self::encode_sentence_str(sentence);
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
            });

        // 第二阶段：单线程将字符串 token 映射为全局 u32 token id，生成常驻 entries。
        self.entries = tmp
            .into_iter()
            .map(|(k, f)| {
                let ids: Vec<u32> = k.iter().map(|tok| self.interner.intern(tok)).collect();
                SeqEntry { tokens: ids, freq: f }
            })
            .collect();
    }

    /// 多进程预处理：将语料分片发送到子进程统计局部词频，再归并到主进程。
    fn build_vocab_multiprocess(&mut self, corpus: &[String]) -> Result<()> {
        let workers = if self.cfg.num_workers == 0 {
            num_cpus::get().max(1)
        } else {
            self.cfg.num_workers
        };
        if corpus.is_empty() {
            self.entries.clear();
            return Ok(());
        }

        let t0 = std::time::Instant::now();
        let chunk_size = ((corpus.len() + workers.saturating_sub(1)) / workers.max(1)).clamp(4_000, 20_000);
        let exe = std::env::current_exe()?;
        log_line(
            "build_vocab_mp",
            format!(
                "start corpus_lines={} workers={} chunk_size={}",
                corpus.len(),
                workers,
                chunk_size.max(1)
            ),
        );

        // 分桶归并：将 key 哈希到固定桶中累加，降低单表内存压力。桶数可配置，至少 64。
        let mut buckets = (workers.max(1)).next_power_of_two().max(256);
        if let Ok(v) = std::env::var("MP_BUCKETS") {
            if let Ok(n) = v.parse::<usize>() {
                buckets = n.max(64);
            }
        }
        // 复用 hasher：避免每条 key 都 RandomState::new()
        let bucket_hasher = RandomState::new();

        let chunk_count = (corpus.len() + chunk_size.max(1) - 1) / chunk_size.max(1);
        let t_bucket = std::time::Instant::now();
        let merged_buckets: Vec<HashMap<Vec<String>, u32, RandomState>> = ThreadPoolBuilder::new()
            .num_threads(workers.max(1))
            .build()
            .expect("build rayon pool for preprocess map")
            .install(|| {
                corpus
                    .chunks(chunk_size.max(1))
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .map(|chunk| preprocess_chunk_in_worker(&exe, chunk))
                    .try_fold(
                        || (0..buckets).map(|_| HashMap::with_hasher(RandomState::new())).collect::<Vec<_>>(),
                        |mut acc, part| -> Result<_, anyhow::Error> {
                            let map = part?;
            for (k, v) in map {
                use std::hash::{BuildHasher, Hash};
                                let mut h = bucket_hasher.build_hasher();
                k.hash(&mut h);
                let b = (h.finish() as usize) & (buckets - 1);
                                *acc[b].entry(k).or_insert(0) += v;
            }
                            Ok(acc)
                        },
                    )
                    .try_reduce(
                        || (0..buckets).map(|_| HashMap::with_hasher(RandomState::new())).collect::<Vec<_>>(),
                        |mut acc, mut shard| -> Result<_, anyhow::Error> {
                            for (b, mut m) in shard.drain(..).enumerate() {
                                let acc_b = &mut acc[b];
                                for (k, v) in m.drain() {
                                    *acc_b.entry(k).or_insert(0) += v;
                                }
                            }
                            Ok(acc)
                        },
                    )
            })?;
        log_line(
            "build_vocab_mp",
            format!(
                "bucket fill done partial_maps={} buckets={} elapsed={:.2}s",
                chunk_count,
                buckets,
                t_bucket.elapsed().as_secs_f32()
            ),
        );

        // 汇总所有桶（仍并行，但桶数固定）
        let t_bucket_reduce = std::time::Instant::now();
        let merged: HashMap<Vec<String>, u32, RandomState> = merged_buckets
            .into_par_iter()
            .reduce(
                || HashMap::with_hasher(RandomState::new()),
                |mut acc, m| {
                    for (k, v) in m {
                        *acc.entry(k).or_insert(0) += v;
                    }
                    acc
                },
            );
        log_line(
            "build_vocab_mp",
            format!(
                "bucket reduce done unique_seqs={} elapsed={:.2}s",
                merged.len(),
                t_bucket_reduce.elapsed().as_secs_f32()
            ),
        );

        // 并行提取全部唯一 token，批量分配全局 id，避免单线程 interner 热点。
        log_line(
            "build_vocab_mp",
            format!(
                "token spill start unique_seqs={} elapsed_total={:.2}s",
                merged.len(),
                t0.elapsed().as_secs_f32()
            ),
        );
        // 磁盘分桶，分批去重，降低内存峰值
        let token_bucket_cnt = 2048usize;
        let tmp_dir = std::env::temp_dir().join(format!(
            "lt_tok_buckets_{}_{}",
            std::process::id(),
            t0.elapsed().as_secs_f32().to_bits()
        ));
        // 尝试清理可能存在的同名目录，避免脏数据
        let _ = fs::remove_dir_all(&tmp_dir);
        fs::create_dir_all(&tmp_dir)?;
        let mut bucket_paths = Vec::with_capacity(token_bucket_cnt);
        let mut bucket_writers = Vec::with_capacity(token_bucket_cnt);
        for i in 0..token_bucket_cnt {
            let path = tmp_dir.join(format!("bucket_{i}.txt"));
            bucket_paths.push(path.clone());
            bucket_writers.push(BufWriter::new(File::create(path)?));
        }

        // 将 token 直接写入桶文件（单次遍历 merged.keys）
        let token_hasher = RandomState::new();
        let mut tok_count: usize = 0;
        for seq in merged.keys() {
            for tok in seq {
                use std::hash::{BuildHasher, Hash};
                let mut h = token_hasher.build_hasher();
                tok.hash(&mut h);
                let idx = (h.finish() as usize) & (token_bucket_cnt - 1);
                let w = &mut bucket_writers[idx];
                w.write_all(tok.as_bytes())?;
                w.write_all(b"\n")?;
                tok_count += 1;
                if tok_count % 1_000_000 == 0 {
                    log_debug(
                        "build_vocab_mp",
                        format!(
                            "token spill progress tokens={} elapsed_total={:.2}s",
                            tok_count,
                            t0.elapsed().as_secs_f32()
                        ),
                    );
                }
            }
        }
        for mut w in bucket_writers {
            w.flush()?;
            if let Some(file) = w.get_ref().try_clone().ok() {
                let _ = file.sync_all();
            }
        }
        // 同步目录
        if let Ok(dir_file) = OpenOptions::new().read(true).open(&tmp_dir) {
            let _ = dir_file.sync_all();
        }
        log_line(
            "build_vocab_mp",
            format!(
                "token spill done tokens={} buckets={} elapsed_total={:.2}s",
                tok_count,
                token_bucket_cnt,
                t0.elapsed().as_secs_f32()
            ),
        );
        log_resources("after_token_spill", Some(t0));

        // 逐桶去重（并行），再汇总
        let t_tok_bucket = std::time::Instant::now();
        let token_sets: Vec<HashSet<String, RandomState>> = bucket_paths
            .par_iter()
            .map(|path| -> Result<HashSet<String, RandomState>> {
                if let Ok(meta) = fs::metadata(path) {
                    if meta.len() == 0 {
                        log_debug(
                            "build_vocab_mp",
                            format!(
                                "warn: empty bucket file {:?} (elapsed_total={:.2}s)",
                                path,
                                t0.elapsed().as_secs_f32()
                            ),
                        );
                    }
                }
                let file = File::open(path)?;
                let reader = BufReader::new(file);
                let mut set: HashSet<String, RandomState> = HashSet::with_hasher(RandomState::new());
                for line in reader.lines() {
                    let l = line?;
                    set.insert(l);
                }
                Ok(set)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut token_vec: Vec<String> = Vec::new();
        for set in token_sets {
            token_vec.extend(set.into_iter());
        }
        // 清理临时目录
        let _ = fs::remove_dir_all(&tmp_dir);

        log_line(
            "build_vocab_mp",
            format!(
                "token buckets dedup done unique_tokens={} buckets={} elapsed={:.2}s",
                token_vec.len(),
                token_bucket_cnt,
                t_tok_bucket.elapsed().as_secs_f32()
            ),
        );
        log_resources("after_token_dedup", Some(t0));

        // 确定性分配 id：按字典序排序后枚举
        let mut tokens_vec: Vec<String> = token_vec;
        let t_sort = std::time::Instant::now();
        tokens_vec.par_sort_unstable();
        log_line(
            "build_vocab_mp",
            format!(
                "token sort done token_set={} elapsed={:.2}s",
                tokens_vec.len(),
                t_sort.elapsed().as_secs_f32()
            ),
        );

        // 并行分配 id
        let t_id = std::time::Instant::now();
        let token_to_id_vec: Vec<(String, u32)> = tokens_vec
            .par_iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        let mut token_to_id: HashMap<String, u32, RandomState> = HashMap::with_hasher(RandomState::new());
        for (t, id) in token_to_id_vec {
            token_to_id.insert(t, id);
        }
        self.interner.id_to_token = tokens_vec;
        self.interner.token_to_id = token_to_id;
        self.interner.rebuild_char_lens();
        self.interner.rebuild_flags();
        self.interner.rebuild_end_counts();
        log_line(
            "build_vocab_mp",
            format!(
                "token id assign done token_set={} elapsed={:.2}s elapsed_total={:.2}s",
                self.interner.id_to_token.len(),
                t_id.elapsed().as_secs_f32(),
                t0.elapsed().as_secs_f32()
            ),
        );

        // 并行将序列字符串映射为 id
        let merged_vec: Vec<(Vec<String>, u32)> = merged.into_iter().collect();
        let t_map = std::time::Instant::now();
        self.entries = merged_vec
            .into_par_iter()
            .map(|(seq, freq)| {
                let tokens: Vec<u32> = seq
                    .iter()
                    .map(|tok| *self.interner.token_to_id.get(tok).expect("token id"))
                    .collect();
                SeqEntry { tokens, freq }
            })
            .collect();
        log_line(
            "build_vocab_mp",
            format!(
                "entries mapped entries={} elapsed={:.2}s elapsed_total={:.2}s",
                self.entries.len(),
                t_map.elapsed().as_secs_f32(),
                t0.elapsed().as_secs_f32()
            ),
        );

        Ok(())
    }

    fn encode_sentence(&mut self, sentence: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        for word in sentence.split_whitespace() {
            for ch in word.chars() {
                let id = self.interner.intern(&ch.to_string());
                tokens.push(id);
            }
            let w_id = self.interner.intern(END_TOKEN);
            tokens.push(w_id);
        }
        tokens
    }

    /// 将句子编码为字符串 token 序列（用于并行预处理阶段）
    fn encode_sentence_str(sentence: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        for word in sentence.split_whitespace() {
            for ch in word.chars() {
                tokens.push(ch.to_string());
            }
            tokens.push(END_TOKEN.to_string());
        }
        tokens
    }

    fn train(&mut self) -> Result<()> {
        if self.cfg.use_multiprocess {
            self.train_multiprocess()?;
            return Ok(());
        }
        self.train_single();
        Ok(())
    }

    fn train_single(&mut self) {
        use std::time::Instant;
        let start_all = Instant::now();
        log_line(
            "train_single",
            format!(
                "start merges={} n_values={:?} recompute_each_step={} use_heap={} num_workers={}",
                self.cfg.num_merges,
                self.cfg.n_values,
                self.cfg.recompute_each_step,
                self.cfg.use_heap,
                self.cfg.num_workers
            ),
        );
        let verify_stats = std::env::var("VERIFY_STATS").is_ok();
        let verify_pre = std::env::var("VERIFY_PRE").is_ok();
        if !self.cfg.recompute_each_step && self.cfg.use_heap {
            self.rebuild_heap();
        } else {
            self.heap.clear();
        }
        for step in 0..self.cfg.num_merges {
            let t0 = Instant::now();
            let mut stats_ref: HashMap<Ngram, Stat, RandomState> = HashMap::with_hasher(RandomState::new());
            let debug_pair: Option<(u32, u32)> = std::env::var("DEBUG_PAIR")
                .ok()
                .and_then(|s| {
                    let mut it = s.split(',');
                    let a = it.next()?;
                    let b = it.next()?;
                    Some((self.interner.intern(a), self.interner.intern(b)))
                });
            if let Some((a, b)) = debug_pair {
                log_debug(
                    "train_single",
                    format!("[debug-pair-ids] step {} ids=({}, {})", step, a, b),
                );
            }
            if self.cfg.recompute_each_step {
                stats_ref = self.compute_stats_full();
                if stats_ref.is_empty() {
                    log_line(
                        "train_single",
                        format!(
                            "step {} stats empty, stop (elapsed_total={:.2}s)",
                            step,
                            start_all.elapsed().as_secs_f32()
                        ),
                    );
                    break;
                }
            } else {
                if verify_pre {
                    let full = self.compute_stats_full();
                    let mut diff = 0usize;
                    let mut missing = 0usize;
                    for (k, v) in &full {
                        match self.global_stats.get(k) {
                            Some(g) if g.freq == v.freq && g.score == v.score => {}
                            Some(_) => diff += 1,
                            None => missing += 1,
                        }
                    }
                    let mut extra = 0usize;
                    for k in self.global_stats.keys() {
                        if !full.contains_key(k) {
                            extra += 1;
                        }
                    }
                    if diff + missing + extra > 0 {
                        log_debug(
                            "train_single",
                            format!(
                                "[verify-pre] step {} diff={}, missing={}, extra={}",
                                step, diff, missing, extra
                            ),
                        );
                    }
                }
                if self.global_stats.is_empty() {
                    log_line(
                        "train_single",
                        format!(
                            "step {} stats empty, stop (elapsed_total={:.2}s)",
                            step,
                            start_all.elapsed().as_secs_f32()
                        ),
                    );
                    break;
                }
            }
            let mut best: Option<(Ngram, Stat)> = None;
            if self.cfg.recompute_each_step {
                let stats_view = &stats_ref;
                // 扫描方式（全量重算模式）
                for (ng, st) in stats_view {
                    if st.freq <= 1 {
                        continue;
                    }
                    if self.cfg.max_token_chars > 0
                        && ngram_char_len(ng, &self.interner.id_to_chars) > self.cfg.max_token_chars
                    {
                        continue;
                    }
                    if self.cfg.forbid_punct_word_mix
                        && ngram_mixes_punct_and_word(ng, &self.interner.id_to_flags)
                    {
                        continue;
                    }
                    if self.cfg.forbid_punct_tokens
                        && ngram_has_forbidden_punct(
                            ng,
                            &self.cfg,
                            &self.interner.id_to_flags,
                            &self.interner.id_to_token,
                        )
                    {
                        continue;
                    }
                    if ngram_cross_word_blocked(
                        ng,
                        &self.cfg,
                        &self.interner.id_to_end_count,
                        self.vocab_size_with_specials(),
                    ) {
                        continue;
                    }
                    if self.cfg.forbid_incomplete_cross_word
                        && ngram_is_incomplete_cross_word(ng, &self.interner.id_to_flags)
                    {
                        continue;
                    }
                    // 调试：关注 wh / gh 频次
                    #[cfg(debug_assertions)]
                    {
                        if ng.len() == 2 && ng[0] == self.interner.token_to_id.get("w").copied().unwrap_or(u32::MAX)
                            && ng[1] == self.interner.token_to_id.get("h").copied().unwrap_or(u32::MAX)
                        {
                            log_debug(
                                "train_single",
                                format!("[pair wh] step {} freq={} score={}", step, st.freq, st.score),
                            );
                        }
                        if ng.len() == 2 && ng[0] == self.interner.token_to_id.get("g").copied().unwrap_or(u32::MAX)
                            && ng[1] == self.interner.token_to_id.get("h").copied().unwrap_or(u32::MAX)
                        {
                            log_debug(
                                "train_single",
                                format!("[pair gh] step {} freq={} score={}", step, st.freq, st.score),
                            );
                        }
                    }
                    if let Some((a, b)) = debug_pair {
                        if ng.len() == 2 && ng[0] == a && ng[1] == b {
                            log_debug(
                                "train_single",
                                format!(
                                    "[debug-pair] step {} pair {:?} freq={} score={}",
                                    step, ng, st.freq, st.score
                                ),
                            );
                        }
                    }
                    match &best {
                        Some((b_ng, b_st)) => {
                            if st.score > b_st.score
                                || (st.score == b_st.score && ng.len() > b_ng.len())
                                || (st.score == b_st.score
                                    && ng.len() == b_ng.len()
                                    && ng.as_slice() > b_ng.as_slice())
                            {
                                best = Some((ng.clone(), st.clone()));
                            }
                        }
                        None => {
                            best = Some((ng.clone(), st.clone()));
                        }
                    }
                }
            } else {
                if self.cfg.use_heap {
                    // 使用堆，避免全表扫描（但内存开销大）
                best = self.pop_best_from_heap();
                } else {
                    // 低内存模式：扫描 global_stats 选择 best（精确 argmax；通常瓶颈在 apply）
                    let stats_view = &self.global_stats;
                    let max_chars = self.cfg.max_token_chars;
                    let id_to_chars = &self.interner.id_to_chars;
                    let id_to_flags = &self.interner.id_to_flags;
                    let id_to_token = &self.interner.id_to_token;
                    let id_to_end_count = &self.interner.id_to_end_count;
                    let forbid_mix = self.cfg.forbid_punct_word_mix;
                    let forbid_punct = self.cfg.forbid_punct_tokens;
                    let forbid_incomplete = self.cfg.forbid_incomplete_cross_word;
                    let cfg = &self.cfg;
                    let vocab_sz = self.vocab_size_with_specials();
                    fn better_ref<'a>(
                        a: (&'a Ngram, &'a Stat),
                        b: (&'a Ngram, &'a Stat),
                    ) -> (&'a Ngram, &'a Stat) {
                        let (ang, ast) = a;
                        let (bng, bst) = b;
                        if bst.score > ast.score
                            || (bst.score == ast.score && bng.len() > ang.len())
                            || (bst.score == ast.score
                                && bng.len() == ang.len()
                                && bng.as_slice() > ang.as_slice())
                        {
                            (bng, bst)
                        } else {
                            (ang, ast)
                        }
                    }
                    best = stats_view
                        .par_iter()
                        .filter(|(ng, st)| {
                            st.freq > 1
                                && (max_chars == 0 || ngram_char_len(ng, id_to_chars) <= max_chars)
                                && (!forbid_mix || !ngram_mixes_punct_and_word(ng, id_to_flags))
                                && (!forbid_punct || !ngram_has_forbidden_punct(ng, cfg, id_to_flags, id_to_token))
                                && !ngram_cross_word_blocked(ng, cfg, id_to_end_count, vocab_sz)
                                && (!forbid_incomplete || !ngram_is_incomplete_cross_word(ng, id_to_flags))
                        })
                        .reduce_with(better_ref)
                        .map(|(ng, st)| (ng.clone(), st.clone()));
                }
            }
            let Some((best_parts, best_stat)) = best else { break };

            if let Some((a, b)) = debug_pair {
                let key = smallvec![a, b];
                let map = if self.cfg.recompute_each_step {
                    &stats_ref
                } else {
                    &self.global_stats
                };
                if let Some(st) = map.get(&key) {
                    log_debug(
                        "train_single",
                        format!(
                            "[debug-pair-lookup] step {} freq={} score={}",
                            step, st.freq, st.score
                        ),
                    );
                } else {
                    log_debug(
                        "train_single",
                        format!("[debug-pair-lookup] step {} not found", step),
                    );
                }
            }

            let elapsed_stats = t0.elapsed().as_secs_f32();
            let t_apply = Instant::now();
            self.apply_merge(&best_parts, best_stat.clone());
            let elapsed_apply = t_apply.elapsed().as_secs_f32();

            log_line(
                "merge",
                format!(
                    "{:>6}: freq={} score={} stats={:.3}s apply={:.3}s total={:.1}s [single]",
                    step,
                    best_stat.freq,
                    best_stat.score,
                    elapsed_stats,
                    elapsed_apply,
                    start_all.elapsed().as_secs_f32()
                ),
            );

            // 目标词表大小：达到后提前停止（避免必须手动精确设置 num_merges）
            if self.cfg.aim_token_num > 0 {
                let vocab_sz = self.vocab_size_with_specials();
                if vocab_sz >= self.cfg.aim_token_num {
                    log_line(
                        "train_single",
                        format!(
                            "reach aim_token_num={} (vocab_size={}), stop at step {} (elapsed_total={:.2}s)",
                            self.cfg.aim_token_num,
                            vocab_sz,
                            step + 1,
                            start_all.elapsed().as_secs_f32()
                        ),
                    );
                    break;
                }
            }

            // 可选：验证增量统计的正确性（开启环境变量 VERIFY_STATS）
            if verify_stats && !self.cfg.recompute_each_step {
                let full = self.compute_stats_full();
                let mut diff = 0usize;
                let mut missing = 0usize;
                for (k, v) in &full {
                    match self.global_stats.get(k) {
                        Some(g) if g.freq == v.freq && g.score == v.score => {}
                        Some(_) => diff += 1,
                        None => missing += 1,
                    }
                }
                let mut extra = 0usize;
                for k in self.global_stats.keys() {
                    if !full.contains_key(k) {
                        extra += 1;
                    }
                }
                if diff + missing + extra > 0 {
                    log_debug(
                        "train_single",
                        format!("[verify] step {} diff={}, missing={}, extra={}", step, diff, missing, extra),
                    );
                }
            }
        }

        // 训练后基于最终 entries 计算 token 总数，并按字符总数输出 TPC（tok/char）
        let total_tokens = Self::total_tokens(&self.entries);
        let total_chars = self.corpus_chars;
        let tpc = if total_chars == 0 {
            0.0
        } else {
            total_tokens as f32 / total_chars as f32
        };
        let elapsed_total = start_all.elapsed().as_secs_f32();
        log_line(
            "tpc",
            format!(
                "total_tokens={} total_chars={} tpc={:.6} tok/char elapsed={:.3}s [single]",
                total_tokens, total_chars, tpc, elapsed_total
            ),
        );
    }

    /// 多进程 map-reduce 训练流程（主进程驱动，子进程执行统计/应用）
    fn train_multiprocess(&mut self) -> Result<()> {
        use std::time::Instant;

        if self.entries.is_empty() {
            return Ok(());
        }

        let start_total = Instant::now();
        let force_inc = std::env::var("MP_FORCE_INCREMENTAL").is_ok();
        // 默认启用增量模式（更快、也更符合“每轮 apply 越来越少所以越跑越快”的直觉）。
        // 如需强制走“每步全量重算 stats”（更慢，但便于 debug/对照），可设置：
        //   MP_FULL_RECOMPUTE=1   或   MP_NO_INCREMENTAL=1
        let full_recompute =
            std::env::var("MP_FULL_RECOMPUTE").is_ok() || std::env::var("MP_NO_INCREMENTAL").is_ok();
        let allow_incremental = !full_recompute;
        // 多进程下 heap 容易被“增量更新大量 key”撑爆（并导致主进程 OOM）。
        // 默认关闭：改用扫描 global_stats 选 best（仍是精确 argmax，只是更省内存）。
        // 如需启用 heap：设置 MP_USE_HEAP=1
        let mp_use_heap = std::env::var("MP_USE_HEAP").is_ok();
        let low_mem = if force_inc {
            false
        } else {
            std::env::var("MP_NO_DIFF").is_ok() || std::env::var("MP_LOW_MEM").is_ok()
        };
        let worker_count = if self.cfg.num_workers == 0 {
            num_cpus::get().max(1)
        } else {
            self.cfg.num_workers
        };

        let chunks = partition_entries(std::mem::take(&mut self.entries), worker_count);
        let mut pool = ProcessPool::new(chunks, self.cfg.n_values.clone())?;

        log_line(
            "train_mp",
            format!(
                "start merges={} n_values={:?} workers={} recompute_each_step={}",
                self.cfg.num_merges, self.cfg.n_values, worker_count, self.cfg.recompute_each_step
            ),
        );

        // 初始化全局统计
        let t_init = Instant::now();
        self.global_stats = pool.compute_stats()?;
        log_line(
            "train_mp",
            format!(
                "initial stats done elapsed={:.2}s entries={} stats_keys={}",
                t_init.elapsed().as_secs_f32(),
                self.entries.len(),
                self.global_stats.len()
            ),
        );
        if mp_use_heap
            && !self.cfg.recompute_each_step
            && (force_inc || allow_incremental)
            && !low_mem
        {
            self.rebuild_heap();
        } else {
            self.heap.clear();
        }
        if low_mem {
            // 低内存模式：不保留全局 stats，避免占用峰值内存
            self.global_stats.clear();
        }
        let start_all = Instant::now();
        log_resources("after_initial_stats", Some(start_all));

        let recompute_every = if force_inc {
            None
        } else {
            std::env::var("MP_RECOMP_EVERY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| v > 0)
        };

        for step in 0..self.cfg.num_merges {
            let t0 = Instant::now();
            let stats_ref: HashMap<Ngram, Stat, RandomState>;
            let use_incremental = !self.cfg.recompute_each_step
                && (force_inc || allow_incremental)
                && !low_mem;
            let stats_view: &HashMap<Ngram, Stat, RandomState> = if self.cfg.recompute_each_step
                || !use_incremental
            {
                stats_ref = pool.compute_stats()?;
                // 同步一份到 global_stats，便于诊断/输出
                if low_mem {
                    // 低内存模式：直接用临时引用，后面即可清空
                    self.global_stats = HashMap::with_hasher(RandomState::new());
                    &stats_ref
                } else {
                self.global_stats = stats_ref.clone();
                &self.global_stats
                }
            } else {
                &self.global_stats
            };

            if stats_view.is_empty() {
                break;
            }

            if step % 10 == 0 {
                // 细碎进度降到 debug（真正的关键打点在后面的 log_line 里）
                log_debug(
                    "train_mp",
                    format!(
                        "step {} stats_ready size={} elapsed_total={:.2}s",
                        step,
                        stats_view.len(),
                        start_all.elapsed().as_secs_f32()
                    ),
                );
            }

            fn better(a: (Ngram, Stat), b: (Ngram, Stat)) -> (Ngram, Stat) {
                let (ang, ast) = &a;
                let (bng, bst) = &b;
                if bst.score > ast.score
                    || (bst.score == ast.score && bng.len() > ang.len())
                    || (bst.score == ast.score
                        && bng.len() == ang.len()
                        && bng.as_slice() > ang.as_slice())
                        {
                    b
                } else {
                    a
                }
            }

            // 写入日志文件：便于 OOM/性能定位（eprintln 进不了 LOG_FILE）
            if step % 10 == 0 {
                log_line(
                    "train_mp",
                    format!(
                        "step={} stats_len={} heap_len={} use_incremental={} mp_use_heap={} elapsed_total={:.2}s",
                        step,
                        stats_view.len(),
                        self.heap.len(),
                        use_incremental,
                        mp_use_heap,
                        start_all.elapsed().as_secs_f32()
                    ),
                );
                log_resources("train_mp_step", Some(start_all));
            }

            // best 选择：heap 仅在显式启用时使用；否则扫描（精确 argmax，低内存更稳）
            let best: Option<(Ngram, Stat)> = if use_incremental && mp_use_heap {
                self.pop_best_from_heap()
            } else {
                fn better_ref<'a>(
                    a: (&'a Ngram, &'a Stat),
                    b: (&'a Ngram, &'a Stat),
                ) -> (&'a Ngram, &'a Stat) {
                    let (ang, ast) = a;
                    let (bng, bst) = b;
                    if bst.score > ast.score
                        || (bst.score == ast.score && bng.len() > ang.len())
                        || (bst.score == ast.score
                            && bng.len() == ang.len()
                            && bng.as_slice() > ang.as_slice())
                    {
                        (bng, bst)
                    } else {
                        (ang, ast)
                    }
                }
                // hashbrown 开启 rayon 特性后可直接 par_iter，避免 par_bridge 的额外开销
                let max_chars = self.cfg.max_token_chars;
                let id_to_chars = &self.interner.id_to_chars;
                let id_to_flags = &self.interner.id_to_flags;
                let id_to_token = &self.interner.id_to_token;
                let id_to_end_count = &self.interner.id_to_end_count;
                let forbid_mix = self.cfg.forbid_punct_word_mix;
                let forbid_punct = self.cfg.forbid_punct_tokens;
                let forbid_incomplete = self.cfg.forbid_incomplete_cross_word;
                let cfg = &self.cfg;
                let vocab_sz = self.vocab_size_with_specials();
                stats_view
                    .par_iter()
                    .filter(|(ng, st)| {
                        st.freq > 1
                            && (max_chars == 0 || ngram_char_len(ng, id_to_chars) <= max_chars)
                            && (!forbid_mix || !ngram_mixes_punct_and_word(ng, id_to_flags))
                            && (!forbid_punct || !ngram_has_forbidden_punct(ng, cfg, id_to_flags, id_to_token))
                            && !ngram_cross_word_blocked(ng, cfg, id_to_end_count, vocab_sz)
                            && (!forbid_incomplete || !ngram_is_incomplete_cross_word(ng, id_to_flags))
                    })
                    .map(|(ng, st)| (ng, st))
                    .reduce_with(better_ref)
                    .map(|(ng, st)| (ng.clone(), st.clone()))
            };

            let Some((best_parts, best_stat)) = best else { break };

            // 为新 token 分配 id，并记录 merge 规则
            let replacement_str: String = best_parts.iter().map(|&id| self.interner.get(id)).collect();
            let replacement_id = self.interner.intern(&replacement_str);

            let elapsed_stats = t0.elapsed().as_secs_f32();
            let t_apply = Instant::now();

            // 增量模式：按 bucket 聚合（去重）后再更新 global_stats，并仅对唯一 key 推入候选堆，
            // 避免“逐条 push_candidate”导致 heap 重复爆炸进而 OOM kill。
            let mut any_change = false;
            if use_incremental {
                let (temp_root, worker_dirs, bucket_cnt) =
                    pool.apply_merge_prepare(&best_parts, replacement_id, self.cfg.recompute_each_step)?;

                // 读取每个 worker 的 manifest（位图）：只打开真实存在的桶文件，避免 bucket_cnt*workers*2 次 open 尝试
                let mut manifests: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
                let mut manifest_ok = true;
                for wdir in &worker_dirs {
                    if let Some(m) = read_diff_manifest(wdir, bucket_cnt) {
                        manifests.push(m);
                    } else {
                        manifest_ok = false;
                        break;
                    }
                }

                let buckets_to_read: Vec<usize> = if manifest_ok {
                    let bits_len = diff_bits_len(bucket_cnt);
                    let mut union_bits = vec![0u8; bits_len];
                    for (old_bits, new_bits) in &manifests {
                        for i in 0..bits_len {
                            union_bits[i] |= old_bits[i] | new_bits[i];
                        }
                    }
                    diff_bits_to_indices(&union_bits, bucket_cnt)
                } else {
                    (0..bucket_cnt).collect()
                };

                // 性能关键路径：主进程读取并合并 diff 桶文件。
                //
                // 旧实现按 bucket 串行读取，容易出现“CPU 用不满，但 apply 很慢”的现象（尤其在前期 diff 很大时）。
                // 这里改为 **分批并行** 读取 bucket：每批并行构建 bucket_old/new，再在主线程顺序更新 global_stats，
                // 以在不显著抬高峰值内存的前提下，尽量把 CPU/IO 利用起来。
                let bucket_batch: usize = std::env::var("MP_BUCKET_BATCH")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    // 默认值偏向“高吞吐”（可通过 MP_BUCKET_BATCH 调小以降低峰值内存）
                    .unwrap_or(256)
                    .clamp(1, 8192);

                for batch in buckets_to_read.chunks(bucket_batch) {
                    // (old, new, any)
                    let mut results: Vec<(
                        HashMap<Ngram, Stat, RandomState>,
                        HashMap<Ngram, Stat, RandomState>,
                        bool,
                    )> = batch
                        .par_iter()
                        .map(|&b| {
                            let mut bucket_old: HashMap<Ngram, Stat, RandomState> =
                                HashMap::with_hasher(RandomState::new());
                            let mut bucket_new: HashMap<Ngram, Stat, RandomState> =
                                HashMap::with_hasher(RandomState::new());
                            let mut any = false;

                    for (wi, wdir) in worker_dirs.iter().enumerate() {
                        // 有 manifest 时用位图过滤，否则 fallback 走 open 失败即跳过
                        let (has_old, has_new) = if manifest_ok {
                            let (ref old_bits, ref new_bits) = manifests[wi];
                            (diff_bit_get(old_bits, b), diff_bit_get(new_bits, b))
                        } else {
                            (true, true)
                        };

                        if has_old {
                            let p_old = wdir.join(format!("old_{b}.bin"));
                                    if let Ok(any_read) =
                                        read_bucket_file_into(&p_old, &mut bucket_old)
                                    {
                                        any |= any_read;
                                // 读完立刻删除，尤其在 TMPDIR=/dev/shm 时可显著降低内存峰值
                                let _ = fs::remove_file(&p_old);
                            }
                        }
                        if has_new {
                            let p_new = wdir.join(format!("new_{b}.bin"));
                                    if let Ok(any_read) =
                                        read_bucket_file_into(&p_new, &mut bucket_new)
                                    {
                                        any |= any_read;
                                let _ = fs::remove_file(&p_new);
                            }
                        }
                            }

                            (bucket_old, bucket_new, any)
                        })
                        .collect();

                    for (bucket_old, bucket_new, any) in results.drain(..) {
                        if any {
                            any_change = true;
                    }

                    if !bucket_old.is_empty() {
                        // 只保留 freq>1，减少 global_stats 体积；不会影响可合并候选集合
                        if mp_use_heap {
                                Self::accumulate_stats(
                                    &mut self.global_stats,
                                    &bucket_old,
                                    1,
                                    false,
                                    true,
                                );
                        } else {
                                Self::accumulate_stats_owned(
                                    &mut self.global_stats,
                                    bucket_old,
                                    1,
                                    false,
                                    true,
                                );
                        }
                    }

                    if !bucket_new.is_empty() {
                        if mp_use_heap {
                                Self::accumulate_stats(
                                    &mut self.global_stats,
                                    &bucket_new,
                                    1,
                                    true,
                                    true,
                                );
                            for k in bucket_new.keys() {
                                // 对“增加项”必须 push：否则 score 变大但堆里仍是旧的低分，会选错 best（影响功能正确性）
                                self.push_candidate(k);
                            }
                        } else {
                                Self::accumulate_stats_owned(
                                    &mut self.global_stats,
                                    bucket_new,
                                    1,
                                    true,
                                    true,
                                );
                            }
                        }
                    }
                }
                let _ = fs::remove_dir_all(&temp_root);
            } else {
            let (old_diff, new_diff) =
                pool.apply_merge(&best_parts, replacement_id, self.cfg.recompute_each_step)?;
                any_change = !(old_diff.is_empty() && new_diff.is_empty());
                if use_incremental {
                    // 不会进入
                }
                if !old_diff.is_empty() {
                    Self::accumulate_stats_owned(&mut self.global_stats, old_diff, 1, false, true);
                }
                if !new_diff.is_empty() {
                    if mp_use_heap {
                        // heap 模式下仍保留 key 以便 push；这里不走 owned 路径
                        Self::accumulate_stats(&mut self.global_stats, &new_diff, 1, true, true);
                        for k in new_diff.keys() {
                            self.push_candidate(k);
                        }
                    } else {
                        Self::accumulate_stats_owned(&mut self.global_stats, new_diff, 1, true, true);
                    }
                }
            }

            // 如果本次合并没有产生任何增量，视为无效 merge：尝试全量重算一次以防止重复选择；
            // 若仍无可合并则直接停止，避免出现重复的 merge 记录。
            let no_change = !any_change;
            if no_change {
                log_warn(
                    "train_mp",
                    format!(
                        "step {} no-op merge for {:?}, recompute stats once",
                        step, best_parts
                    ),
                );
                if self.cfg.recompute_each_step || !use_incremental {
                    break;
                }
                    self.global_stats = pool.compute_stats()?;
                if mp_use_heap {
                    self.rebuild_heap();
                } else {
                    self.heap.clear();
                }
                    if self.global_stats.is_empty() {
                        log_warn("train_mp", "stats empty after recompute, stop");
                        break;
                    }
                    continue;
            }

            if use_incremental {
                // 周期性全量重算，防止增量误差累积
                if let Some(k) = recompute_every {
                    if (step + 1) % k == 0 {
                        self.global_stats = pool.compute_stats()?;
                        if mp_use_heap {
                        self.rebuild_heap();
                        } else {
                            self.heap.clear();
                }
                    }
                }
            }

            // 记录 merge
            self.merges.push(MergeRule {
                parts: best_parts
                    .iter()
                    .map(|&id| self.interner.get(id).to_string())
                    .collect(),
                replacement: self.interner.get(replacement_id).to_string(),
                freq: best_stat.freq,
                score: best_stat.score,
            });

            let elapsed_apply = t_apply.elapsed().as_secs_f32();
            log_line(
                "merge",
                format!(
                    "{:>6}: freq={} score={} stats={:.3}s apply={:.3}s total={:.1}s [mp]",
                    step,
                    best_stat.freq,
                    best_stat.score,
                    elapsed_stats,
                    elapsed_apply,
                    start_all.elapsed().as_secs_f32()
                ),
            );

            // 目标词表大小：达到后提前停止（多进程路径）。
            if self.cfg.aim_token_num > 0 {
                let vocab_sz = self.vocab_size_with_specials();
                if vocab_sz >= self.cfg.aim_token_num {
                    log_line(
                        "train_mp",
                        format!(
                            "reach aim_token_num={} (vocab_size={}), stop at step {} (elapsed_total={:.2}s)",
                            self.cfg.aim_token_num,
                            vocab_sz,
                            step + 1,
                            start_all.elapsed().as_secs_f32()
                        ),
                    );
                    break;
                }
            }

            // 可选：校验增量统计是否与全量重算一致（仅在 VERIFY_STATS_MP=1 时启用，方便定位 drift）
            if !self.cfg.recompute_each_step && std::env::var("VERIFY_STATS_MP").is_ok() {
                let full = pool.compute_stats()?;
                if !use_incremental {
                    // 重算模式下直接同步，不做差异检查
                    self.global_stats = full;
                } else {
                    let verbose = std::env::var("VERIFY_STATS_MP_VERBOSE").is_ok();
                    let mut diff = 0usize;
                    let mut missing = 0usize;
                    let mut sample_logged = 0usize;
                    for (k, v) in &full {
                        match self.global_stats.get(k) {
                            Some(g) if g.freq == v.freq && g.score == v.score => {}
                            Some(g) => {
                                diff += 1;
                                if verbose && sample_logged < 5 {
                                    let toks: Vec<&str> =
                                        k.iter().map(|&id| self.interner.get(id)).collect();
                                    log_debug(
                                        "verify_mp",
                                        format!(
                                            "step {} mismatch {:?} full=({},{}) inc=({},{})",
                                            step, toks, v.freq, v.score, g.freq, g.score
                                        ),
                                    );
                                    sample_logged += 1;
                                }
                            }
                            None => {
                                missing += 1;
                                if verbose && sample_logged < 5 {
                                    let toks: Vec<&str> =
                                        k.iter().map(|&id| self.interner.get(id)).collect();
                                    log_debug(
                                        "verify_mp",
                                        format!(
                                            "step {} missing {:?} full=({},{})",
                                            step, toks, v.freq, v.score
                                        ),
                                    );
                                    sample_logged += 1;
                                }
                            }
                        }
                    }
                    let mut extra = 0usize;
                    for k in self.global_stats.keys() {
                        if !full.contains_key(k) {
                            extra += 1;
                            if verbose && sample_logged < 5 {
                                let toks: Vec<&str> =
                                    k.iter().map(|&id| self.interner.get(id)).collect();
                                let g = self.global_stats.get(k).unwrap();
                                log_debug(
                                    "verify_mp",
                                    format!(
                                        "step {} extra {:?} inc=({},{})",
                                        step, toks, g.freq, g.score
                                    ),
                                );
                                sample_logged += 1;
                            }
                        }
                    }
                    if diff + missing + extra > 0 {
                        log_debug(
                            "verify_mp",
                            format!(
                                "step {} diff={}, missing={}, extra={}",
                                step, diff, missing, extra
                            ),
                        );
                    }
                }
            }

            // 若禁用增量模式，则在合并后重算一次全量统计，保证下一轮使用最新的全局表
            if !use_incremental {
                let fresh = pool.compute_stats()?;
                if low_mem {
                    self.global_stats.clear();
                } else {
                    self.global_stats = fresh;
                }
            }
            if low_mem {
                // 循环末尾，确保释放本轮的临时 stats
                self.global_stats.clear();
            }
        }

        // 收尾：取回各 worker 的 entries，并关闭进程。
        //
        // 注意：在多进程训练路径中，`global_stats`/heap 在训练过程中可能非常大。
        // 如果在收尾阶段再做一次 `rebuild_global_stats()`（它会做一次“全量 n-gram 统计”），
        // 很容易在短时间内触发巨大的内存峰值（甚至被 OOM killer 直接杀掉）。
        //
        // 对于常见用法（训练后直接导出 HF tokenizer / token_table），收尾阶段并不需要重建全局统计。
        // 如确实需要（调试/诊断），可通过环境变量显式开启：
        //   FINAL_REBUILD_GLOBAL_STATS=1
        let do_final_rebuild = std::env::var("FINAL_REBUILD_GLOBAL_STATS").is_ok();

        // 尽早释放候选堆，降低内存峰值
        self.heap.clear();
        if !do_final_rebuild {
            // 直接 drop 掉训练期间累积的全局统计，避免“entries 回收 + 重建统计”叠加导致 OOM
            self.global_stats = HashMap::with_hasher(RandomState::new());
        }

        self.entries = pool.collect_entries()?;
        pool.shutdown();

        if do_final_rebuild {
            // 先 drop 旧表，避免出现“两份巨大的 global_stats 同时存在”的峰值
            self.global_stats = HashMap::with_hasher(RandomState::new());
        self.rebuild_global_stats();
        }

        // 训练后基于最终 entries 计算 token 总数，并按字符总数输出 TPC（tok/char）
        let total_tokens = Self::total_tokens(&self.entries);
        let total_chars = self.corpus_chars;
        let tpc = if total_chars == 0 {
            0.0
        } else {
            total_tokens as f32 / total_chars as f32
        };
        let elapsed_total = start_total.elapsed().as_secs_f32();
        log_line(
            "tpc",
            format!(
                "total_tokens={} total_chars={} tpc={:.6} tok/char elapsed={:.3}s [mp]",
                total_tokens, total_chars, tpc, elapsed_total
            ),
        );
        Ok(())
    }

    fn apply_merge(&mut self, parts: &[u32], best_stat: Stat) {
        // 构造替换 token（字符串拼接后再 intern）
        let replacement_str: String = parts
            .iter()
            .map(|&id| self.interner.get(id))
            .collect();
        let replacement_id = self.interner.intern(&replacement_str);
        let plen = parts.len();
        let n_vals = self.cfg.n_values.clone();

        #[derive(Default)]
        struct EntryDelta {
            tokens: Vec<u32>,
            freq: u32,
            old_local: HashMap<Ngram, Stat, RandomState>,
            new_local: HashMap<Ngram, Stat, RandomState>,
        }

        fn estimate_from_starts(starts: &HashMap<usize, Vec<usize>>) -> usize {
            starts.values().map(|v| v.len()).sum()
        }

        #[derive(Default)]
        struct Buffers {
            positions: Vec<usize>,
            positions_new: Vec<usize>,
            new_tokens: Vec<u32>,
            old_local: HashMap<Ngram, Stat, RandomState>,
            new_local: HashMap<Ngram, Stat, RandomState>,
            tmp_ngram: [u32; 16],
        }

        let entries = std::mem::take(&mut self.entries);
        let mut deltas: Vec<EntryDelta> = self.with_pool(|| {
            entries
                .into_par_iter()
                .map_init(
                    || Buffers::default(),
                    |bufs, entry| {
                        bufs.positions.clear();
                        bufs.positions_new.clear();
                        bufs.new_tokens.clear();
                        bufs.old_local.clear();
                        bufs.new_local.clear();

                        let len_old = entry.tokens.len();
                        if len_old < plen {
                            return EntryDelta {
                                tokens: entry.tokens,
                                freq: entry.freq,
                                ..Default::default()
                            };
                        }

                    // 找出所有匹配起点（不重叠向后扫描）
                        let positions = &mut bufs.positions;
                        let mut idx = 0;
                        while idx + plen <= len_old {
                            if entry.tokens[idx..idx + plen] == parts[..] {
                                positions.push(idx);
                                idx += plen;
                            } else {
                                idx += 1;
                            }
                        }
                    if positions.is_empty() {
                        return EntryDelta {
                            tokens: entry.tokens,
                            freq: entry.freq,
                            ..Default::default()
                        };
                    }

                    // 计算受影响窗口起点集合（旧序列）
                    fn collect_starts(
                        len: usize,
                        positions: &[usize],
                        n_vals: &[usize],
                        plen: usize,
                    ) -> HashMap<usize, Vec<usize>> {
                        let mut map: HashMap<usize, Vec<usize>> =
                            HashMap::with_capacity(n_vals.len());
                        for &pos in positions {
                            for &n in n_vals {
                                let start_lo = pos.saturating_sub(n.saturating_sub(1));
                                let start_hi = pos + plen.saturating_sub(1);
                                let v = map.entry(n).or_insert_with(Vec::new);
                                for s in start_lo..=start_hi {
                                    if s + n <= len {
                                        v.push(s);
                                    }
                                }
                            }
                        }
                        for v in map.values_mut() {
                            v.sort_unstable();
                            v.dedup();
                        }
                        map
                    }

                    let affected_old = collect_starts(len_old, &positions, &n_vals, plen);

                    // 构造新 tokens（按记录的匹配位置）
                        let new_len_est = len_old
                            .saturating_sub(positions.len().saturating_mul(plen.saturating_sub(1)));
                        let new_tokens = &mut bufs.new_tokens;
                        new_tokens.clear();
                        new_tokens.reserve(new_len_est.max(len_old));
                        let mut p_iter = positions.iter().copied().peekable();
                        let mut i = 0;
                        let mut shift = 0usize;
                        let positions_new = &mut bufs.positions_new;
                        positions_new.clear();
                        positions_new.reserve(positions.len());
                        while i < len_old {
                            if let Some(&p) = p_iter.peek() {
                                if i == p {
                                    positions_new.push(i - shift);
                                    new_tokens.push(replacement_id);
                                    i += plen;
                                    p_iter.next();
                                    shift += plen.saturating_sub(1);
                                    continue;
                                }
                            }
                            new_tokens.push(entry.tokens[i]);
                            i += 1;
                        }

                        let len_new = new_tokens.len();
                        let affected_new = collect_starts(len_new, &positions_new, &n_vals, 1);

                    // 旧窗口（仅受影响区域）统计
                        let old_cap = estimate_from_starts(&affected_old);
                        let old_local = &mut bufs.old_local;
                        old_local.clear();
                        if old_local.capacity() < old_cap {
                            old_local.reserve(old_cap - old_local.capacity());
                        }
                        let tmp = &mut bufs.tmp_ngram;
                        for (&n, starts) in &affected_old {
                            for &s in starts {
                                if s + n <= len_old {
                                    tmp[..n].copy_from_slice(&entry.tokens[s..s + n]);
                                    let ng = SmallVec::<[u32; 9]>::from_slice(&tmp[..n]);
                                    let e = old_local.entry(ng).or_default();
                                    e.freq += 1;
                                    e.score += n.saturating_sub(1) as u64;
                                }
                            }
                        }

                    // 新窗口（仅受影响区域）统计
                        let new_cap = estimate_from_starts(&affected_new);
                        let new_local = &mut bufs.new_local;
                        new_local.clear();
                        if new_local.capacity() < new_cap {
                            new_local.reserve(new_cap - new_local.capacity());
                        }
                        let tmp = &mut bufs.tmp_ngram;
                        for (&n, starts) in &affected_new {
                            for &s in starts {
                                if s + n <= len_new {
                                    tmp[..n].copy_from_slice(&new_tokens[s..s + n]);
                                    let ng = SmallVec::<[u32; 9]>::from_slice(&tmp[..n]);
                                    let e = new_local.entry(ng).or_default();
                                    e.freq += 1;
                                    e.score += n.saturating_sub(1) as u64;
                                }
                            }
                        }

                    EntryDelta {
                            tokens: new_tokens.clone(),
                        freq: entry.freq,
                            old_local: std::mem::take(old_local),
                            new_local: std::mem::take(new_local),
                    }
                    },
                )
                .collect()
        });

        // 合并全局统计
        if !self.cfg.recompute_each_step {
            // 关键优化：
            // - 这里消费（take）每个 delta 的 old/new local map，用 `accumulate_stats_owned` 避免 `ng.clone()`
            // - 同时尽早释放每条 entry 的局部统计，降低 apply 阶段峰值内存
            for delta in deltas.iter_mut() {
                if !delta.old_local.is_empty() {
                    let old_local = std::mem::take(&mut delta.old_local);
                    if self.cfg.use_heap {
                        for ng in old_local.keys() {
                            self.push_candidate(ng);
                        }
                    }
                    Self::accumulate_stats_owned(
                        &mut self.global_stats,
                        old_local,
                        delta.freq,
                        false,
                        true,
                    );
                }
                if !delta.new_local.is_empty() {
                    let new_local = std::mem::take(&mut delta.new_local);
                    if self.cfg.use_heap {
                        for ng in new_local.keys() {
                    self.push_candidate(ng);
                        }
                    }
                    Self::accumulate_stats_owned(
                        &mut self.global_stats,
                        new_local,
                        delta.freq,
                        true,
                        true,
                    );
                }
            }
        }

        // 合并重复序列：分桶聚合，减少全局冲突
        let bucket_count = std::cmp::max(8, self.cfg.num_workers.max(1) * 2);
        let new_buckets = || {
            (0..bucket_count)
                .map(|_| HashMap::with_hasher(RandomState::new()))
                .collect::<Vec<_>>()
        };

        let buckets = deltas
            .into_par_iter()
            .fold(
                new_buckets,
                |mut shards, d| {
                    let bucket = d
                        .tokens
                        .first()
                        .copied()
                        .unwrap_or(0) as usize
                        % bucket_count;
                    *shards[bucket].entry(d.tokens).or_insert(0) += d.freq;
                    shards
                },
            )
            .reduce(
                new_buckets,
                |mut acc, mut shards| {
                    for (i, mut m) in shards.drain(..).enumerate() {
                        let acc_b = &mut acc[i];
                        for (k, v) in m.drain() {
                            *acc_b.entry(k).or_insert(0) += v;
                        }
                    }
                    acc
                },
            );

        self.entries = buckets
            .into_iter()
            .flat_map(|m| m.into_iter())
            .map(|(tokens, freq)| SeqEntry { tokens, freq })
            .collect();

        // 记录 merges
        self.merges.push(MergeRule {
            parts: parts
                .iter()
                .map(|&id| self.interner.get(id).to_string())
                .collect(),
            replacement: self.interner.get(replacement_id).to_string(),
            freq: best_stat.freq,
            score: best_stat.score,
        });
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        // 默认使用“TPC 最小”的 DP 分词。
        // 如需原先的 BPE 顺序分词，可调用 `tokenize_bpe()`。
        self.tokenize_ids_min_tpc(text)
            .into_iter()
            .map(|id| self.interner.get(id).to_string())
            .collect()
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        // 保存时再重建字符串形式的 vocab，保持与原实现一致
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for entry in &self.entries {
            let key = entry
                .tokens
                .iter()
                .map(|&id| self.interner.get(id))
                .collect::<Vec<_>>()
                .join(" ");
            *vocab.entry(key).or_insert(0) += entry.freq;
        }
        let table = TokenTable {
            merges: self.merges.clone(),
            vocab,
        };
        let data = serde_json::to_vec_pretty(&table)?;
        let mut f = File::create(path)?;
        f.write_all(&data)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P, cfg: TokenizerConfig) -> Result<Self> {
        let f = File::open(path)?;
        let table: TokenTable = serde_json::from_reader(f)?;
        let mut interner = Interner::new();
        // 先构造 entries
        let mut entries: Vec<SeqEntry> = Vec::new();
        for (k, f) in table.vocab.iter() {
            let toks: Vec<u32> = k
                .split_whitespace()
                .map(|s| interner.intern(s))
                .collect();
            entries.push(SeqEntry {
                tokens: toks,
                freq: *f,
            });
        }
        // 确保 merges 中的 token 也注册进 interner
        for m in &table.merges {
            for p in &m.parts {
                interner.intern(p);
            }
            interner.intern(&m.replacement);
        }

        let mut tk = Self {
            entries,
            merges: table.merges,
            cfg,
            global_stats: HashMap::with_hasher(RandomState::new()),
            interner,
            corpus_chars: 0, // 无原始语料信息，保留 0
            heap: BinaryHeap::new(),
            token_trie: OnceLock::new(),
        };
        tk.rebuild_global_stats();
        Ok(tk)
    }

    /// 从“流式统计”得到的序列计数（token 序列 -> freq）直接构建并训练。
    ///
    /// 这个入口用于 Python iterator/parquet 这类流式输入：避免必须把整个语料 Vec<String> 常驻内存。
    pub fn new_from_counts(
        counts: HashMap<Vec<String>, u32, RandomState>,
        corpus_chars: u64,
        cfg: TokenizerConfig,
    ) -> Result<Self> {
        let mut interner = Interner::new();
        let mut entries: Vec<SeqEntry> = Vec::with_capacity(counts.len());
        for (k, f) in counts {
            let toks: Vec<u32> = k.iter().map(|s| interner.intern(s)).collect();
            entries.push(SeqEntry { tokens: toks, freq: f });
        }

        let mut tk = Self {
            entries,
            merges: Vec::new(),
            cfg,
            global_stats: HashMap::with_hasher(RandomState::new()),
            interner,
            corpus_chars,
            heap: BinaryHeap::new(),
            token_trie: OnceLock::new(),
        };

        if tk.cfg.use_multiprocess {
            log_line("init", "multiprocess path: skip single-thread rebuild_global_stats");
            tk.train()?;
        } else {
            tk.rebuild_global_stats();
            tk.train()?;
        }
        Ok(tk)
    }

    fn rebuild_global_stats(&mut self) {
        // 并行重建全局统计（否则大语料下会长时间处于单核运行，给人“卡死”的感觉）。
        // 注意：此处的内存峰值主要由 global_stats 本身决定；并行仅增加少量局部 map 开销，但能显著加速初始化。
        let t0 = std::time::Instant::now();
        log_line(
            "stats",
            format!(
                "rebuild_global_stats start entries={} n_values={:?} num_workers={}",
                self.entries.len(),
                self.cfg.n_values,
                self.cfg.num_workers
            ),
        );

        self.global_stats = self.compute_stats_full();

        log_line(
            "stats",
            format!(
                "rebuild_global_stats done keys={} elapsed={:.2}s",
                self.global_stats.len(),
                t0.elapsed().as_secs_f32()
            ),
        );
        if !self.cfg.recompute_each_step && self.cfg.use_heap {
            self.rebuild_heap();
        } else {
            self.heap.clear();
        }
    }

    fn entry_stats_tokens(
        tokens: &[u32],
        n_vals: &[usize],
    ) -> HashMap<Ngram, Stat, RandomState> {
        let mut local: HashMap<Ngram, Stat, RandomState> = HashMap::with_capacity_and_hasher(
            Self::estimate_windows(tokens.len(), n_vals),
            RandomState::new(),
        );

        // 专用滑窗，避免重复长度判断，兼容未来更大 n
        for &n in n_vals {
            let len = tokens.len();
            if len < n {
                continue;
            }
            // 滑窗复制，当前 n 大小
            for i in 0..=len - n {
                let mut ng = SmallVec::<[u32; 9]>::with_capacity(n);
                ng.extend_from_slice(&tokens[i..i + n]);
                let entry = local.entry(ng).or_default();
                entry.freq += 1;
                entry.score += n.saturating_sub(1) as u64;
            }
        }
        local
    }

    fn accumulate_stats(
        global: &mut HashMap<Ngram, Stat, RandomState>,
        local: &HashMap<Ngram, Stat, RandomState>,
        mult: u32,
        add: bool,
        prune_le1: bool,
    ) {
        for (ng, st) in local {
            let dfreq = st.freq.saturating_mul(mult);
            let dscore = st.score.saturating_mul(mult as u64);
            if add {
                let g = global.entry(ng.clone()).or_default();
                g.freq += dfreq;
                g.score += dscore;
                // 仅在需要时保留 freq > 1，减少全局表体积
                if prune_le1 && g.freq <= 1 {
                    global.remove(ng);
                }
            } else if let Some(g) = global.get_mut(ng) {
                g.freq = g.freq.saturating_sub(dfreq);
                g.score = g.score.saturating_sub(dscore);
                if prune_le1 && g.freq <= 1 {
                    global.remove(ng);
                }
            }
        }
    }

    /// 与 `accumulate_stats` 等价，但消费（drain）本地 map，尽量避免 `ng.clone()`。
    /// 仅用于主进程增量 apply 阶段（bucket_* 聚合后更新 global_stats）。
    fn accumulate_stats_owned(
        global: &mut HashMap<Ngram, Stat, RandomState>,
        mut local: HashMap<Ngram, Stat, RandomState>,
        mult: u32,
        add: bool,
        prune_le1: bool,
    ) {
        use hashbrown::hash_map::Entry;
        if add {
            for (ng, st) in local.drain() {
                let dfreq = st.freq.saturating_mul(mult);
                let dscore = st.score.saturating_mul(mult as u64);
                if prune_le1 && dfreq <= 1 {
                    // 等价于“插入后立即 remove”，但少一次 clone/rehash
                    continue;
                }
                match global.entry(ng) {
                    Entry::Occupied(mut o) => {
                        let g = o.get_mut();
                        g.freq += dfreq;
                        g.score += dscore;
                        if prune_le1 && g.freq <= 1 {
                            let _ = o.remove_entry();
                        }
                    }
                    Entry::Vacant(v) => {
                        v.insert(Stat {
                            freq: dfreq,
                            score: dscore,
                        });
                    }
                }
            }
        } else {
            for (ng, st) in local.drain() {
                let dfreq = st.freq.saturating_mul(mult);
                let dscore = st.score.saturating_mul(mult as u64);
                if let Entry::Occupied(mut o) = global.entry(ng) {
                    let g = o.get_mut();
                    g.freq = g.freq.saturating_sub(dfreq);
                    g.score = g.score.saturating_sub(dscore);
                    if prune_le1 && g.freq <= 1 {
                        let _ = o.remove_entry();
                    }
                }
            }
        }
    }
}

/// 将 entries 均匀分配到 worker
fn partition_entries(entries: Vec<SeqEntry>, workers: usize) -> Vec<Vec<SeqEntry>> {
    let worker_num = workers.max(1);
    let mut buckets: Vec<Vec<SeqEntry>> = vec![Vec::new(); worker_num];
    for (idx, entry) in entries.into_iter().enumerate() {
        buckets[idx % worker_num].push(entry);
    }
    buckets.into_iter().filter(|b| !b.is_empty()).collect()
}

#[derive(Serialize, Deserialize, Debug)]
enum WorkerRequest {
    Init {
        entries: Vec<SeqEntry>,
        n_values: Vec<usize>,
    },
    /// 预处理：接收原始行文本，返回局部词频（字符串 token）
    Preprocess {
        lines: Vec<String>,
    },
    ComputeStats,
    ApplyMerge {
        parts: Vec<u32>,
        replacement_id: u32,
        recompute_each_step: bool,
        return_diff: bool,
        bucket_cnt: usize,
        temp_dir: String,
    },
    DumpEntries,
    Shutdown,
}

#[derive(Serialize, Deserialize, Debug)]
enum WorkerResponse {
    Ack,
    /// 预处理回包：局部词频（字符串 token）
    PreStats { stats: Vec<(Vec<String>, u32)> },
    Stats { stats: HashMap<Ngram, Stat, RandomState> },
    ApplyResult {
        entry_count: usize,
    },
    Entries { entries: Vec<SeqEntry> },
    Error(String),
}

fn write_msg<W: Write, T: Serialize>(w: &mut W, msg: &T) -> Result<()> {
    let data = bincode::serialize(msg)?;
    let len = data.len() as u64;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&data)?;
    w.flush()?;
    Ok(())
}

fn read_msg_tagged<R: Read, T: DeserializeOwned>(r: &mut R, tag: &str) -> Result<T> {
    let mut len_buf = [0u8; 8];
    r.read_exact(&mut len_buf)
        .map_err(|e| anyhow::anyhow!("[ipc:{}] read len failed: {}", tag, e))?;
    let len = u64::from_le_bytes(len_buf) as usize;
    let mut data = vec![0u8; len];
    r.read_exact(&mut data)
        .map_err(|e| anyhow::anyhow!("[ipc:{}] read body failed (want {} bytes): {}", tag, len, e))?;
    bincode::deserialize(&data).map_err(|e| anyhow::anyhow!("[ipc:{}] deserialize failed: {}", tag, e))
}

fn counts_vec_to_map(vec: Vec<(Vec<String>, u32)>) -> HashMap<Vec<String>, u32, RandomState> {
    let mut map = HashMap::with_hasher(RandomState::new());
    for (k, v) in vec {
        map.insert(k, v);
    }
    map
}

fn counts_map_to_vec(map: HashMap<Vec<String>, u32, RandomState>) -> Vec<(Vec<String>, u32)> {
    map.into_iter().collect()
}

/// 调用子进程处理语料分片，返回局部词频
fn preprocess_chunk_in_worker(
    exe: &Path,
    chunk: Vec<String>,
) -> Result<HashMap<Vec<String>, u32, RandomState>> {
    let mut cmd = Command::new(exe);
    // 当库被 Python 扩展调用时，current_exe() 通常是 python，本地 `--as-worker` 不存在；
    // 允许通过环境变量切换为 python -c 调用扩展模块的 worker 入口。
    if std::env::var("MP_PY_WORKER").is_ok() {
        cmd.arg("-c").arg("import length_tokenizer_rs as m; m._run_worker()");
    } else {
        cmd.arg("--as-worker");
    }
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;

    let mut writer = BufWriter::new(
        child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("preprocess worker missing stdin"))?,
    );
    let mut reader = BufReader::new(
        child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("preprocess worker missing stdout"))?,
    );

    write_msg(&mut writer, &WorkerRequest::Preprocess { lines: chunk })?;
    let stats = match read_msg_tagged::<_, WorkerResponse>(&mut reader, "worker:preprocess")? {
        WorkerResponse::PreStats { stats } => stats,
        other => {
            return Err(anyhow::anyhow!("unexpected preprocess reply: {:?}", other));
        }
    };

    // 关闭 worker
    let _ = write_msg(&mut writer, &WorkerRequest::Shutdown);
    let _ = child.wait();

    Ok(counts_vec_to_map(stats))
}

fn merge_stats_into(
    acc: &mut HashMap<Ngram, Stat, RandomState>,
    from: HashMap<Ngram, Stat, RandomState>,
) {
    for (k, v) in from {
        let e = acc.entry(k).or_default();
        e.freq += v.freq;
        e.score += v.score;
    }
}

fn prune_le1(map: &mut HashMap<Ngram, Stat, RandomState>) {
    map.retain(|_, st| st.freq > 1);
}

struct WorkerHandle {
    id: usize,
    child: Child,
    writer: BufWriter<ChildStdin>,
    reader: BufReader<ChildStdout>,
}

impl WorkerHandle {
    /// 兼容原有阻塞式发送并等待回复
    fn send(&mut self, req: &WorkerRequest, tag: &str) -> Result<WorkerResponse> {
        write_msg(&mut self.writer, req)?;
        read_msg_tagged(&mut self.reader, tag)
    }

    /// 仅发送请求（写入并 flush），不阻塞等待回复
    fn send_only(&mut self, req: &WorkerRequest) -> Result<()> {
        write_msg(&mut self.writer, req)
    }

    /// 仅接收回复（阻塞读）
    fn recv(&mut self, tag: &str) -> Result<WorkerResponse> {
        read_msg_tagged(&mut self.reader, tag)
    }
}

struct ProcessPool {
    workers: Vec<WorkerHandle>,
}

impl ProcessPool {
    fn new(chunks: Vec<Vec<SeqEntry>>, n_values: Vec<usize>) -> Result<Self> {
        let exe = std::env::current_exe()?;
        let mut workers = Vec::new();

        // 默认 worker 线程数：避免每个 worker 都使用 num_cpus() 导致严重超额并发（性能反而下降）。
        // 如果用户显式设置了 WORKER_THREADS，则尊重用户设置。
        let default_worker_threads: Option<String> = if std::env::var("WORKER_THREADS").is_ok() {
            None
        } else {
            let total = num_cpus::get().max(1);
            let worker_total = chunks.len().max(1);
            let per = (total / worker_total).max(1);
            // 限制上限，避免 worker_total 较小时每个 worker 开太多线程造成抖动/内存膨胀
            let per = per.clamp(1, 4);
            Some(per.to_string())
        };

        for (idx, entries) in chunks.into_iter().enumerate() {
            let mut cmd = Command::new(&exe);
            if let Some(ref v) = default_worker_threads {
                cmd.env("WORKER_THREADS", v);
            }
            if std::env::var("MP_PY_WORKER").is_ok() {
                cmd.arg("-c").arg("import length_tokenizer_rs as m; m._run_worker()");
            } else {
                cmd.arg("--as-worker");
            }
            let mut child = cmd
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|e| anyhow::anyhow!("spawn worker {} failed: {}", idx, e))?;

            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| anyhow::anyhow!("worker {} missing stdin", idx))?;
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| anyhow::anyhow!("worker {} missing stdout", idx))?;

            let mut writer = BufWriter::new(stdin);
            let mut reader = BufReader::new(stdout);
            let init_req = WorkerRequest::Init {
                entries,
                n_values: n_values.clone(),
            };
            log_debug("proc_pool", format!("send Init to worker {}", idx));
            write_msg(&mut writer, &init_req)?;
            match read_msg_tagged::<_, WorkerResponse>(&mut reader, &format!("worker:{}:ack", idx))? {
                WorkerResponse::Ack => {}
                other => {
                    return Err(anyhow::anyhow!("worker {} init failed: {:?}", idx, other));
                }
            }
            log_debug("proc_pool", format!("worker {} ack received", idx));

            workers.push(WorkerHandle {
                id: idx,
                child,
                writer,
                reader,
            });
        }
        Ok(Self { workers })
    }

    fn compute_stats(&mut self) -> Result<HashMap<Ngram, Stat, RandomState>> {
        use crossbeam_channel::unbounded;
        use std::thread;

        let t = std::time::Instant::now();
        let worker_total = self.workers.len();
        log_debug("proc_pool", format!("stats broadcast workers={}", worker_total));
        // 异步收包 + 并行 reduce
        let (tx, rx) = unbounded();
        let acc = thread::scope(|scope| -> Result<HashMap<Ngram, Stat, RandomState>> {
        for worker in self.workers.iter_mut() {
            worker.send_only(&WorkerRequest::ComputeStats)?;
        }
        for (i, worker) in self.workers.iter_mut().enumerate() {
                let tx = tx.clone();
                scope.spawn(move || {
            let t_w = std::time::Instant::now();
                    let res = match worker.recv(&format!("worker:{}:stats", i)) {
                        Ok(WorkerResponse::Stats { stats }) => {
                            log_debug(
                                "proc_pool",
                                format!(
                                    "stats {}/{} reply_keys={} elapsed={:.2}s",
                                    i + 1,
                                    worker_total,
                                    stats.len(),
                                    t_w.elapsed().as_secs_f32()
                                ),
                            );
                            tx.send(Ok(stats)).ok();
                        }
                        Ok(other) => {
                            tx.send(Err(anyhow::anyhow!("unexpected stats reply: {:?}", other)))
                                .ok();
                }
                        Err(e) => {
                            tx.send(Err(e)).ok();
            }
                    };
                    res
                });
            }
            drop(tx);
            let stats_vec: Result<Vec<_>> = rx.into_iter().collect();
            let stats_vec = stats_vec?;
            let total_keys: usize = stats_vec.iter().map(|m| m.len()).sum();
            let shard = (total_keys / worker_total.max(1)).max(8);
            let acc = stats_vec
                .into_par_iter()
                .reduce(
                    || HashMap::with_capacity_and_hasher(shard, RandomState::new()),
                    |mut a, b| {
                        merge_stats_into(&mut a, b);
                        a
                    },
                );
            log_debug(
                "proc_pool",
                format!(
                    "stats done keys={} elapsed={:.2}s",
                    acc.len(),
                    t.elapsed().as_secs_f32()
                ),
            );
            Ok(acc)
        })?;
        Ok(acc)
    }

    fn apply_merge_prepare(
        &mut self,
        parts: &[u32],
        replacement_id: u32,
        recompute_each_step: bool,
    ) -> Result<(PathBuf, Vec<PathBuf>, usize)> {
        let low_mem = std::env::var("MP_NO_DIFF").is_ok() || std::env::var("MP_LOW_MEM").is_ok();
        let bucket_cnt = std::env::var("MP_BUCKETS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| (self.workers.len().next_power_of_two()).max(1024));
        let bucket_cnt = bucket_cnt.next_power_of_two().max(64);

        // 临时目录选择顺序：
        // 1) 用户显式设置 TMPDIR
        // 2) Linux 下优先 /dev/shm（内存盘，明显加速 diff 桶文件读写）
        // 3) 系统默认 temp_dir
        let base_tmp = std::env::var("TMPDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let shm = PathBuf::from("/dev/shm");
                if shm.is_dir() {
                    shm
                } else {
                    std::env::temp_dir()
                }
            });
        let temp_root = base_tmp.join(format!(
            "lt_apply_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&temp_root)?;

        let worker_total = self.workers.len();
        let worker_dirs: Vec<PathBuf> = (0..worker_total)
            .map(|i| temp_root.join(format!("w{}", i)))
            .collect();

        let t = std::time::Instant::now();
        log_debug(
            "proc_pool",
            format!(
                "apply broadcast workers={} parts_len={}",
                worker_total,
                parts.len()
            ),
        );
        for (i, worker) in self.workers.iter_mut().enumerate() {
            let worker_temp = &worker_dirs[i];
            fs::create_dir_all(worker_temp)?;
            let req = WorkerRequest::ApplyMerge {
                parts: parts.to_vec(),
                replacement_id,
                recompute_each_step,
                return_diff: !low_mem,
                bucket_cnt,
                temp_dir: worker_temp.display().to_string(),
            };
            worker.send_only(&req)?;
        }

        for (i, worker) in self.workers.iter_mut().enumerate() {
            match worker.recv(&format!("worker:{}:apply", i))? {
                WorkerResponse::ApplyResult { .. } => {
                    log_debug(
                        "proc_pool",
                        format!(
                            "apply {}/{} done elapsed={:.2}s",
                            i + 1,
                            worker_total,
                            t.elapsed().as_secs_f32()
                        ),
                    );
                }
                other => {
                    return Err(anyhow::anyhow!("unexpected apply reply: {:?}", other));
                }
            }
        }

        Ok((temp_root, worker_dirs, bucket_cnt))
    }

    fn apply_merge(
        &mut self,
        parts: &[u32],
        replacement_id: u32,
        recompute_each_step: bool,
    ) -> Result<(HashMap<Ngram, Stat, RandomState>, HashMap<Ngram, Stat, RandomState>)> {
        let low_mem = std::env::var("MP_NO_DIFF").is_ok() || std::env::var("MP_LOW_MEM").is_ok();
        let (temp_root, worker_dirs, bucket_cnt) =
            self.apply_merge_prepare(parts, replacement_id, recompute_each_step)?;
        if low_mem {
            let _ = fs::remove_dir_all(&temp_root);
            return Ok((HashMap::with_hasher(RandomState::new()), HashMap::with_hasher(RandomState::new())));
        }

        // 读取 manifest（位图）：只打开存在的桶文件，避免 bucket_cnt*workers*2 次 open 尝试
        let mut manifests: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut manifest_ok = true;
        for wdir in &worker_dirs {
            if let Some(m) = read_diff_manifest(wdir, bucket_cnt) {
                manifests.push(m);
            } else {
                manifest_ok = false;
                break;
            }
        }

        let buckets_to_read: Vec<usize> = if manifest_ok {
            let bits_len = diff_bits_len(bucket_cnt);
            let mut union_bits = vec![0u8; bits_len];
            for (old_bits, new_bits) in &manifests {
                for i in 0..bits_len {
                    union_bits[i] |= old_bits[i] | new_bits[i];
                }
            }
            diff_bits_to_indices(&union_bits, bucket_cnt)
        } else {
            (0..bucket_cnt).collect()
        };

        // 并行按 bucket 归并：只处理“真的有 diff 的桶”，大幅降低 I/O 与 syscalls
        let (old_acc, new_acc) = buckets_to_read
            .into_par_iter()
            .map(|b| {
                let mut bucket_old = HashMap::with_hasher(RandomState::new());
                let mut bucket_new = HashMap::with_hasher(RandomState::new());
                for (wi, wdir) in worker_dirs.iter().enumerate() {
                    let (has_old, has_new) = if manifest_ok {
                        let (ref old_bits, ref new_bits) = manifests[wi];
                        (diff_bit_get(old_bits, b), diff_bit_get(new_bits, b))
                    } else {
                        (true, true)
                    };

                    if has_old {
                        let p_old = wdir.join(format!("old_{b}.bin"));
                        if let Ok(_) = read_bucket_file_into(&p_old, &mut bucket_old) {
                            let _ = fs::remove_file(&p_old);
                        }
                    }
                    if has_new {
                        let p_new = wdir.join(format!("new_{b}.bin"));
                        if let Ok(_) = read_bucket_file_into(&p_new, &mut bucket_new) {
                            let _ = fs::remove_file(&p_new);
                        }
                    }
                }
                (bucket_old, bucket_new)
            })
            .reduce(
                || (HashMap::with_hasher(RandomState::new()), HashMap::with_hasher(RandomState::new())),
                |mut acc, (bo, bn)| {
                    merge_stats_into(&mut acc.0, bo);
                    merge_stats_into(&mut acc.1, bn);
                    acc
                },
            );

        let _ = fs::remove_dir_all(&temp_root);

        Ok((old_acc, new_acc))
    }

    fn collect_entries(&mut self) -> Result<Vec<SeqEntry>> {
        let mut merged: HashMap<Vec<u32>, u32, RandomState> = HashMap::with_hasher(RandomState::new());
        // 同样先发请求再收，避免串行等待
        for worker in self.workers.iter_mut() {
            worker.send_only(&WorkerRequest::DumpEntries)?;
        }
        for worker in self.workers.iter_mut() {
            match worker.recv("worker:dump")? {
                WorkerResponse::Entries { entries } => {
                    for e in entries {
                        *merged.entry(e.tokens).or_insert(0) += e.freq;
                    }
                }
                other => {
                    return Err(anyhow::anyhow!("unexpected entries reply: {:?}", other));
                }
            }
        }
        Ok(merged
            .into_iter()
            .map(|(tokens, freq)| SeqEntry { tokens, freq })
            .collect())
    }

    fn shutdown(&mut self) {
        for worker in self.workers.iter_mut() {
            let _ = worker.send_only(&WorkerRequest::Shutdown);
        }
        for worker in self.workers.iter_mut() {
            let _ = worker.recv("worker:shutdown");
            let _ = worker.child.wait();
        }
    }
}

#[derive(Default)]
struct WorkerState {
    entries: Vec<SeqEntry>,
    n_values: Vec<usize>,
}

static WORKER_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

fn worker_pool() -> &'static rayon::ThreadPool {
    WORKER_POOL.get_or_init(|| {
        let threads = WorkerState::worker_threads();
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build worker rayon pool")
    })
}

impl WorkerState {
    /// 决定 worker 内 rayon 线程数；可用环境变量 WORKER_THREADS 覆盖
    fn worker_threads() -> usize {
        if let Ok(v) = std::env::var("WORKER_THREADS") {
            if let Ok(n) = v.parse::<usize>() {
                return n.max(1);
            }
        }
        num_cpus::get().max(1)
    }

    fn worker_chunk(entries: usize, threads: usize) -> usize {
        if let Ok(v) = std::env::var("WORKER_CHUNK") {
            if let Ok(n) = v.parse::<usize>() {
                return n.max(1024);
            }
        }
        let base = entries
            .checked_div(threads.saturating_mul(4).max(1))
            .unwrap_or(1024);
        base.clamp(8_192, 20_000)
    }

    fn compute_stats(&self) -> HashMap<Ngram, Stat, RandomState> {
        let t = std::time::Instant::now();
        let entry_cnt = self.entries.len();
        let threads = Self::worker_threads();
        let chunk = Self::worker_chunk(entry_cnt, threads);
        let chunk_total = (entry_cnt + chunk.saturating_sub(1)) / chunk.max(1);
        let progress = AtomicUsize::new(0);
        log_debug(
            "worker",
            format!(
                "stats start entries={} threads={} chunk={}",
                entry_cnt,
                threads,
                chunk
            ),
        );

        let pool = worker_pool();
        let res = pool.install(|| {
        self.entries
            .par_chunks(chunk)
            .map(|entries| {
                let done = progress.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                if done == 1 || done == chunk_total || done % 4 == 0 {
                    log_debug(
                        "worker",
                        format!(
                            "stats progress {}/{} elapsed={:.2}s",
                            done,
                            chunk_total,
                            t.elapsed().as_secs_f32()
                        ),
                    );
                }
                let mut local = HashMap::with_hasher(RandomState::new());
                for entry in entries {
                    let win_est =
                        LengthTokenizer::estimate_windows(entry.tokens.len(), &self.n_values);
                let mut local_map = HashMap::with_hasher(RandomState::new());
                    local_map.reserve(win_est.max(8));
                    let stats = LengthTokenizer::entry_stats_tokens(&entry.tokens, &self.n_values);
                    LengthTokenizer::accumulate_stats(&mut local_map, &stats, entry.freq, true, true);
                    merge_stats_into(&mut local, local_map);
                }
                local
            })
            .reduce(
                || HashMap::with_hasher(RandomState::new()),
                |mut acc, m| {
                    merge_stats_into(&mut acc, m);
                    acc
                },
            )
        });

        log_debug(
            "worker",
            format!(
                "stats done entries={} keys={} elapsed={:.2}s",
                entry_cnt,
                res.len(),
                t.elapsed().as_secs_f32()
            ),
        );
        res
    }
}

fn collect_starts(
    len: usize,
    positions: &[usize],
    n_vals: &[usize],
    plen: usize,
) -> HashMap<usize, Vec<usize>> {
    let mut map: HashMap<usize, Vec<usize>> = HashMap::with_capacity(n_vals.len());
    for &pos in positions {
        for &n in n_vals {
            let start_lo = pos.saturating_sub(n.saturating_sub(1));
            let start_hi = pos + plen.saturating_sub(1);
            let v = map.entry(n).or_insert_with(Vec::new);
            for s in start_lo..=start_hi {
                if s + n <= len {
                    v.push(s);
                }
            }
        }
    }
    for v in map.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    map
}

fn estimate_from_starts(starts: &HashMap<usize, Vec<usize>>) -> usize {
    starts.values().map(|v| v.len()).sum()
}

fn worker_apply_merge(
    state: &mut WorkerState,
    parts: &[u32],
    replacement_id: u32,
    n_vals: &[usize],
    return_diff: bool,
    bucket_cnt: usize,
    temp_dir: &Path,
) -> usize {
    #[derive(Default)]
    struct Buffers {
        positions: Vec<usize>,
        positions_new: Vec<usize>,
        old_local: HashMap<Ngram, Stat, RandomState>,
        new_local: HashMap<Ngram, Stat, RandomState>,
        tmp_ngram: [u32; 16],
    }

    /// 桶文件写入器：文件头写入 record_count(u64)，随后写入 `record_count` 个 bincode 记录。
    /// 这样主进程读取时不需要靠 `UnexpectedEof` 退出，减少错误路径开销。
    struct BucketWriter {
        w: BufWriter<File>,
        count: u64,
    }

    impl BucketWriter {
        fn create(path: &Path) -> Self {
            let mut w = BufWriter::new(File::create(path).expect("create bucket file"));
            // 预留 record_count，最终在 finalize() 回填
            w.write_all(&0u64.to_le_bytes())
                .expect("write bucket header");
            Self { w, count: 0 }
        }

        #[inline]
        fn write_record(&mut self, ng: &Ngram, st: &Stat) {
            bincode::serialize_into(&mut self.w, &(ng, st)).expect("write bucket record");
            self.count += 1;
        }

        fn finalize(&mut self) {
            let _ = self.w.flush();
            let f = self.w.get_mut();
            // 回填 record_count
            if f.seek(SeekFrom::Start(0)).is_ok() {
                let _ = f.write_all(&self.count.to_le_bytes());
                let _ = f.flush();
            }
        }
    }

    let plen = parts.len();

    let t_worker = std::time::Instant::now();
    // 消费 entries：避免大量 `Vec<u32>` clone，并允许原地改写 tokens。
    let mut entries = std::mem::take(&mut state.entries);
    let entry_cnt_before = entries.len();
    let threads = WorkerState::worker_threads();
    let chunk = WorkerState::worker_chunk(entry_cnt_before, threads);
    log_debug(
        "worker",
        format!(
            "apply start entries={} parts_len={} threads={} chunk={}",
            entry_cnt_before,
            plen,
            threads,
            chunk
        ),
    );

    let pool = worker_pool();

    // 准备桶文件（old/new），仅在 return_diff 时启用。
    // 关键优化：**不要预先创建 bucket_cnt*2 个空文件**，否则主进程会为大量空桶反复 open/读 EOF，
    // 严重浪费时间与 inode。改为“按需创建”：只有桶里真的有数据时才创建对应文件。
    let mut old_writers: Vec<Arc<Mutex<Option<BucketWriter>>>> = Vec::new();
    let mut new_writers: Vec<Arc<Mutex<Option<BucketWriter>>>> = Vec::new();
    let mut old_paths: Vec<PathBuf> = Vec::new();
    let mut new_paths: Vec<PathBuf> = Vec::new();
    if return_diff {
        fs::create_dir_all(temp_dir).ok();
        old_paths = (0..bucket_cnt)
            .map(|b| temp_dir.join(format!("old_{b}.bin")))
            .collect();
        new_paths = (0..bucket_cnt)
            .map(|b| temp_dir.join(format!("new_{b}.bin")))
            .collect();
        for _ in 0..bucket_cnt {
            old_writers.push(Arc::new(Mutex::new(None)));
            new_writers.push(Arc::new(Mutex::new(None)));
        }
    }

    let old_writers_shared = old_writers.clone();
    let new_writers_shared = new_writers.clone();

    // 分桶哈希必须在整个 apply 调用内保持一致：否则同一 Ngram 会跨桶分散，导致主进程重复合并/重复 push，
    // 严重拖慢并放大内存（之前的 OOM/慢很多就是这个原因之一）。
    let bucket_hasher = RandomState::new();

    let buckets: HashMap<Vec<u32>, u32, RandomState> = pool.install(|| {
        entries
            .par_chunks_mut(chunk)
            .map(|chunk_entries| {
                let mut chunk_old: HashMap<Ngram, Stat, RandomState> =
                    HashMap::with_hasher(RandomState::new());
                let mut chunk_new: HashMap<Ngram, Stat, RandomState> =
                    HashMap::with_hasher(RandomState::new());
                let mut chunk_buckets: HashMap<Vec<u32>, u32, RandomState> =
                    HashMap::with_hasher(RandomState::new());

                // 复用缓冲区，避免每条 entry 都重新分配 Vec/HashMap（apply 阶段的主要热点之一）
                let mut bufs = Buffers::default();

                for entry in chunk_entries.iter_mut() {
                    bufs.positions.clear();
                    bufs.positions_new.clear();
                    bufs.old_local.clear();
                    bufs.new_local.clear();

                    let freq = entry.freq;
                    let mut tokens: Vec<u32> = std::mem::take(&mut entry.tokens);
                    let len_old = tokens.len();

                    if plen == 0 || len_old < plen {
                        *chunk_buckets.entry(tokens).or_insert(0) += freq;
                        continue;
                    }

                    if return_diff {
                        // 匹配位置（不重叠）
                        let mut idx = 0usize;
                        while idx + plen <= len_old {
                            if tokens[idx..idx + plen] == parts[..] {
                                bufs.positions.push(idx);
                                idx += plen;
                            } else {
                                idx += 1;
                            }
                        }
                        if bufs.positions.is_empty() {
                            *chunk_buckets.entry(tokens).or_insert(0) += freq;
                            continue;
                        }

                        // 受影响窗口（旧）
                        let affected_old = collect_starts(len_old, &bufs.positions, n_vals, plen);

                        // 旧窗口统计
                        let old_cap = estimate_from_starts(&affected_old);
                        if bufs.old_local.capacity() < old_cap {
                            bufs.old_local.reserve(
                                old_cap.saturating_sub(bufs.old_local.capacity()),
                            );
                        }
                        for (&n, starts) in &affected_old {
                            for &s in starts {
                                if s + n <= len_old {
                                    bufs.tmp_ngram[..n]
                                        .copy_from_slice(&tokens[s..s + n]);
                                    let ng = SmallVec::<[u32; 9]>::from_slice(
                                        &bufs.tmp_ngram[..n],
                                    );
                                    let e = bufs.old_local.entry(ng).or_default();
                                    e.freq += 1;
                                    e.score += n.saturating_sub(1) as u64;
                                }
                            }
                        }
                        if !bufs.old_local.is_empty() {
                            LengthTokenizer::accumulate_stats(
                                &mut chunk_old,
                                &bufs.old_local,
                                freq,
                                true,
                                false,
                            );
                        }

                        // 原地构造新 tokens（用 positions 引导，避免再次 slice 比较）
                        let mut p_iter = bufs.positions.iter().copied().peekable();
                        let mut read = 0usize;
                        let mut write = 0usize;
                        while read < len_old {
                            if let Some(&p) = p_iter.peek() {
                                if read == p {
                                    bufs.positions_new.push(write);
                                    tokens[write] = replacement_id;
                                    write += 1;
                                    read += plen;
                                    p_iter.next();
                                    continue;
                                }
                            }
                            if write != read {
                                tokens[write] = tokens[read];
                            }
                            write += 1;
                            read += 1;
                        }
                        tokens.truncate(write);

                        let len_new = tokens.len();
                        let affected_new =
                            collect_starts(len_new, &bufs.positions_new, n_vals, 1);

                        // 新窗口统计
                        let new_cap = estimate_from_starts(&affected_new);
                        if bufs.new_local.capacity() < new_cap {
                            bufs.new_local.reserve(
                                new_cap.saturating_sub(bufs.new_local.capacity()),
                            );
                        }
                        for (&n, starts) in &affected_new {
                            for &s in starts {
                                if s + n <= len_new {
                                    bufs.tmp_ngram[..n]
                                        .copy_from_slice(&tokens[s..s + n]);
                                    let ng = SmallVec::<[u32; 9]>::from_slice(
                                        &bufs.tmp_ngram[..n],
                                    );
                                    let e = bufs.new_local.entry(ng).or_default();
                                    e.freq += 1;
                                    e.score += n.saturating_sub(1) as u64;
                                }
                            }
                        }
                        if !bufs.new_local.is_empty() {
                            LengthTokenizer::accumulate_stats(
                                &mut chunk_new,
                                &bufs.new_local,
                                freq,
                                true,
                                false,
                            );
                        }

                        *chunk_buckets.entry(tokens).or_insert(0) += freq;
                    } else {
                        // 低内存/不返回 diff：仅原地替换 tokens（不构造 positions/局部统计）
                        let mut read = 0usize;
                        let mut write = 0usize;
                        while read < len_old {
                            if read + plen <= len_old && tokens[read..read + plen] == parts[..] {
                                tokens[write] = replacement_id;
                                write += 1;
                                read += plen;
                            } else {
                                if write != read {
                                    tokens[write] = tokens[read];
                                }
                                write += 1;
                                read += 1;
                            }
                        }
                        tokens.truncate(write);
                        *chunk_buckets.entry(tokens).or_insert(0) += freq;
                    }
                }

                if return_diff {
                    // 先按桶聚合，减少锁竞争与序列化调用次数
                    let mut bucket_old: HashMap<usize, Vec<(Ngram, Stat)>> = HashMap::new();
                    let mut bucket_new: HashMap<usize, Vec<(Ngram, Stat)>> = HashMap::new();

                    for (ng, st) in chunk_old.drain() {
                        use std::hash::{BuildHasher, Hash};
                        let mut h = bucket_hasher.build_hasher();
                        ng.hash(&mut h);
                        let b = (h.finish() as usize) & (bucket_cnt - 1);
                        bucket_old.entry(b).or_default().push((ng, st));
                    }
                    for (ng, st) in chunk_new.drain() {
                        use std::hash::{BuildHasher, Hash};
                        let mut h = bucket_hasher.build_hasher();
                        ng.hash(&mut h);
                        let b = (h.finish() as usize) & (bucket_cnt - 1);
                        bucket_new.entry(b).or_default().push((ng, st));
                    }

                    for (b, items) in bucket_old {
                        if let Ok(mut guard) = old_writers_shared[b].lock() {
                            if guard.is_none() {
                                *guard = Some(BucketWriter::create(&old_paths[b]));
                            }
                            let w = guard.as_mut().expect("old writer present");
                            for (ng, st) in items {
                                w.write_record(&ng, &st);
                            }
                        }
                    }
                    for (b, items) in bucket_new {
                        if let Ok(mut guard) = new_writers_shared[b].lock() {
                            if guard.is_none() {
                                *guard = Some(BucketWriter::create(&new_paths[b]));
                            }
                            let w = guard.as_mut().expect("new writer present");
                            for (ng, st) in items {
                                w.write_record(&ng, &st);
                            }
                        }
                    }
                }

                chunk_buckets
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
    });

    let mut diff_files_present: usize = 0;
    if return_diff {
        // finalize bucket files + 写 manifest：主进程只打开存在的桶，显著减少 open/ENOENT
        let bits_len = diff_bits_len(bucket_cnt);
        let mut old_bits = vec![0u8; bits_len];
        let mut new_bits = vec![0u8; bits_len];

        for (b, w) in old_writers.iter().enumerate() {
            if let Ok(mut guard) = w.lock() {
                if let Some(ref mut bw) = *guard {
                    bw.finalize();
                    diff_bit_set(&mut old_bits, b);
                }
            }
        }
        for (b, w) in new_writers.iter().enumerate() {
            if let Ok(mut guard) = w.lock() {
                if let Some(ref mut bw) = *guard {
                    bw.finalize();
                    diff_bit_set(&mut new_bits, b);
                }
            }
        }
        diff_files_present = old_bits
            .iter()
            .map(|b| b.count_ones() as usize)
            .sum::<usize>()
            + new_bits
                .iter()
                .map(|b| b.count_ones() as usize)
                .sum::<usize>();
        write_diff_manifest(temp_dir, bucket_cnt, &old_bits, &new_bits)
            .expect("write diff manifest");
    }

    state.entries = buckets
        .into_iter()
        .map(|(tokens, freq)| SeqEntry { tokens, freq })
        .collect();

    log_debug(
        "worker",
        format!(
            "apply done entries={}→{} elapsed={:.2}s (diff_files={})",
            entry_cnt_before,
            state.entries.len(),
            t_worker.elapsed().as_secs_f32(),
            if return_diff { diff_files_present } else { 0 }
        ),
    );

    state.entries.len()
}

fn worker_loop() -> Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());
    let mut state: Option<WorkerState> = None;
    log_debug("worker", format!("ready pid={}", std::process::id()));

    loop {
        let req: WorkerRequest = match read_msg_tagged(&mut reader, "worker:req") {
            Ok(v) => v,
            Err(e) => {
                // 无法读取更多数据，退出
                return Err(e);
            }
        };

        match req {
            WorkerRequest::Init { entries, n_values } => {
                state = Some(WorkerState { entries, n_values });
                write_msg(&mut writer, &WorkerResponse::Ack)?;
            }
            WorkerRequest::Preprocess { lines } => {
                let mut local: HashMap<Vec<String>, u32, RandomState> =
                    HashMap::with_hasher(RandomState::new());
                for sentence in lines {
                    let encoded = LengthTokenizer::encode_sentence_str(&sentence);
                    *local.entry(encoded).or_insert(0) += 1;
                }
                write_msg(&mut writer, &WorkerResponse::PreStats { stats: counts_map_to_vec(local) })?;
            }
            WorkerRequest::ComputeStats => {
                if let Some(st) = state.as_ref() {
                    let stats = st.compute_stats();
                    write_msg(&mut writer, &WorkerResponse::Stats { stats })?;
                } else {
                    write_msg(
                        &mut writer,
                        &WorkerResponse::Error("worker not initialized".to_string()),
                    )?;
                }
            }
            WorkerRequest::ApplyMerge { parts, replacement_id, recompute_each_step: _, return_diff, bucket_cnt, temp_dir } => {
                if let Some(st) = state.as_mut() {
                    let n_vals = st.n_values.clone();
                    let entry_count = worker_apply_merge(
                        st,
                        &parts,
                        replacement_id,
                        &n_vals,
                        return_diff,
                        bucket_cnt,
                        &Path::new(&temp_dir),
                    );
                    write_msg(
                        &mut writer,
                        &WorkerResponse::ApplyResult {
                            entry_count,
                        },
                    )?;
                } else {
                    write_msg(
                        &mut writer,
                        &WorkerResponse::Error("worker not initialized".to_string()),
                    )?;
                }
            }
            WorkerRequest::DumpEntries => {
                if let Some(st) = state.as_ref() {
                    write_msg(
                        &mut writer,
                        &WorkerResponse::Entries {
                            entries: st.entries.clone(),
                        },
                    )?;
                } else {
                    write_msg(
                        &mut writer,
                        &WorkerResponse::Error("worker not initialized".to_string()),
                    )?;
                }
            }
            WorkerRequest::Shutdown => {
                write_msg(&mut writer, &WorkerResponse::Ack)?;
                break;
            }
        }
    }
    Ok(())
}

/// 由 main.rs 调用，用于 worker 进程入口
pub fn run_worker() -> Result<()> {
    worker_loop()
}

// ===== HuggingFace 导出（供 CLI / PyO3 复用）=====
pub mod hf_export;

// ===== PyO3 Python 扩展（maturin build）=====
// 说明：
// - 仅在启用 feature=python 时编译
// - 供 HuggingFace remote-code tokenizer 优先调用以获得接近原生 Rust 的分词速度
#[cfg(feature = "python")]
mod pyo3_ext;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_default_uses_dp_trie() {
        // 使用仓库自带的 smoke 表，避免大文件与长时间训练
        let table = Path::new(env!("CARGO_MANIFEST_DIR")).join("token_table_smoke.json");
        let cfg = TokenizerConfig::default();
        let tk = LengthTokenizer::load(&table, cfg).expect("load smoke token table");

        // DP tokenize 会初始化 trie；BPE tokenize 不会
        assert!(tk.token_trie.get().is_none());
        let _ = tk.tokenize("hello world");
        assert!(tk.token_trie.get().is_some());
    }
}

