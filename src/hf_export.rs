use anyhow::Result;
use hashbrown::{HashMap, HashSet};
use serde_json::json;
use std::cmp::Ordering;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::LengthTokenizer;

// HF 常用 special tokens（保持与 export_hf_tokenizer.rs 一致）
pub const UNK: &str = "<unk>";
pub const PAD: &str = "<pad>";
pub const BOS: &str = "<s>";
pub const EOS: &str = "</s>";
pub const MASK: &str = "<mask>";

// 直接复用仓库内 remote-code 实现与 README（编译期嵌入，wheel 里也能导出）
const REMOTE_CODE_PY: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/hf_tokenizer_out/tokenization_length_tokenizer.py"));
const OUT_README_MD: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/hf_tokenizer_out/README.md"));

fn write_json_pretty(path: &Path, value: &serde_json::Value) -> Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    serde_json::to_writer_pretty(&mut w, value)?;
    w.write_all(b"\n")?;
    w.flush()?;
    Ok(())
}

fn is_special(s: &str) -> bool {
    matches!(s, UNK | PAD | BOS | EOS | MASK)
}

fn special_rank(s: &str) -> u32 {
    match s {
        UNK => 0,
        PAD => 1,
        BOS => 2,
        EOS => 3,
        MASK => 4,
        _ => 100,
    }
}

/// 从一个 token 集合构建 `vocab.json`（token -> id），并保持确定性：
/// - special tokens 固定放在最前（UNK/PAD/BOS/EOS/MASK）
/// - 其余 token 按字典序排序
pub fn build_vocab(tokens: impl IntoIterator<Item = String>) -> HashMap<String, u32> {
    let mut set: HashSet<String> = HashSet::new();
    for t in tokens {
        set.insert(t);
    }
    for s in [UNK, PAD, BOS, EOS, MASK] {
        set.insert(s.to_string());
    }

    let mut toks: Vec<String> = set.into_iter().collect();
    toks.sort_by(|a, b| match (is_special(a), is_special(b)) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (true, true) => special_rank(a).cmp(&special_rank(b)),
        (false, false) => a.cmp(b),
    });

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, t) in toks.iter().enumerate() {
        vocab.insert(t.clone(), i as u32);
    }
    vocab
}

/// 写出一个可直接上传 HuggingFace Hub 的 tokenizer 目录（remote code）。
///
/// 生成文件：
/// - vocab.json
/// - tokenizer_config.json
/// - special_tokens_map.json
/// - tokenization_length_tokenizer.py
/// - README.md
pub fn write_hf_tokenizer_dir(out_dir: &Path, vocab: &HashMap<String, u32>) -> Result<()> {
    fs::create_dir_all(out_dir)?;

    // vocab.json（Transformers slow tokenizer 常用格式）
    let vocab_v = serde_json::to_value(vocab)?;
    write_json_pretty(&out_dir.join("vocab.json"), &vocab_v)?;

    // tokenizer_config.json（remote code）
    let tokenizer_config = json!({
        "tokenizer_class": "LengthTokenizer",
        "auto_map": {
            "AutoTokenizer": ["tokenization_length_tokenizer.LengthTokenizer", null]
        },
        "model_max_length": 1000000000,
        "unk_token": UNK,
        "pad_token": PAD,
        "bos_token": BOS,
        "eos_token": EOS,
        "mask_token": MASK
    });
    write_json_pretty(&out_dir.join("tokenizer_config.json"), &tokenizer_config)?;

    // special_tokens_map.json
    let special_tokens_map = json!({
        "unk_token": UNK,
        "pad_token": PAD,
        "bos_token": BOS,
        "eos_token": EOS,
        "mask_token": MASK
    });
    write_json_pretty(&out_dir.join("special_tokens_map.json"), &special_tokens_map)?;

    // remote code
    fs::write(out_dir.join("tokenization_length_tokenizer.py"), REMOTE_CODE_PY)?;
    if !REMOTE_CODE_PY.ends_with('\n') {
        fs::OpenOptions::new()
            .append(true)
            .open(out_dir.join("tokenization_length_tokenizer.py"))?
            .write_all(b"\n")?;
    }

    // README
    fs::write(out_dir.join("README.md"), OUT_README_MD)?;
    if !OUT_README_MD.ends_with('\n') {
        fs::OpenOptions::new()
            .append(true)
            .open(out_dir.join("README.md"))?
            .write_all(b"\n")?;
    }

    Ok(())
}

/// 从训练好的 `LengthTokenizer` 直接导出 HF tokenizer 目录（不依赖 token_table.json）。
pub fn export_from_trained(tk: &LengthTokenizer, out_dir: &Path) -> Result<HashMap<String, u32>> {
    // interner 内已包含训练过程中见过的所有 token（含单字符与 END_TOKEN）
    let tokens = tk.interner.id_to_token.clone();
    let vocab = build_vocab(tokens);
    write_hf_tokenizer_dir(out_dir, &vocab)?;
    Ok(vocab)
}


