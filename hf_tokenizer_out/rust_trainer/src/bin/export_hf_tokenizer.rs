use anyhow::Result;
use hashbrown::{HashMap, HashSet};
use serde::de::{self, DeserializeSeed, MapAccess, Visitor};
use serde::Deserialize;
use serde_json::json;
use std::cmp::Ordering;
use std::fmt;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const END_TOKEN: &str = "Ġ";

// HF 常用 special tokens（你也可以按自己模型偏好改）
const UNK: &str = "<unk>";
const PAD: &str = "<pad>";
const BOS: &str = "<s>";
const EOS: &str = "</s>";
const MASK: &str = "<mask>";

#[derive(Debug, Clone, Deserialize)]
struct MergeRuleLite {
    parts: Vec<String>,
    replacement: String,
}

fn load_merges_and_vocab_tokens(path: &Path, token_set: &mut HashSet<String>) -> Result<Vec<MergeRuleLite>> {
    let f = File::open(path)?;
    let reader = BufReader::with_capacity(16 * 1024 * 1024, f);
    let mut de = serde_json::Deserializer::from_reader(reader);

    let mut merges: Vec<MergeRuleLite> = Vec::new();
    let seed = RootSeed {
        merges: &mut merges,
        token_set,
    };
    seed.deserialize(&mut de).map_err(|e| anyhow::anyhow!(e))?;
    Ok(merges)
}

struct RootSeed<'a> {
    merges: &'a mut Vec<MergeRuleLite>,
    token_set: &'a mut HashSet<String>,
}

impl<'de, 'a> DeserializeSeed<'de> for RootSeed<'a> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(RootVisitor {
            merges: self.merges,
            token_set: self.token_set,
        })
    }
}

struct RootVisitor<'a> {
    merges: &'a mut Vec<MergeRuleLite>,
    token_set: &'a mut HashSet<String>,
}

struct VocabSeed<'a> {
    token_set: &'a mut HashSet<String>,
}

impl<'de, 'a> DeserializeSeed<'de> for VocabSeed<'a> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(VocabVisitor {
            token_set: self.token_set,
        })
    }
}

struct VocabVisitor<'a> {
    token_set: &'a mut HashSet<String>,
}

impl<'de, 'a> Visitor<'de> for VocabVisitor<'a> {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "token table vocab map")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        while let Some(key) = map.next_key::<String>()? {
            // vocab key is a token sequence joined by whitespace (e.g. "helloĠ w o r l d Ġ")
            for tok in key.split_whitespace() {
                if !self.token_set.contains(tok) {
                    self.token_set.insert(tok.to_string());
                }
            }
            let _ = map.next_value::<de::IgnoredAny>()?;
        }
        Ok(())
    }
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
                    // collect tokens from merges (parts + replacement)
                    for m in self.merges.iter() {
                        for p in m.parts.iter() {
                            self.token_set.insert(p.clone());
                        }
                        self.token_set.insert(m.replacement.clone());
                    }
                }
                "vocab" => {
                    // Stream over vocab keys (ignore values) to ensure we keep base character tokens.
                    map.next_value_seed(VocabSeed {
                        token_set: self.token_set,
                    })?;
                }
                _ => {
                    let _ = map.next_value::<de::IgnoredAny>()?;
                }
            }
        }
        Ok(())
    }
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    serde_json::to_writer_pretty(&mut w, value)?;
    w.write_all(b"\n")?;
    w.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    // 用法：
    // cargo run --release --bin export_hf_tokenizer -- <token_table.json> <out_dir>
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "用法: {} <token_table.json> <out_dir>\n示例: {} token_table_safe.json hf_tokenizer",
            args[0], args[0]
        );
        std::process::exit(2);
    }
    let table_path = PathBuf::from(&args[1]);
    let out_dir = PathBuf::from(&args[2]);
    fs::create_dir_all(&out_dir)?;

    eprintln!("[export] 读取 merges + vocab keys（流式，不加载大 vocab）: {:?}", table_path);
    let mut token_set: HashSet<String> = HashSet::new();
    token_set.insert(END_TOKEN.to_string());
    let merges = load_merges_and_vocab_tokens(&table_path, &mut token_set)?;
    eprintln!("[export] merges={}", merges.len());

    // 为了 DP 兜底，补齐“单字符 token”（至少覆盖 merges 中出现过的字符）
    let mut single_chars: HashSet<String> = HashSet::new();
    for t in token_set.iter() {
        for ch in t.chars() {
            single_chars.insert(ch.to_string());
        }
    }
    for c in single_chars {
        token_set.insert(c);
    }

    // 追加 HF 常用 special tokens
    for s in [UNK, PAD, BOS, EOS, MASK] {
        token_set.insert(s.to_string());
    }

    // 确定性排序并分配 id：special tokens 固定在前面，其余按字典序
    let mut tokens: Vec<String> = token_set.into_iter().collect();
    tokens.sort_by(|a, b| a.cmp(b));

    // 把 special tokens 放到最前（保持顺序）
    fn is_special(s: &str) -> bool {
        matches!(s, UNK | PAD | BOS | EOS | MASK)
    }
    tokens.sort_by(|a, b| match (is_special(a), is_special(b)) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (true, true) => {
            // 固定顺序
            let rank = |x: &str| match x {
                UNK => 0,
                PAD => 1,
                BOS => 2,
                EOS => 3,
                MASK => 4,
                _ => 100,
            };
            rank(a).cmp(&rank(b))
        }
        (false, false) => a.cmp(b),
    });

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, t) in tokens.iter().enumerate() {
        vocab.insert(t.clone(), i as u32);
    }
    eprintln!("[export] vocab_size={}", vocab.len());

    // 写 vocab.json（Transformers slow tokenizer 常用格式）
    write_json(&out_dir.join("vocab.json"), &serde_json::to_value(&vocab)?)?;

    // 写 tokenizer_config.json（remote code）
    // 说明：用户加载时需要 trust_remote_code=True
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
    write_json(&out_dir.join("tokenizer_config.json"), &tokenizer_config)?;

    // 写 special_tokens_map.json
    let special_tokens_map = json!({
        "unk_token": UNK,
        "pad_token": PAD,
        "bos_token": BOS,
        "eos_token": EOS,
        "mask_token": MASK
    });
    write_json(&out_dir.join("special_tokens_map.json"), &special_tokens_map)?;

    // 写 README.md（简短说明）
    let readme = r#"
### LengthTokenizer（DP 最小 token / 最低 TPC）

这是一个用于 Hugging Face Transformers 的 **自定义 tokenizer**（remote code），分词策略为：
- 先按空白切词，并把每个词变换为：`词 + "Ġ"`（与本仓库 Rust 训练口径一致）
- 在给定 vocab 下，用 Trie + DP 找到 **token 数最少** 的全局最优切分（TPC 最低）

#### 使用方法（Transformers）

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("YOUR_USER/YOUR_REPO", trust_remote_code=True)
print(tok.tokenize("hello world"))
```

> 注意：这是 remote code，需要 `trust_remote_code=True`。
"#;
    fs::write(out_dir.join("README.md"), readme.trim_start())?;

    eprintln!("[export] done: {:?}", out_dir);
    Ok(())
}


