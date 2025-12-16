use anyhow::Result;
use hashbrown::HashSet;
use serde::de::{self, DeserializeSeed, MapAccess, Visitor};
use serde::Deserialize;
use std::fmt;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

const END_TOKEN: &str = "Ġ";

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

    // 统一复用库内导出逻辑（会写出 vocab/config/special_tokens/remote-code/README）
    let vocab = length_tokenizer::hf_export::build_vocab(token_set.into_iter().collect::<Vec<_>>());
    eprintln!("[export] vocab_size={}", vocab.len());
    length_tokenizer::hf_export::write_hf_tokenizer_dir(&out_dir, &vocab)?;

    eprintln!("[export] done: {:?}", out_dir);
    Ok(())
}


