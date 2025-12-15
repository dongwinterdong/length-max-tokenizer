use serde::de::{self, DeserializeSeed, MapAccess, Visitor};
use serde::{Deserializer};
use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::time::Instant;

const MAX_MERGES: usize = 50_000;
const MAX_VOCAB: usize = 1_000;
const SRC_PATH: &str = "/home/arxiv_code/tokenizers_rust/token_table_safe.json";
const DST_PATH: &str = "/home/arxiv_code/tokenizers_rust/token_table_sampled.json";
const STOP_PREFIX: &str = "__STOP_SAMPLE__";

fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    println!("Sampling {} -> {}", SRC_PATH, DST_PATH);

    let f_in = File::open(SRC_PATH)?;
    let reader = BufReader::with_capacity(16 * 1024 * 1024, f_in);
    let mut de = serde_json::Deserializer::from_reader(reader);

    let f_out = File::create(DST_PATH)?;
    let mut writer = BufWriter::new(f_out);

    writer.write_all(b"{\n")?;

    let mut stats = SamplingStats {
        merges: 0,
        vocab: 0,
    };

    // 使用 Seed 来反序列化，这样可以传入可变引用
    let seed = SamplerSeed {
        writer: &mut writer,
        stats: &mut stats,
    };

    // 采样只需要“前 N 条”，不需要把整个 2.8GB 文件解析到底。
    // 通过在 Visitor 内部触发一个带特殊前缀的自定义错误来“主动停止”，这里捕获并视为成功。
    match seed.deserialize(&mut de) {
        Ok(()) => {}
        Err(e) => {
            let msg = e.to_string();
            if !msg.starts_with(STOP_PREFIX) {
                return Err(e.into());
            }
        }
    }

    writer.write_all(b"\n}\n")?;
    writer.flush()?;

    println!(
        "Done! Merges: {}, Vocab: {}. Time: {:.2}s",
        stats.merges,
        stats.vocab,
        start.elapsed().as_secs_f32()
    );
    Ok(())
}

struct SamplingStats {
    merges: usize,
    vocab: usize,
}

struct SamplerSeed<'a, W: Write> {
    writer: &'a mut W,
    stats: &'a mut SamplingStats,
}

impl<'de, 'a, W: Write> DeserializeSeed<'de> for SamplerSeed<'a, W> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(SamplerVisitor {
            writer: self.writer,
            stats: self.stats,
            first_key: true,
        })
    }
}

struct SamplerVisitor<'a, W: Write> {
    writer: &'a mut W,
    stats: &'a mut SamplingStats,
    first_key: bool,
}

impl<'de, 'a, W: Write> Visitor<'de> for SamplerVisitor<'a, W> {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "root object")
    }

    fn visit_map<A>(mut self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        while let Some(key) = map.next_key::<String>()? {
            if !self.first_key {
                self.writer.write_all(b",\n").map_err(de::Error::custom)?;
            }
            self.first_key = false;

            match key.as_str() {
                "merges" => {
                    self.writer
                        .write_all(b"  \"merges\": [")
                        .map_err(de::Error::custom)?;
                    
                    map.next_value_seed(MergesSampler {
                        writer: self.writer,
                        count: &mut self.stats.merges,
                    })?;

                    self.writer.write_all(b"\n  ]").map_err(de::Error::custom)?;
                }
                "vocab" => {
                    self.writer
                        .write_all(b"  \"vocab\": {")
                        .map_err(de::Error::custom)?;

                    map.next_value_seed(VocabSampler {
                        writer: self.writer,
                        count: &mut self.stats.vocab,
                    })?;

                    self.writer.write_all(b"\n  }").map_err(de::Error::custom)?;
                }
                _ => {
                    let _ = map.next_value::<de::IgnoredAny>()?;
                }
            }
        }
        Ok(())
    }
}

// --- Merges ---

struct MergesSampler<'a, W: Write> {
    writer: &'a mut W,
    count: &'a mut usize,
}

impl<'de, 'a, W: Write> DeserializeSeed<'de> for MergesSampler<'a, W> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(self)
    }
}

impl<'de, 'a, W: Write> Visitor<'de> for MergesSampler<'a, W> {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "merges array")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let mut first = true;
        loop {
            if *self.count < MAX_MERGES {
                let Some(item) = seq.next_element::<serde_json::Value>()? else {
                    break;
                };
                if !first {
                    self.writer.write_all(b",").map_err(de::Error::custom)?;
                }
                self.writer.write_all(b"\n    ").map_err(de::Error::custom)?;
                serde_json::to_writer(&mut *self.writer, &item).map_err(de::Error::custom)?;
                first = false;
                *self.count += 1;

                if *self.count % 10000 == 0 {
                    print!("\rSampling merges: {}", self.count);
                    std::io::stdout().flush().ok();
                }
            } else {
                // 超过上限后只消费输入，不构造 Value，减少分配
                let Some(_ignored) = seq.next_element::<de::IgnoredAny>()? else {
                    break;
                };
            }
        }
        println!();
        Ok(())
    }
}

// --- Vocab ---

struct VocabSampler<'a, W: Write> {
    writer: &'a mut W,
    count: &'a mut usize,
}

impl<'de, 'a, W: Write> DeserializeSeed<'de> for VocabSampler<'a, W> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(self)
    }
}

impl<'de, 'a, W: Write> Visitor<'de> for VocabSampler<'a, W> {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "vocab map")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut first = true;
        while let Some(key) = map.next_key::<String>()? {
            let val: u32 = map.next_value()?;
            
            if *self.count < MAX_VOCAB {
                if !first {
                    self.writer.write_all(b",").map_err(de::Error::custom)?;
                }
                self.writer.write_all(b"\n    ").map_err(de::Error::custom)?;
                
                serde_json::to_writer(&mut *self.writer, &key).map_err(de::Error::custom)?;
                write!(self.writer, ": {}", val).map_err(de::Error::custom)?;
                
                first = false;
                *self.count += 1;

                if *self.count % 100 == 0 {
                    print!("\rSampling vocab: {}", self.count);
                    std::io::stdout().flush().ok();
                }

                // vocab 采满后立即关闭 vocab 对象并停止整个反序列化流程（不再解析剩余 2.8GB）。
                if *self.count >= MAX_VOCAB {
                    self.writer.write_all(b"\n  }").map_err(de::Error::custom)?;
                    return Err(de::Error::custom(STOP_PREFIX));
                }
            }
        }
        println!();
        Ok(())
    }
}
