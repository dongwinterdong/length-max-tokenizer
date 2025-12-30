### Length-MAX Tokenizer

#### 1) Install

```bash
pip install length-tokenizer-rs
```

If your corpus is parquet (or you want streaming reads via pyarrow):

```bash
pip install pyarrow
```

#### 2) Train a vocab from your corpus (export a local tokenizer directory)

After training you will get a directory (e.g. `./tokenizer_out/`) containing:
`vocab.json` / `tokenizer_config.json` / `special_tokens_map.json` / `tokenization_length_tokenizer.py` / `README.md`

##### 2.1 Text corpus (one sentence per line)

```python
from length_tokenizer_rs import train_to_hf

train_to_hf(
    corpus_file="corpus.txt",   # one sentence per line
    out_dir="./tokenizer_out",
    num_merges=50000,
    aim_token_num=20000,
    n_max=6,
    num_workers=8,
    multi_process=False,
)
```

##### 2.2 Parquet corpus (streaming read via pyarrow)

```python
from length_tokenizer_rs import train_to_hf_parquet

train_to_hf_parquet(
    parquet_path="/path/to/parquet_dir_or_file",
    out_dir="./tokenizer_out",
    text_column="text",
    max_docs=0,
    batch_size=8192,
    recursive=True,
    num_merges=50000,
    aim_token_num=20000,
    n_max=6,
    num_workers=8,
    multi_process=False,
    chunk_size=4096,
)
```

#### 3) Tokenize your corpus with the new vocab → write ids (data prep)

The examples below write `ids.txt`: one sample per line, space-separated token ids (high throughput: Rust `DpTokenizer.encode_batch()`).

##### 3.1 Text corpus (one sentence per line) → ids.txt

```python
import json
from pathlib import Path

from length_tokenizer_rs import DpTokenizer

TOKENIZER_DIR = Path("./tokenizer_out")
VOCAB = TOKENIZER_DIR / "vocab.json"
dp = DpTokenizer(str(VOCAB), "<unk>")

# Optional: add BOS/EOS (adjust to your training pipeline)
vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
bos = vocab.get("<s>")
eos = vocab.get("</s>")

IN_TXT = Path("corpus.txt")
OUT_IDS = Path("corpus.ids.txt")

BATCH = 256
buf = []
with IN_TXT.open("r", encoding="utf-8", errors="ignore") as r, OUT_IDS.open("w", encoding="utf-8") as w:
    for line in r:
        s = line.strip()
        if not s:
            continue
        buf.append(s)
        if len(buf) >= BATCH:
            for ids in dp.encode_batch(buf):
                if bos is not None:
                    w.write(str(int(bos)) + " ")
                w.write(" ".join(str(int(x)) for x in ids))
                if eos is not None:
                    w.write(" " + str(int(eos)))
                w.write("\n")
            buf.clear()
    if buf:
        for ids in dp.encode_batch(buf):
            if bos is not None:
                w.write(str(int(bos)) + " ")
            w.write(" ".join(str(int(x)) for x in ids))
            if eos is not None:
                w.write(" " + str(int(eos)))
            w.write("\n")
```

##### 3.2 Parquet corpus (streaming read) → ids.txt

```python
import json
from pathlib import Path

import pyarrow.dataset as ds
from length_tokenizer_rs import DpTokenizer

PARQUET = "/path/to/parquet_dir_or_file"
TEXT_COL = "text"

TOKENIZER_DIR = Path("./tokenizer_out")
VOCAB = TOKENIZER_DIR / "vocab.json"
dp = DpTokenizer(str(VOCAB), "<unk>")

vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
bos = vocab.get("<s>")
eos = vocab.get("</s>")

OUT_IDS = Path("parquet.ids.txt")
BATCH = 256
buf = []

dataset = ds.dataset(PARQUET, format="parquet")
scanner = dataset.scanner(columns=[TEXT_COL], batch_size=8192, use_threads=True)

with OUT_IDS.open("w", encoding="utf-8") as w:
    for batch in scanner.to_batches():
        col = batch.column(0)
        for s in col.to_pylist():
            if not s or not str(s).strip():
                continue
            buf.append(str(s))
            if len(buf) >= BATCH:
                for ids in dp.encode_batch(buf):
                    if bos is not None:
                        w.write(str(int(bos)) + " ")
                    w.write(" ".join(str(int(x)) for x in ids))
                    if eos is not None:
                        w.write(" " + str(int(eos)))
                    w.write("\n")
                buf.clear()
    if buf:
        for ids in dp.encode_batch(buf):
            if bos is not None:
                w.write(str(int(bos)) + " ")
            w.write(" ".join(str(int(x)) for x in ids))
            if eos is not None:
                w.write(" " + str(int(eos)))
            w.write("\n")
```

#### 4) Load the tokenizer for training (local directory)

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("./tokenizer_out", trust_remote_code=True)
assert getattr(tok, "_rust", None) is not None, "Rust extension not active"
```
