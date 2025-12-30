"""
HuggingFace remote-code tokenizer for LengthTokenizer.

特点：
- 复刻本仓库 Rust 端的 normalize 逻辑：按空白 split，每个词后追加 'Ġ'（END_TOKEN）
- 使用 Trie + DP 做“全局最少 token 数”的切分（最小 TPC）

使用：
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("user/repo", trust_remote_code=True)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer


END_TOKEN = "Ġ"

_RUST_AVAILABLE = False
_RustDpTokenizer = None
try:
    # 高性能路径：如果用户安装了 wheel（maturin/PyO3），优先使用 Rust DP 分词
    from length_tokenizer_rs import DpTokenizer as _RustDpTokenizer  # type: ignore

    _RUST_AVAILABLE = True
except Exception:
    _RUST_AVAILABLE = False
    _RustDpTokenizer = None


class _TrieNode:
    __slots__ = ("next", "term_id")

    def __init__(self) -> None:
        self.next: Dict[str, int] = {}
        self.term_id: Optional[int] = None


class _TokenTrie:
    def __init__(self, vocab: Dict[str, int]) -> None:
        self.nodes: List[_TrieNode] = [_TrieNode()]
        forbid_end_inner = os.environ.get("LENGTH_TOKENIZER_FORBID_END_INNER") is not None
        cross_word_whole_only = (not forbid_end_inner) and (
            os.environ.get("LENGTH_TOKENIZER_CROSS_WORD_WHOLE_ONLY") is not None
        )
        # token ids that must start at a word boundary (i==0 or previous char is END_TOKEN)
        self._require_word_start: set[int] = set()
        for tok, tid in vocab.items():
            end_inner = END_TOKEN in tok[:-1]
            if forbid_end_inner and end_inner:
                continue
            if cross_word_whole_only and end_inner:
                # Cross-word token must end with END_TOKEN, otherwise it ends mid-word.
                if not tok.endswith(END_TOKEN):
                    continue
                # Also require token begins with a non-END char (i.e., starts with a word).
                if tok.startswith(END_TOKEN):
                    continue
                self._require_word_start.add(int(tid))
            self._insert(tok, tid)

    def _insert(self, tok: str, tid: int) -> None:
        cur = 0
        for ch in tok:
            nxt = self.nodes[cur].next.get(ch)
            if nxt is None:
                nxt = len(self.nodes)
                self.nodes.append(_TrieNode())
                self.nodes[cur].next[ch] = nxt
            cur = nxt
        self.nodes[cur].term_id = tid

    def dp_min_ids(self, s: str, unk_id: int) -> List[int]:
        # DP：最少 token 数，tie-break：更长 token 优先
        n = len(s)
        if n == 0:
            return []

        INF = 10 ** 18
        dp = [INF] * (n + 1)
        back: List[Optional[Tuple[int, int]]] = [None] * (n + 1)  # (next_pos, token_id)
        dp[n] = 0

        for i in range(n - 1, -1, -1):
            node = 0
            at_word_start = i == 0 or s[i - 1] == END_TOKEN
            # 枚举所有从 i 开始的 token
            for j in range(i, n):
                ch = s[j]
                nxt = self.nodes[node].next.get(ch)
                if nxt is None:
                    break
                node = nxt
                tid = self.nodes[node].term_id
                if tid is not None:
                    if tid in self._require_word_start and not at_word_start:
                        continue
                    cand = 1 + dp[j + 1]
                    better = False
                    if cand < dp[i]:
                        better = True
                    elif cand == dp[i]:
                        # tie-break：更长 token 优先
                        if back[i] is None:
                            better = True
                        else:
                            best_j, _ = back[i]
                            if (j + 1 - i) > (best_j - i):
                                better = True
                    if better:
                        dp[i] = cand
                        back[i] = (j + 1, tid)

            # 兜底：若没有任何 token 命中，消费 1 个字符，输出 <unk>
            if back[i] is None:
                dp[i] = 1 + dp[i + 1]
                back[i] = (i + 1, unk_id)

        # 回溯
        out: List[int] = []
        i = 0
        while i < n:
            step = back[i]
            if step is None:
                out.append(unk_id)
                i += 1
                continue
            j, tid = step
            out.append(tid)
            i = j
        return out


def _normalize(text: str) -> str:
    # 与 Rust 的 normalize_chars 一致：split_whitespace + 每个词后追加 END_TOKEN
    parts: List[str] = []
    for w in text.split():
        parts.append(w)
        parts.append(END_TOKEN)
    return "".join(parts)


class LengthTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        mask_token: str = "<mask>",
        **kwargs,
    ) -> None:
        if not os.path.isfile(vocab_file):
            raise FileNotFoundError(f"vocab_file not found: {vocab_file}")
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        # vocab.json: token -> id
        self.vocab: Dict[str, int] = {k: int(v) for k, v in vocab.items()}
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            **kwargs,
        )

        # trie 在 super() 后构建：确保 special tokens 已就绪
        if self.unk_token not in self.vocab:
            raise ValueError(f"unk_token {self.unk_token!r} not in vocab")
        self._unk_id = self.vocab[self.unk_token]
        self._trie = _TokenTrie(self.vocab)

        # Rust 扩展（可选）：如果可用，则优先用它跑 DP（更快）
        # 可通过环境变量禁用，便于对齐/排障：LENGTH_TOKENIZER_DISABLE_RUST=1
        self._rust = None
        if _RUST_AVAILABLE and os.environ.get("LENGTH_TOKENIZER_DISABLE_RUST") is None:
            try:
                self._rust = _RustDpTokenizer(vocab_file, self.unk_token)  # type: ignore
            except Exception:
                self._rust = None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        if self._rust is not None:
            ids = self._rust.encode(text)
            return [self.id_to_token[int(i)] for i in ids]

        norm = _normalize(text)
        ids = self._trie.dp_min_ids(norm, self._unk_id)
        return [self.id_to_token[i] for i in ids]

    def __call__(  # type: ignore[override]
        self,
        text,
        text_pair=None,
        add_special_tokens: bool = True,
        padding=False,
        truncation=False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors=None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        # 批量输入时优先走 Rust encode_batch（吞吐更高），其余复杂特性回退给 transformers 默认实现。
        if (
            self._rust is not None
            and text_pair is None
            and isinstance(text, (list, tuple))
            and not is_split_into_words
            and stride == 0
            and not return_overflowing_tokens
            and not return_special_tokens_mask
            and not return_offsets_mapping
            and not kwargs
        ):
            return self._batch_encode_rust(
                list(text),
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_length=return_length,
            )

        return super().__call__(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    def batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):  # type: ignore[override]
        # 兼容用户直接调用 batch_encode_plus；同样优先走 Rust encode_batch
        if (
            self._rust is not None
            and isinstance(batch_text_or_text_pairs, (list, tuple))
            and batch_text_or_text_pairs
            and all(isinstance(x, str) for x in batch_text_or_text_pairs)
        ):
            return self.__call__(list(batch_text_or_text_pairs), **kwargs)
        return super().batch_encode_plus(batch_text_or_text_pairs, **kwargs)

    def _batch_encode_rust(
        self,
        texts: List[str],
        add_special_tokens: bool,
        padding,
        truncation,
        max_length: Optional[int],
        pad_to_multiple_of: Optional[int],
        return_tensors,
        return_token_type_ids: Optional[bool],
        return_attention_mask: Optional[bool],
        return_length: bool,
    ):
        # 延迟导入，避免在无 transformers 的测试环境里 import 失败
        from transformers.tokenization_utils_base import BatchEncoding

        assert self._rust is not None

        # 1) Rust DP 批量分词（输出 vocab id）
        input_ids: List[List[int]] = self._rust.encode_batch(texts)

        # 2) 特殊 token（可选）
        if add_special_tokens:
            bos_id = self.vocab.get(self.bos_token, None) if self.bos_token is not None else None
            eos_id = self.vocab.get(self.eos_token, None) if self.eos_token is not None else None
            if bos_id is not None or eos_id is not None:
                new_ids: List[List[int]] = []
                for ids in input_ids:
                    if bos_id is not None:
                        ids = [int(bos_id)] + list(ids)
                    if eos_id is not None:
                        ids = list(ids) + [int(eos_id)]
                    new_ids.append(ids)
                input_ids = new_ids

        # 3) 截断（简化版：仅支持单序列，按 max_length 截断）
        if truncation and max_length is not None:
            input_ids = [ids[: max_length] for ids in input_ids]

        # 4) padding（支持 True/\"longest\"/\"max_length\"）
        pad_id = self.vocab.get(self.pad_token, self._unk_id)
        attention_mask: Optional[List[List[int]]] = None
        if return_attention_mask is None:
            return_attention_mask = True

        pad_len: Optional[int] = None
        if padding is True or padding == "longest":
            pad_len = max((len(ids) for ids in input_ids), default=0)
        elif padding == "max_length":
            pad_len = max_length if max_length is not None else None

        if pad_len is not None and pad_to_multiple_of:
            if pad_len % pad_to_multiple_of != 0:
                pad_len = ((pad_len // pad_to_multiple_of) + 1) * pad_to_multiple_of

        if pad_len is not None:
            padded: List[List[int]] = []
            if return_attention_mask:
                attention_mask = []
            for ids in input_ids:
                n = len(ids)
                if n < pad_len:
                    pad_n = pad_len - n
                    padded.append(ids + [int(pad_id)] * pad_n)
                    if return_attention_mask:
                        attention_mask.append([1] * n + [0] * pad_n)
                else:
                    padded.append(ids)
                    if return_attention_mask:
                        attention_mask.append([1] * n)
            input_ids = padded
        else:
            if return_attention_mask:
                attention_mask = [[1] * len(ids) for ids in input_ids]

        data = {"input_ids": input_ids}
        if return_attention_mask:
            data["attention_mask"] = attention_mask

        if return_token_type_ids:
            data["token_type_ids"] = [[0] * len(ids) for ids in input_ids]

        if return_length:
            data["length"] = [len(ids) for ids in input_ids]

        return BatchEncoding(data, tensor_type=return_tensors)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self._unk_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.id_to_token.get(int(index), self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # 简单反归一化：把 'Ġ' 还原为空格
        s = "".join(tokens)
        return s.replace(END_TOKEN, " ").strip()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        out = (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        path = os.path.join(save_directory, out)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return (path,)


