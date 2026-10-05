"""Regenerates golden.json from the Rust `tokenizers` library.

    pip install tokenizers
    python gen_golden.py   # run from this directory

tokenizer.json is sentence-transformers/all-MiniLM-L6-v2's (Apache-2.0), the
same bert-base-uncased WordPiece pipeline the text classifier uses.
"""
import json
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")
tok.no_padding()

texts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "Café naïve résumé coöperate ÉCOLE",
    "UNAFFABLE unaffable tokenization",
    "Check out https://example.com/path?q=1&x=y#frag now",
    "price: $1,299.99 (20% off) -- 3+4=7 ^_^ ~tilde~ `code` {braces} [brackets] <tags>",
    "中文字符测试 日本語のテキスト 한국어 텍스트",
    "ext-b \U00020000 ext-e-gap \U0002B820 \U0002B8FF ext-e \U0002B920",
    "zero​width‍joiner﻿bom soft­hyphen",
    "control\x01\x02chars\x7f and\ttabs\nnewlines\r\nand nbsp line　ideographic",
    "İstanbul DİYARBAKIR straße ΣΊΣΥΦΟΣ",
    "emoji 😀🎉 and symbols ©®™ ½ ∑ → ★",
    "a" * 100,
    "b" * 101,
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "Ünïcödé ñ ç ø å æ œ ł đ ħ",
    "",
    "   ",
    "� replacement \x00 null",
    "mixed123numbers456 and 3.14159 and 1e10",
    "don't won't it's y'all rock'n'roll",
    "#hashtag @mention r/subreddit u/user",
]
long_text = " ".join(["The adult content filter reads every page title and visible text."] * 40)

cases = []
for t in texts:
    enc = tok.encode(t)
    cases.append({"text": t, "max_len": 0, "ids": enc.ids, "tokens": enc.tokens})
for n in (128, 16, 3):
    tok.enable_truncation(n)
    enc = tok.encode(long_text)
    cases.append({"text": long_text, "max_len": n, "ids": enc.ids, "tokens": enc.tokens})
tok.no_truncation()
enc = tok.encode(long_text)
cases.append({"text": long_text, "max_len": -1, "ids": enc.ids, "tokens": enc.tokens})

with open("golden.json", "w", encoding="utf-8") as f:
    json.dump(cases, f, ensure_ascii=False, indent=1)
print(len(cases), "cases")
