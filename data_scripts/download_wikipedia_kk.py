"""Download Kazakh Wikipedia (wikimedia/wikipedia, 20231101.kk, ~180 MB) and build the text corpus used by
the synthetic generator:  data/text/kk_sentences.txt, kk_vocab.tsv (word<TAB>count), kk_names.txt
"""

from __future__ import annotations

import argparse
import collections
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from kzocr.common import DATA_DIR, KAZAKH_LETTERS  # noqa: E402

CYR = "А-ЯЁӘҒҚҢӨҰҮҺІа-яёәғқңөұүһі"
WORD = re.compile(rf"[{CYR}]+(?:-[{CYR}]+)?")
SENT_SPLIT = re.compile(rf"(?<=[.!?])\s+(?=[«\"(]?[{CYR}A-Z0-9])")
ALLOWED = re.compile(rf"^[{CYR}A-Za-z0-9\s.,:;!?()«»\"'/%№–—\-+]+$")


def good_sentence(s: str) -> bool:
    if not (25 <= len(s) <= 260) or not ALLOWED.match(s):
        return False
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 15:
        return False
    cyr = sum(bool(re.match(f"[{CYR}]", c)) for c in letters)
    return cyr / len(letters) > 0.85 and any(c in KAZAKH_LETTERS for c in s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA_DIR / "text")
    ap.add_argument("--max-sentences", type=int, default=400_000)
    args = ap.parse_args()

    path = hf_hub_download("wikimedia/wikipedia", "20231101.kk/train-00000-of-00001.parquet", repo_type="dataset")
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    print(f"{len(texts)} articles")

    sentences, vocab, caps, lowers = set(), collections.Counter(), collections.Counter(), collections.Counter()
    for t in texts:
        for para in t.split("\n"):
            para = para.strip()
            if len(para) < 25:
                continue
            for s in SENT_SPLIT.split(para):
                s = re.sub(r"\s+", " ", s).strip()
                if good_sentence(s):
                    sentences.add(s)
            for i, w in enumerate(WORD.findall(para)):
                vocab[w.lower()] += 1
                if w[0].isupper() and i > 0:
                    caps[w] += 1
                elif w[0].islower():
                    lowers[w.capitalize()] += 1

    sentences = sorted(sentences)
    random.Random(0).shuffle(sentences)
    sentences = sentences[: args.max_sentences]
    words = [(w, c) for w, c in vocab.most_common(250_000) if c >= 3 and len(w) >= 2]
    # Proper names: capitalized mid-sentence much more often than lowercase, containing a Kazakh letter.
    names = [w for w, c in caps.most_common() if c >= 3 and c > 4 * lowers[w] and any(x in KAZAKH_LETTERS for x in w)
             and len(w) >= 3][:30_000]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "kk_sentences.txt").write_text("\n".join(sentences) + "\n", encoding="utf-8")
    (args.out / "kk_vocab.tsv").write_text("".join(f"{w}\t{c}\n" for w, c in words), encoding="utf-8")
    (args.out / "kk_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"{len(sentences)} sentences, {len(words)} words, {len(names)} names -> {args.out}")


if __name__ == "__main__":
    main()
