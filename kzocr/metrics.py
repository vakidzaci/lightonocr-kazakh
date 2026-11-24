"""OCR metrics: CER / WER, Kazakh-specific letter recall and substitution statistics."""

from __future__ import annotations

import collections
import html
import re

from rapidfuzz.distance import Levenshtein

from .common import KAZAKH_LETTERS

_TAG = re.compile(r"<[^>]+>")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def normalize(text: str, lowercase: bool = False) -> str:
    """Make model output and references comparable: drop Markdown/HTML markup, collapse whitespace."""
    text = _MD_IMAGE.sub(" ", text)                      # image placeholders / grounding boxes
    text = re.sub(r"</t[dh]>", " ", text)               # keep table cells apart
    text = _TAG.sub(" ", text)                          # other HTML tags (tables, <br>)
    text = html.unescape(text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.M)  # Markdown headings
    text = text.replace("**", "").replace("__", "")    # bold
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.M)  # bullets
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text


def cer(ref: str, hyp: str) -> tuple[int, int]:
    return Levenshtein.distance(ref, hyp), len(ref)


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = ref.split(), hyp.split()
    return Levenshtein.distance(r, h), len(r)


def kazakh_letters(ref: str, hyp: str) -> tuple[int, int, collections.Counter]:
    """(#Kazakh letters in ref, #reproduced at the aligned position, Counter of (ref_char -> hyp_char))."""
    total = sum(c in KAZAKH_LETTERS for c in ref)
    ok = 0
    subs: collections.Counter = collections.Counter()
    for op in Levenshtein.opcodes(ref, hyp):
        seg = ref[op.src_start:op.src_end]
        if op.tag == "equal":
            ok += sum(c in KAZAKH_LETTERS for c in seg)
        elif op.tag == "replace":
            hseg = hyp[op.dest_start:op.dest_end]
            for i, c in enumerate(seg):
                if c in KAZAKH_LETTERS:
                    subs[(c, hseg[i] if i < len(hseg) else "∅")] += 1
        elif op.tag == "delete":
            for c in seg:
                if c in KAZAKH_LETTERS:
                    subs[(c, "∅")] += 1
    return total, ok, subs


class Scorer:
    """Accumulates metrics over a test set."""

    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase
        self.n = 0
        self.char_edits = self.chars = self.word_edits = self.words = 0
        self.page_cers: list[float] = []
        self.kz_total = self.kz_ok = 0
        self.subs: collections.Counter = collections.Counter()
        self.exact = 0
        self.runaways = 0
        # misspelling probes, per misspelled word: copied as printed (good) vs silently restored to correct Kazakh
        self.misspelled_words = self.copied_words = self.restored_words = 0

    def add(self, ref: str, hyp: str, hit_token_limit: bool = False, orig: str | None = None) -> dict:
        r, h = normalize(ref, self.lowercase), normalize(hyp, self.lowercase)
        ce, cn = cer(r, h)
        we, wn = wer(r, h)
        kt, ko, subs = kazakh_letters(r, h)
        self.n += 1
        self.char_edits += ce; self.chars += cn; self.word_edits += we; self.words += wn
        page_cer = ce / max(cn, 1)
        self.page_cers.append(min(page_cer, 1.0))
        self.kz_total += kt; self.kz_ok += ko; self.subs.update(subs)
        self.exact += r == h
        self.runaways += hit_token_limit
        if orig is not None:
            self._misspellings(r.split(), normalize(orig, self.lowercase).split(), h.split())
        return {"cer": page_cer, "kz_total": kt, "kz_ok": ko, "exact": r == h}

    def _misspellings(self, printed: list[str], correct: list[str], hyp: list[str]) -> None:
        if len(printed) != len(correct):
            return
        for op in Levenshtein.opcodes(printed, hyp):
            if op.tag not in ("equal", "replace") or (op.src_end - op.src_start) != (op.dest_end - op.dest_start):
                continue
            for i, j in zip(range(op.src_start, op.src_end), range(op.dest_start, op.dest_end)):
                if printed[i] != correct[i]:
                    self.misspelled_words += 1
                    self.copied_words += hyp[j] == printed[i]
                    self.restored_words += hyp[j] == correct[i]

    def summary(self) -> dict:
        out = {
            "samples": self.n,
            "cer": self.char_edits / max(self.chars, 1),
            "wer": self.word_edits / max(self.words, 1),
            "mean_page_cer_capped": sum(self.page_cers) / max(len(self.page_cers), 1),
            "exact_match": self.exact / max(self.n, 1),
            "kazakh_letter_recall": self.kz_ok / max(self.kz_total, 1) if self.kz_total else None,
            "runaway_generations": self.runaways,
            "top_substitutions": [f"{a}->{b} x{n}" for (a, b), n in self.subs.most_common(12)],
        }
        if self.misspelled_words:
            out["misspelled_words"] = self.misspelled_words
            out["copied_misspelling"] = self.copied_words / self.misspelled_words        # want high
            out["restored_correct_spelling"] = self.restored_words / self.misspelled_words  # want ~0
        return out
