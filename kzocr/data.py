"""Training data: manifest mixing, the torch Dataset, the collator that builds masked labels, and a sampler
that keeps word/line crops and full pages in separate batches (so small crops are not padded to page size)."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from .common import REPO_ROOT, fit_image, load_image, load_manifest, user_message

log = logging.getLogger(__name__)

IGNORE_INDEX = -100


def build_mixture(sources: list[dict], seed: int = 42) -> list[dict]:
    """sources: [{manifest, max_samples?, repeat?, optional?, max_chars?}] -> shuffled list of records."""
    rng = random.Random(seed)
    records: list[dict] = []
    for src in sources:
        path = Path(src["manifest"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            if src.get("optional"):
                log.warning("skipping missing optional manifest %s", path)
                continue
            raise FileNotFoundError(f"manifest not found: {path} (run the download/generate scripts first)")
        rows = load_manifest(path)
        max_chars = src.get("max_chars")
        if max_chars:
            rows = [r for r in rows if len(r["text"]) <= max_chars]
        rng.shuffle(rows)
        if src.get("max_samples"):
            rows = rows[: int(src["max_samples"])]
        rows = rows * int(src.get("repeat", 1))
        log.info("%-60s %7d samples", str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path), len(rows))
        records.extend(rows)
    rng.shuffle(records)
    if not records:
        raise ValueError("training mixture is empty")
    return records


class OcrDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records
        self.kinds = [("page" if r.get("kind", "page") == "page" else "crop") for r in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        try:
            img = fit_image(load_image(r["image"]), r.get("kind", "page"))
        except Exception as e:  # a corrupt image should not kill a multi-hour run
            log.warning("could not load %s (%s); using a neighbour", r["image"], e)
            return self[(i + 1) % len(self)]
        return {"image": img, "text": r["text"]}


class OcrCollator:
    """Turns {image, text} samples into model inputs. Only the answer tokens (+ <|im_end|>) are trained on."""

    MIN_TEXT_BUDGET = 512  # tokens always left for the answer after the image

    def __init__(self, processor, max_length: int = 4096):
        self.processor = processor
        self.max_length = max_length
        ip = processor.image_processor
        self.longest_edge = ip.size["longest_edge"]
        self.cell = ip.patch_size["height"] if isinstance(ip.patch_size, dict) else ip.patch_size
        self.cell *= getattr(processor, "spatial_merge_size", 2)
        tok = processor.tokenizer
        self.pad_id = tok.pad_token_id
        self.im_start = tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.assistant = tok.encode("assistant", add_special_tokens=False)

    def _answer_start(self, ids: list[int]) -> int:
        n = len(self.assistant)
        for i in range(len(ids) - n - 1, -1, -1):
            if ids[i] == self.im_start and ids[i + 1 : i + 1 + n] == self.assistant:
                return i + 1 + n + 1  # skip "<|im_start|>assistant\n"
        raise ValueError("assistant turn not found in tokenized sample")

    def _image_tokens(self, w: int, h: int) -> int:
        s = min(1.0, self.longest_edge / max(w, h))
        rows, cols = -(-round(h * s) // self.cell), -(-round(w * s) // self.cell)
        return rows * cols + 1  # matches the processor for pages (slight over-estimate for thin crops: safe)

    def _fit(self, img):
        """Shrink an image whose tokens alone would exceed the budget - truncation must never cut image tokens."""
        budget = self.max_length - self.MIN_TEXT_BUDGET
        n = self._image_tokens(*img.size)
        if n <= budget:
            return img
        s = min(1.0, self.longest_edge / max(img.size)) * (budget / n) ** 0.5
        while True:
            size = (max(self.cell, int(img.width * s)), max(self.cell, int(img.height * s)))
            if self._image_tokens(*size) <= budget or size[0] <= self.cell:
                return img.resize(size)
            s *= 0.95

    def __call__(self, batch: list[dict]) -> dict:
        convs = [[user_message(self._fit(s["image"])), {"role": "assistant", "content": s["text"]}] for s in batch]
        self.processor.tokenizer.padding_side = "right"
        enc = self.processor.apply_chat_template(
            convs, tokenize=True, return_dict=True, return_tensors="pt", padding=True,
        )
        input_ids, attn = enc["input_ids"], enc["attention_mask"]
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for b in range(input_ids.shape[0]):
            ids = input_ids[b][attn[b].bool()].tolist()
            start = self._answer_start(ids)
            try:
                end = ids.index(self.im_end, start) + 1  # include <|im_end|> so the model learns to stop
            except ValueError:
                end = len(ids)
            labels[b, start:end] = input_ids[b, start:end]
        if input_ids.shape[1] > self.max_length:
            # Rare very long pages: truncate. The cut-off end carries no EOS label, so nothing wrong is learned.
            input_ids, attn, labels = (t[:, : self.max_length] for t in (input_ids, attn, labels))
        out = {k: v for k, v in enc.items() if k not in ("input_ids", "attention_mask")}
        out.update(input_ids=input_ids, attention_mask=attn, labels=labels)
        return out


class KindGroupedSampler(Sampler[int]):
    """Shuffles, but yields indices so that every chunk of `batch_size` consecutive indices has one kind."""

    def __init__(self, kinds: list[str], batch_size: int, seed: int = 42):
        self.kinds, self.batch_size, self.seed, self.epoch = kinds, batch_size, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _groups(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for i, k in enumerate(self.kinds):
            groups.setdefault(k, []).append(i)
        return groups

    def __len__(self) -> int:
        bs = self.batch_size
        return sum(-(-len(idx) // bs) * bs for idx in self._groups().values())

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        chunks = []
        for idx in self._groups().values():
            rng.shuffle(idx)
            for j in range(0, len(idx), self.batch_size):
                chunk = idx[j : j + self.batch_size]
                while len(chunk) < self.batch_size:  # top up the last chunk so batches stay aligned
                    chunk.append(rng.choice(idx))
                chunks.append(chunk)
        rng.shuffle(chunks)
        for c in chunks:
            yield from c
