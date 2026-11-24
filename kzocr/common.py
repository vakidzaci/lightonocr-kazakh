"""Constants and helpers shared by data scripts, training, evaluation and inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("KZOCR_DATA", REPO_ROOT / "data"))

# The 9 letters (18 with case) that Kazakh Cyrillic adds on top of the Russian alphabet.
KAZAKH_LETTERS = set("ӘәҒғҚқҢңӨөҰұҮүҺһІі")
# Visually similar Russian letters that OCR models typically output instead.
KZ_TO_RU_LOOKALIKE = {
    "Ә": "А", "ә": "а", "Ғ": "Г", "ғ": "г", "Қ": "К", "қ": "к", "Ң": "Н", "ң": "н",
    "Ө": "О", "ө": "о", "Ұ": "У", "ұ": "у", "Ү": "У", "ү": "у", "Һ": "Х", "һ": "х", "І": "И", "і": "и",
}

# LightOnOCR (v2 and v3-1B) is trained on pages whose longest side is 1540 px. The processor already
# shrinks larger images; small *pages* are enlarged here, which noticeably improves accuracy.
# Word/line crops are left at their native size (their text is already at page-like pixel size).
PAGE_LONGEST_SIDE = 1540
UPSCALE_PAGES_BELOW = 1024


def load_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def fit_image(img: Image.Image, kind: str = "page") -> Image.Image:
    if kind != "page":
        return img
    longest = max(img.size)
    if longest < UPSCALE_PAGES_BELOW:
        s = PAGE_LONGEST_SIDE / longest
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    return img


def user_message(image: Image.Image) -> dict:
    # LightOnOCR is prompted with the image only (no text instruction).
    return {"role": "user", "content": [{"type": "image", "image": image}]}


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_manifest(path: str | Path) -> list[dict]:
    """Read a manifest and resolve each record's image path (stored relative to the manifest file)."""
    path = Path(path)
    rows = read_jsonl(path)
    for r in rows:
        img = Path(r["image"])
        r["image"] = str(img if img.is_absolute() else (path.parent / img).resolve())
        r.setdefault("kind", "page")
    return rows


def image_ext(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".img"
