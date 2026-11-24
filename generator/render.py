"""Rendering text to images (single lines and full pages) plus scan/photo-style degradations.

The target text of a page follows LightOnOCR's own output style: blocks in reading order separated by a blank
line, wrapped paragraphs joined into one line, titles as Markdown headings, tables as HTML. Stamps are drawn on
top but are not part of the target (LightOnOCR does not transcribe stamps either).
"""

from __future__ import annotations

import io
import math
import random
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .fonts import FontPicker
from .textgen import TextSource

LEVELS = ("clean", "medium", "hard")


@lru_cache(maxsize=512)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _paper(rng: random.Random, size: tuple[int, int], level: str) -> Image.Image:
    w, h = size
    if level == "clean" and rng.random() < 0.6:
        base = (255, 255, 255)
    else:
        tint = rng.choice([(250, 248, 240), (245, 242, 230), (240, 240, 240), (236, 230, 214), (248, 246, 250)])
        base = tuple(max(0, c - rng.randint(0, 18 if level == "hard" else 8)) for c in tint)
    arr = np.empty((h, w, 3), dtype=np.float32)
    arr[:] = base
    if level != "clean":  # low-frequency stains / uneven paper
        small = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 1, (max(2, h // 64), max(2, w // 64), 1))
        blot = np.asarray(Image.fromarray(((small - small.min()) / (np.ptp(small) + 1e-6) * 255).astype(np.uint8)[:, :, 0])
                          .resize((w, h), Image.BICUBIC), dtype=np.float32)[:, :, None]
        arr -= (blot / 255.0) * (14 if level == "hard" else 6)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _ink(rng: random.Random) -> tuple[int, int, int]:
    if rng.random() < 0.12:
        return (rng.randint(10, 40), rng.randint(20, 50), rng.randint(90, 140))  # blue ink / blue print
    g = rng.randint(0, 60)
    return (g, g, g)


def _wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if font.getlength(cand) <= width or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _stamp(img: Image.Image, rng: random.Random, picker: FontPicker, src: TextSource) -> None:
    r = rng.randint(70, 130)
    layer = Image.new("RGBA", (2 * r + 20, 2 * r + 20), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = (rng.randint(20, 70), rng.randint(40, 90), rng.randint(150, 220), rng.randint(140, 210))
    c = r + 10
    d.ellipse([c - r, c - r, c + r, c + r], outline=col, width=rng.randint(3, 6))
    d.ellipse([c - r + 18, c - r + 18, c + r - 18, c + r - 18], outline=col, width=2)
    f = _font(picker.pick(rng)["path"], max(10, r // 7))
    for k, word in enumerate(src.text(rng, "natural", 40)[0].upper().split()[:10]):  # letters around the ring
        for j, ch in enumerate(word):
            a = (k * 5 + j) * 0.21
            x, y = c + (r - 11) * math.cos(a), c + (r - 11) * math.sin(a)
            d.text((x, y), ch, fill=col, font=f, anchor="mm")
    layer = layer.rotate(rng.uniform(0, 360), resample=Image.BICUBIC)
    x, y = rng.randint(0, max(0, img.width - layer.width)), rng.randint(0, max(0, img.height - layer.height))
    img.paste(layer, (x, y), layer)


def degrade(img: Image.Image, rng: random.Random, level: str, page: bool) -> Image.Image:
    strength = {"clean": 0, "medium": 1, "hard": 2}[level]
    fill = img.getpixel((2, 2))
    angle = rng.uniform(-1, 1) * (0.3, 1.2, 2.5)[strength]
    if abs(angle) > 0.05:
        img = img.rotate(angle, resample=Image.BICUBIC, expand=not page, fillcolor=fill)
    if strength == 2 and rng.random() < 0.5:  # mild perspective (phone photo)
        w, h = img.size
        j = lambda: rng.uniform(0, 0.03) * min(w, h)  # noqa: E731
        quad = (j(), j(), j(), h - j(), w - j(), h - j(), w - j(), j())
        img = img.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC, fillcolor=fill)
    if strength and rng.random() < 0.5:  # low-resolution round trip
        s = rng.uniform(0.45 if strength == 2 else 0.65, 0.9)
        w, h = img.size
        img = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR).resize((w, h), Image.BICUBIC)
    blur = rng.uniform(0, (0.35, 0.8, 1.4)[strength])
    if blur > 0.2:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if strength:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.15))
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.1))
    if page and strength == 2 and rng.random() < 0.4:  # shadow band
        arr = np.asarray(img, dtype=np.float32)
        h, w = arr.shape[:2]
        x0 = rng.randint(0, w)
        ramp = np.clip(1 - np.abs(np.arange(w) - x0) / (w * rng.uniform(0.15, 0.4)), 0, 1) * rng.uniform(0.15, 0.35)
        arr *= (1 - ramp)[None, :, None]
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    sigma = rng.uniform(0, (2, 6, 12)[strength])
    if sigma > 0.5:
        arr = np.asarray(img, dtype=np.float32)
        arr += np.random.default_rng(rng.randint(0, 2**31)).normal(0, sigma, arr.shape[:2])[:, :, None]
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if strength and rng.random() < 0.7:  # JPEG artefacts
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(25, 60) if strength == 2 else rng.randint(55, 85))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


def render_line(text: str, rng: random.Random, picker: FontPicker, level: str) -> Image.Image:
    font_info = picker.pick(rng)
    size = rng.randint(18, 44)
    font = _font(font_info["path"], size)
    pad_x, pad_y = rng.randint(6, 30), rng.randint(4, 16)
    l, t, r, b = font.getbbox(text)
    w, h = int(r - min(l, 0)) + 2 * pad_x, int(max(b, size * 1.25) - min(t, 0)) + 2 * pad_y
    img = _paper(rng, (w, h), level)
    ImageDraw.Draw(img).text((pad_x - min(l, 0), pad_y - min(t, 0)), text, fill=_ink(rng), font=font)
    return degrade(img, rng, level, page=False)


def render_probe_page(lines: list[str], rng: random.Random, picker: FontPicker) -> Image.Image:
    """A clean page with one short text per line - the format LightOnOCR expects (isolated tiny crops make the
    base model drift into LaTeX/math mode, which would measure format adaptation instead of reading)."""
    font = _font(picker.pick(rng, bold=False)["path"], rng.randint(28, 38))
    line_h, margin = int(font.size * 1.7), 110
    img = _paper(rng, (1240, max(900, 2 * margin + line_h * len(lines))), "clean")
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        d.text((margin, margin + i * line_h), ln, fill=(20, 20, 20), font=font)
    return degrade(img, rng, "clean", page=True)


def render_page(src: TextSource, rng: random.Random, picker: FontPicker, level: str,
                modes: list[str], mode_weights: list[float]) -> tuple[Image.Image, str]:
    W, H = rng.choice([(1240, 1754), (1240, 1754), (1275, 1650), (1100, 1556)])
    margin = rng.randint(70, 140)
    body = picker.pick(rng)
    bold = picker.pick(rng, bold=True) if rng.random() < 0.5 else body
    size = rng.randint(20, 30)
    f_body, f_title = _font(body["path"], size), _font(bold["path"], int(size * rng.uniform(1.15, 1.6)))
    img = _paper(rng, (W, H), level)
    d = ImageDraw.Draw(img)
    ink = _ink(rng)
    y = margin
    blocks: list[str] = []
    line_h = int(size * rng.uniform(1.3, 1.6))
    width = W - 2 * margin

    def fits(n_lines: int, lh: int) -> bool:
        return y + n_lines * lh <= H - margin

    def text_block(text: str, font, align: str = "left", lh: int = line_h) -> bool:
        nonlocal y
        lines = _wrap(text, font, width)
        if not lines or not fits(len(lines), lh):
            return False
        for ln in lines:
            lw = font.getlength(ln)
            x = margin + (width - lw) / 2 if align == "center" else margin + width - lw if align == "right" else margin
            d.text((x, y), ln, fill=ink, font=font)
            y += lh
        y += int(lh * rng.uniform(0.3, 0.9))
        return True

    if rng.random() < 0.6:  # organisation header
        hdr = src.text(rng, "entities", 60)[0]
        hdr = f"«{hdr}»" if rng.random() < 0.3 else hdr
        if text_block(hdr, f_body, rng.choice(["left", "center"])):
            blocks.append(hdr)
    if rng.random() < 0.85:  # title
        title = rng.choice([src.text(rng, "entities", 40)[0], src.text(rng, "natural", 50)[0]])
        title = title.upper() if rng.random() < 0.5 else title
        if text_block(title, f_title, rng.choice(["center", "center", "left"]), int(f_title.size * 1.4)):
            blocks.append("# " + title)
    for _ in range(rng.randint(2, 7)):
        kind = rng.random()
        if kind < 0.65:  # paragraph
            mode = rng.choices(modes, weights=mode_weights)[0]
            para = src.text(rng, mode, rng.randint(150, 600))[0]
            if not text_block(para, f_body, rng.choice(["left", "left", "center"])):
                break
            blocks.append(para)
        elif kind < 0.85:  # key: value lines
            kv = [src.entity(rng) for _ in range(rng.randint(2, 5))]
            if not fits(len(kv) + 1, line_h):
                break
            done = []
            for ln in kv:
                if not text_block(ln, f_body, "left"):
                    break
                done.append(ln)  # only lines that were actually drawn go into the target
                y -= int(line_h * 0.3)
            if done:
                blocks.append("\n".join(done))
        else:  # simple table
            cols, rows = rng.randint(2, 4), rng.randint(2, 5)
            cells = [[(src.entity(rng) if rng.random() < 0.5 else src.text(rng, "pseudo", 18)[0])[:24]
                      for _ in range(cols)] for _ in range(rows)]
            row_h = int(line_h * 1.4)
            if not fits(rows + 1, row_h):
                break
            col_w = width / cols
            fs = _font(body["path"], max(14, int(size * 0.8)))
            for ri, row in enumerate(cells):
                for ci, cell in enumerate(row):
                    x0, y0 = margin + ci * col_w, y + ri * row_h
                    d.rectangle([x0, y0, x0 + col_w, y0 + row_h], outline=ink, width=1)
                    while fs.getlength(cell) > col_w - 10 and len(cell) > 1:
                        cell = cell[:-1]
                    row[ci] = cell
                    d.text((x0 + 5, y0 + (row_h - fs.size) / 2), cell, fill=ink, font=fs)
            y += rows * row_h + line_h
            blocks.append("<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                                              for r in cells) + "</table>")
    if rng.random() < 0.4 and fits(1, line_h):  # signature line
        sig = f"{rng.choice(['Басшы', 'Орынбасары', 'Жауапты маман', 'Директор'])}    {src.person(rng)}"
        if text_block(sig, f_body, "left"):
            blocks.append(sig)
    if rng.random() < 0.3:  # page number
        num = str(rng.randint(1, 400))
        d.text((W / 2, H - margin / 2), num, fill=ink, font=f_body, anchor="mm")
        blocks.append(num)
    if rng.random() < 0.3:
        _stamp(img, rng, picker, src)
    return degrade(img, rng, level, page=True), "\n\n".join(blocks)
