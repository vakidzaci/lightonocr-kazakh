"""Find installed fonts that can render every Kazakh Cyrillic letter.

Fonts are searched in ./fonts (drop your own .ttf/.otf files there, e.g. Times New Roman, which is common in
Kazakh official documents) and in the system font folders. Results are cached in data/fonts.json.
Run directly to list what was found:  python -m generator.fonts
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from fontTools.ttLib import TTFont

from kzocr.common import DATA_DIR, KAZAKH_LETTERS, REPO_ROOT

FONT_DIRS = [REPO_ROOT / "fonts", Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
             Path.home() / ".fonts", Path.home() / ".local/share/fonts"]
REQUIRED = (KAZAKH_LETTERS | set("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя")
            | set("0123456789.,:;!?()-–—«»№%\"'/"))
SKIP_WORDS = ("emoji", "symbol", "math", "music", "braille", "dingbat", "icons", "ornament", "barcode",
              "z003", "chancery")  # calligraphic script fonts: almost never used in documents


def _check(path: Path) -> dict | None:
    try:
        font = TTFont(str(path), lazy=True, fontNumber=0)
        cmap = font.getBestCmap() or {}
        if not all(ord(c) in cmap for c in REQUIRED):
            return None
        name = font["name"]
        family = name.getDebugName(16) or name.getDebugName(1) or path.stem
        style = name.getDebugName(17) or name.getDebugName(2) or "Regular"
        return {"path": str(path), "family": family, "style": style,
                "bold": "bold" in style.lower() or "black" in style.lower(),
                "italic": "italic" in style.lower() or "oblique" in style.lower()}
    except Exception:
        return None


def find_fonts(refresh: bool = False, cache: Path = DATA_DIR / "fonts.json") -> list[dict]:
    if cache.exists() and not refresh:
        fonts = json.loads(cache.read_text())
        if fonts and all(Path(f["path"]).exists() for f in fonts):
            return fonts
    fonts, seen = [], set()
    for d in FONT_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in (".ttf", ".otf") or p.name in seen:
                continue
            if any(w in p.name.lower() for w in SKIP_WORDS):
                continue
            seen.add(p.name)
            info = _check(p)
            if info:
                fonts.append(info)
    if not fonts:
        raise RuntimeError("No font with full Kazakh coverage found. Run scripts/setup_vastai.sh "
                           "(installs font packages) or put .ttf files into ./fonts")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(fonts, ensure_ascii=False, indent=1))
    return fonts


class FontPicker:
    """Picks a family uniformly (so big families like Noto don't dominate), then a style within it."""

    def __init__(self, fonts: list[dict]):
        self.families: dict[str, list[dict]] = defaultdict(list)
        for f in fonts:
            self.families[f["family"]].append(f)
        self.names = sorted(self.families)

    def pick(self, rng: random.Random, bold: bool | None = None) -> dict:
        fam = self.families[rng.choice(self.names)]
        if bold is not None:
            matching = [f for f in fam if f["bold"] == bold and not f["italic"]]
            if matching:
                return rng.choice(matching)
        return rng.choice(fam)


if __name__ == "__main__":
    fonts = find_fonts(refresh=True)
    fams = FontPicker(fonts).families
    print(f"{len(fonts)} font files in {len(fams)} families can render Kazakh:")
    for fam, fs in sorted(fams.items()):
        print(f"  {fam:<40} {len(fs)} styles")
