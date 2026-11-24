# Extra fonts

Put additional `.ttf` / `.otf` files here to use them in the synthetic data generator. Fonts that cannot render
all Kazakh letters (Ә Ғ Қ Ң Ө Ұ Ү Һ І) are skipped automatically.

Good additions: the fonts real Kazakh documents use, e.g. **Times New Roman** and **Arial** (copy them from a
Windows/macOS machine you own a license for), and any Kazakh book typefaces you have.

Font files in this folder are ignored by git (they may be proprietary). After adding fonts, refresh the cache:

```bash
python -m generator.fonts
```
