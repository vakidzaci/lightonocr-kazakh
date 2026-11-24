"""OCR images or PDFs with a base, LoRA-adapted, or merged LightOnOCR model.

  python infer.py scan.jpg
  python infer.py scan.jpg other.png --adapter outputs/lora/final
  python infer.py book.pdf --pages 1-5 --model your-hf-name/lightonocr-kazakh --out results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kzocr.common import PAGE_LONGEST_SIDE, fit_image, load_image
from kzocr.model import load_model, ocr_batch


def pages_of(path: Path, page_range: str | None):
    if path.suffix.lower() != ".pdf":
        yield path.stem, fit_image(load_image(path), "page")
        return
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    a, _, b = (page_range or f"1-{len(pdf)}").partition("-")
    for i in range(int(a) - 1, min(int(b or a), len(pdf))):
        page = pdf[i]
        scale = PAGE_LONGEST_SIDE / max(page.get_size())
        yield f"{path.stem}_p{i + 1}", page.render(scale=scale).to_pil().convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--model", default="lightonai/LightOnOCR-2-1B-base")
    ap.add_argument("--adapter")
    ap.add_argument("--pages", help="PDF page range, e.g. 3 or 1-5")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--out", type=Path, help="write <name>.md files here")
    args = ap.parse_args()

    model, processor = load_model(args.model, args.adapter)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    for path in args.inputs:
        for name, img in pages_of(path, args.pages):
            text, capped = ocr_batch(model, processor, [img], args.max_new_tokens)[0]
            print(f"\n===== {name} =====\n{text}" + ("\n[warning: hit max-new-tokens]" if capped else ""))
            if args.out:
                (args.out / f"{name}.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
