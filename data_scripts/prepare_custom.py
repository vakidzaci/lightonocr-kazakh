"""Turn your own labeled pages (e.g. scanned Kazakh books) into train/test manifests.

Expected layout - one folder per book, each image next to a UTF-8 .txt file with the same name:
    my_books/
      abai_joldary/        page_001.jpg  page_001.txt  page_002.jpg  page_002.txt ...
      altynsarin_hrest/    ...
Whole books are held out for testing (never split a book between train and test, or the test score lies).

  python data_scripts/prepare_custom.py my_books --test-books abai_joldary
Outputs: data/train/custom/manifest.jsonl, data/test/custom/manifest.jsonl (absolute image paths)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kzocr.common import DATA_DIR, write_jsonl  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--test-books", default="", help="comma-separated book folder names to hold out")
    ap.add_argument("--kind", default="page", choices=["page", "line"])
    args = ap.parse_args()

    test_books = {b for b in args.test_books.split(",") if b}
    train, test, missing = [], [], 0
    for img in sorted(p for p in args.root.rglob("*") if p.suffix.lower() in EXTS):
        txt = img.with_suffix(".txt")
        if not txt.exists():
            missing += 1
            continue
        book = img.relative_to(args.root).parts[0] if img.parent != args.root else "_root"
        rec = {"image": str(img.resolve()), "text": txt.read_text(encoding="utf-8").strip(), "kind": args.kind,
               "source": "custom", "book": book}
        (test if book in test_books else train).append(rec)
    unknown = test_books - {r["book"] for r in test}
    if unknown:
        sys.exit(f"--test-books not found: {sorted(unknown)}")
    print(f"train: {write_jsonl(DATA_DIR / 'train' / 'custom' / 'manifest.jsonl', train)} pages, "
          f"test: {write_jsonl(DATA_DIR / 'test' / 'custom' / 'manifest.jsonl', test)} pages, "
          f"{missing} images without a .txt skipped")


if __name__ == "__main__":
    main()
