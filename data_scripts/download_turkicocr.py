"""Download the Kazakh pages of alenisaw/turkicocr-cyrillic (synthetic documents, CC-BY-4.0).

29 document types (official letters, certificates «АНЫҚТАМА», forms, tables, books, ...) with stamps,
signatures, phone-photo and scan degradations - close to real Kazakh paperwork.
About half of the pages are Kyrgyz; only Kazakh (and Kazakh+Russian bilingual) pages are kept.

Images are streamed out of the tar shards (nothing large is stored). Targets are built from the page zones in
reading order, in LightOnOCR's output style (titles as Markdown headings, tables as HTML). Stamps and decorative
elements are not targets (the dataset marks them is_ocr_target=false).

Outputs: data/train/turkicocr, data/val/turkicocr, data/test/turkicocr_kk  (manifest + images)
  python data_scripts/download_turkicocr.py --config medium --splits train,validation,test
Sizes (Kazakh pages kept): tiny ~12k train pages (~14 GB streamed), medium ~24k (~27 GB), large ~48k (~55 GB).
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download, hf_hub_url  # noqa: E402

from kzocr.common import DATA_DIR, write_jsonl  # noqa: E402

REPO = "alenisaw/turkicocr-cyrillic"
KAZAKH_LANGS = {"kk", "ru_kk", "bilingual_kk_ru"}
OUT = {"train": DATA_DIR / "train" / "turkicocr", "validation": DATA_DIR / "val" / "turkicocr",
       "test": DATA_DIR / "test" / "turkicocr_kk"}


def table_html(zone: dict) -> str:
    rows: dict[int, list] = defaultdict(list)
    for c in zone["cells"]:
        rows[c["row"]].append(c)
    out = ["<table>"]
    for r in sorted(rows):
        tag = "th" if all((c.get("metadata") or {}).get("header") for c in rows[r]) else "td"
        cells = "".join(f"<{tag}>{html.escape(c['text'].strip(), quote=False)}</{tag}>"
                        for c in sorted(rows[r], key=lambda c: c["col"]))
        out.append(f"<tr>{cells}</tr>")
    out.append("</table>")
    return "".join(out)


def page_target(zones: list[dict]) -> str:
    blocks = []
    for z in sorted(zones, key=lambda z: z.get("reading_order", 0)):
        if not z.get("is_ocr_target", True):
            continue
        if z.get("zone_type") == "table" and z.get("cells"):
            blocks.append(table_html(z))
            continue
        text = (z.get("text") or "").strip()
        if not text:
            continue
        if z.get("zone_type") == "title":
            text = "# " + text.replace("\n", " ")
        elif z.get("zone_type") == "subtitle":
            text = "## " + text.replace("\n", " ")
        blocks.append(text)
    return "\n\n".join(blocks)


def is_kazakh(zones: list[dict]) -> bool:
    langs = {z.get("language") for z in zones if z.get("language")}
    return bool(langs) and langs <= KAZAKH_LANGS


def stream_extract(tar_path: str, wanted: set[str], out_dir: Path, retries: int = 5) -> None:
    url = hf_hub_url(REPO, tar_path, repo_type="dataset")
    for attempt in range(retries):
        todo = {w for w in wanted if not (out_dir / w).exists()}
        if not todo:
            return
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "kzocr"}), timeout=120) as resp:
                with tarfile.open(fileobj=resp, mode="r|*") as tar:
                    for m in tar:
                        name = Path(m.name).name
                        if m.isfile() and name in todo:
                            data = tar.extractfile(m).read()
                            (out_dir / name).write_bytes(data)
                            todo.discard(name)
                            if not todo:
                                return
            return
        except Exception as e:  # network hiccup: retry, already-extracted files are skipped
            print(f"   retry {attempt + 1}/{retries} for {tar_path}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to stream {tar_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="medium", choices=["tiny", "medium", "large"])
    ap.add_argument("--splits", default="train,validation,test")
    ap.add_argument("--max-pages", type=int, default=None, help="per split (Kazakh pages); for quick tests")
    args = ap.parse_args()

    for split in args.splits.split(","):
        idx = hf_hub_download(REPO, f"indexes/{args.config}/{split}.parquet", repo_type="dataset")
        rows = pq.read_table(idx, columns=["page_id", "layout_id", "tar_path", "image_filename", "zones_json"]).to_pylist()
        keep = []
        for r in rows:
            zones = json.loads(r["zones_json"])
            if is_kazakh(zones):
                text = page_target(zones)
                if text:
                    keep.append({**r, "text": text})
        keep = keep[: args.max_pages]
        print(f"[{split}] {len(rows)} pages in index, {len(keep)} Kazakh pages kept", flush=True)
        out_dir = OUT[split]
        img_dir = out_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        by_tar: dict[str, set[str]] = defaultdict(set)
        for r in keep:
            by_tar[r["tar_path"]].add(r["image_filename"])
        for i, (tar_path, names) in enumerate(sorted(by_tar.items()), 1):
            print(f"[{split}] shard {i}/{len(by_tar)} {tar_path}: {len(names)} images", flush=True)
            stream_extract(tar_path, names, img_dir)
        recs = [{"image": f"images/{r['image_filename']}", "text": r["text"], "kind": "page", "source": "turkicocr",
                 "layout": r["layout_id"]} for r in keep if (img_dir / r["image_filename"]).exists()]
        write_jsonl(out_dir / "manifest.jsonl", recs)
        print(f"[{split}] {len(recs)} samples -> {out_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
