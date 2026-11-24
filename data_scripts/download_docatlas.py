"""Download Russian (and optionally English) pages of ahmedheakl/docatlas_instruct (CC-BY-4.0).

Real documents rendered from native DOCX files, with exact (not model-generated) Markdown targets. Mixed into
training so the model keeps its Russian/English skills while learning Kazakh (catastrophic-forgetting guard).

The full dataset is 72 GB with all languages mixed in every shard, so instead of downloading it we page through
the Hugging Face dataset-viewer API, which can filter by language, and fetch only the matching images.

  python data_scripts/download_docatlas.py --langs ru,en --max-per-lang 3000 --test-per-lang 100
Outputs: data/train/docatlas/manifest.jsonl, data/test/docatlas_<lang>/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kzocr.common import DATA_DIR, write_jsonl  # noqa: E402

API = "https://datasets-server.huggingface.co/filter"
DATASET = "ahmedheakl/docatlas_instruct"


def get_json(url: str, retries: int = 6) -> dict:
    for a in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "kzocr"}), timeout=90) as r:
                return json.loads(r.read())
        except Exception as e:
            print(f"   API retry {a + 1}: {e}", flush=True)
            time.sleep(3 * (a + 1))
    raise RuntimeError(f"API failed: {url}")


def fetch(url: str, path: Path, retries: int = 5) -> bool:
    if path.exists():
        return True
    for a in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "kzocr"}), timeout=90) as r:
                path.write_bytes(r.read())
            return True
        except Exception:
            time.sleep(2 * (a + 1))
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="ru,en")
    ap.add_argument("--max-per-lang", type=int, default=3000, help="train pages per language")
    ap.add_argument("--test-per-lang", type=int, default=100, help="held-out pages per language")
    ap.add_argument("--max-chars", type=int, default=6000, help="skip very long pages")
    args = ap.parse_args()

    train_dir = DATA_DIR / "train" / "docatlas"
    (train_dir / "images").mkdir(parents=True, exist_ok=True)
    train_recs = []
    for lang in args.langs.split(","):
        test_dir = DATA_DIR / "test" / f"docatlas_{lang}"
        (test_dir / "images").mkdir(parents=True, exist_ok=True)
        want = args.max_per_lang + args.test_per_lang
        test_recs, kept, offset, failed = [], 0, 0, 0
        while kept < want:
            q = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": "train",
                                        "where": f"\"language\"='{lang}'", "offset": offset, "length": 100})
            d = get_json(f"{API}?{q}")
            batch = d.get("rows", [])
            if not batch:
                break
            offset += len(batch)
            jobs = []
            for r in batch:
                x = r["row"]
                if kept >= want or not x.get("markdown") or len(x["markdown"]) > args.max_chars:
                    continue
                is_test = kept < args.test_per_lang  # the first pages of each language are held out
                base = test_dir if is_test else train_dir
                rel = f"images/{lang}_{x['id']}.jpg"
                rec = {"image": rel, "text": x["markdown"], "kind": "page", "source": "docatlas", "lang": lang}
                jobs.append((x["image"]["src"], base / rel, rec, is_test))
                kept += 1
            # image URLs are signed and expire, so each batch is downloaded right after it is listed
            with ThreadPoolExecutor(8) as ex:
                ok = list(ex.map(lambda j: fetch(j[0], j[1]), jobs))
            for (_, _, rec, is_test), good in zip(jobs, ok):
                if good:
                    (test_recs if is_test else train_recs).append(rec)
                else:
                    failed += 1
            print(f"[{lang}] listed {offset}/{d.get('num_rows_total')}, downloaded {kept - failed}", flush=True)
        write_jsonl(test_dir / "manifest.jsonl", test_recs)
        print(f"[{lang}] test: {len(test_recs)} -> {test_dir / 'manifest.jsonl'}  ({failed} failed downloads skipped)")
    write_jsonl(train_dir / "manifest.jsonl", train_recs)
    print(f"train: {len(train_recs)} -> {train_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
