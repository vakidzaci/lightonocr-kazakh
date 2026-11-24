"""Download evaluation-only datasets (never train on these) into data/test/<name>/manifest.jsonl.

  real_printed        Omar-2002/kz_benchmark_data, 1,429 real printed/digital Kazakh pages (certificates, letters,
                      forms, worksheets, screenshots). NOTE: references were written by a VLM and not verified by a
                      human, so treat scores as approximate. Images: non-commercial research use only.
  real_handwritten    same benchmark, 150 real handwritten pages (references human-checked).
  ocrbench_printed    issai/OCRBench-Kazakh "Regular Text Recognition" (199, manually collected; scored lowercase).
  ocrbench_handwritten issai/OCRBench-Kazakh "Handwritten Text Recognition" (100; scored lowercase).
  kazakhocr_cyrillic  henrygagnier/kazakh-ocr, Cyrillic subset (2,402 synthetic paragraphs, 406 fonts).

  python data_scripts/download_testsets.py                 # everything (~0.6 GB)
  python data_scripts/download_testsets.py --limit 50      # first 50 samples of each, for a quick check
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

from kzocr.common import DATA_DIR, image_ext, write_jsonl  # noqa: E402

TEST = DATA_DIR / "test"
RAW = TEST / ".raw"


def real_benchmark(limit: int | None) -> None:
    repo, raw = "Omar-2002/kz_benchmark_data", RAW / "kz_benchmark_data"
    for subset, name in (("real_non_handwritten_data", "real_printed"), ("real_handwritten_data", "real_handwritten")):
        csv_path = hf_hub_download(repo, f"{subset}/data.csv", repo_type="dataset", local_dir=raw)
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))[:limit]
        snapshot_download(repo, repo_type="dataset", local_dir=raw,
                          allow_patterns=[f"{subset}/{r['image_path']}" for r in rows])
        recs = [{"image": f"../.raw/kz_benchmark_data/{subset}/{r['image_path']}", "text": r["text"], "kind": "page",
                 "source": "kz_benchmark_data"} for r in rows]
        print(f"{name}: {write_jsonl(TEST / name / 'manifest.jsonl', recs)} samples")


def ocrbench(limit: int | None) -> None:
    path = hf_hub_download("issai/OCRBench-Kazakh", "data/test-00000-of-00001.parquet", repo_type="dataset")
    rows = pq.read_table(path).to_pylist()
    for qtype, name in (("Regular Text Recognition", "ocrbench_printed"),
                        ("Handwritten Text Recognition", "ocrbench_handwritten")):
        sel = [r for r in rows if r["question_type"] == qtype][:limit]
        out = TEST / name
        (out / "images").mkdir(parents=True, exist_ok=True)
        recs = []
        for i, r in enumerate(sel):
            data = r["image"]["bytes"]
            rel = f"images/{i:04d}{image_ext(data)}"
            (out / rel).write_bytes(data)
            recs.append({"image": rel, "text": r["answer"], "kind": "page", "source": "ocrbench_kk", "lowercase": True})
        print(f"{name}: {write_jsonl(out / 'manifest.jsonl', recs)} samples")


def kazakhocr(limit: int | None) -> None:
    repo, raw = "henrygagnier/kazakh-ocr", RAW / "kazakh-ocr"
    meta = hf_hub_download(repo, "metadata.csv", repo_type="dataset", local_dir=raw)
    with open(meta, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["script"] == "cyrillic"][:limit]
    snapshot_download(repo, repo_type="dataset", local_dir=raw, allow_patterns=[r["image_path"] for r in rows])
    recs = [{"image": f"../.raw/kazakh-ocr/{r['image_path']}", "text": r["text"], "kind": "page",
             "source": "kazakhocr", "font": r["font_used"]} for r in rows]
    print(f"kazakhocr_cyrillic: {write_jsonl(TEST / 'kazakhocr_cyrillic' / 'manifest.jsonl', recs)} samples")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="real,ocrbench,kazakhocr")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    todo = set(args.only.split(","))
    if "ocrbench" in todo:
        ocrbench(args.limit)
    if "kazakhocr" in todo:
        kazakhocr(args.limit)
    if "real" in todo:
        real_benchmark(args.limit)


if __name__ == "__main__":
    main()
