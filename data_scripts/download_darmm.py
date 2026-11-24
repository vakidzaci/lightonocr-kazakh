"""Download Darmm/darmm-ocr-kazakh-v2 (synthetic Kazakh Cyrillic, Apache-2.0) and convert it to image files +
JSONL manifests.

Tiers: clean (word/line crops), hard (degraded word/line crops), page (full pages).
Outputs:
  train      -> data/train/darmm/{clean,hard,page}.jsonl
  validation -> data/val/darmm/{clean,hard,page}.jsonl
  test       -> data/test/darmm_{clean,hard,page}/manifest.jsonl     (held out - never train on it)

Example (defaults keep all hard crops and pages, and 40k of the 190k clean crops):
  python data_scripts/download_darmm.py --splits train,validation,test
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from kzocr.common import DATA_DIR, image_ext, write_jsonl  # noqa: E402

REPO = "Darmm/darmm-ocr-kazakh-v2"
EXPECTED = {  # approximate tier sizes from the dataset card, used to subsample uniformly
    "train": {"clean": 190_000, "hard": 66_000, "page": 11_000},
    "validation": {"clean": 8_000, "hard": 2_000, "page": 500},
    "test": {"clean": 2_000, "hard": 2_000, "page": 500},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="train,validation,test")
    ap.add_argument("--max-clean", type=int, default=40_000, help="per split cap (train); val/test keep all")
    ap.add_argument("--max-hard", type=int, default=None)
    ap.add_argument("--max-page", type=int, default=None)
    ap.add_argument("--max-shards", type=int, default=None, help="for quick tests")
    ap.add_argument("--keep-parquet", action="store_true")
    args = ap.parse_args()

    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
    tmp = DATA_DIR / ".downloads" / "darmm"
    for split in args.splits.split(","):
        shards = [f for f in files if Path(f).name.startswith(split + "-")][: args.max_shards]
        caps = {"clean": args.max_clean, "hard": args.max_hard, "page": args.max_page} if split == "train" else {}
        rng = random.Random(split)
        keep_p = {t: min(1.0, (caps.get(t) or 10**9) / EXPECTED[split][t] * 1.05) for t in EXPECTED[split]}
        out_dirs = {t: (DATA_DIR / {"train": "train", "validation": "val"}[split] / "darmm" if split != "test"
                        else DATA_DIR / "test" / f"darmm_{t}") for t in EXPECTED[split]}
        recs: dict[str, list] = {t: [] for t in EXPECTED[split]}
        n = 0
        for shard in shards:
            print(f"[{split}] {shard}", flush=True)
            local = hf_hub_download(REPO, shard, repo_type="dataset", local_dir=tmp)
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=256, columns=["image", "label", "level", "tier"]):
                for row in batch.to_pylist():
                    tier = row["tier"]
                    cap = caps.get(tier)
                    if (cap is not None and len(recs[tier]) >= cap) or rng.random() > keep_p[tier]:
                        continue
                    data = row["image"]["bytes"]
                    sub = f"images/{tier}/{n // 1000:04d}/{n:07d}{image_ext(data)}" if split != "test" \
                        else f"images/{n:07d}{image_ext(data)}"
                    path = out_dirs[tier] / sub
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    recs[tier].append({"image": sub, "text": row["label"], "kind": "page" if tier == "page" else "line",
                                       "source": "darmm", "level": row["level"], "tier": tier})
                    n += 1
            if not args.keep_parquet:
                Path(local).unlink(missing_ok=True)
        for tier, rows in recs.items():
            if not rows:
                continue
            man = out_dirs[tier] / (f"{tier}.jsonl" if split != "test" else "manifest.jsonl")
            write_jsonl(man, rows)
            print(f"[{split}] {tier}: {len(rows)} samples -> {man}")
    if not args.keep_parquet:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
