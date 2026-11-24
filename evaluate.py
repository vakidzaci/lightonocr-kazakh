"""Evaluate a LightOnOCR model (base, LoRA adapter, or merged) on the test sets in data/test/.

  python evaluate.py --model lightonai/LightOnOCR-2-1B-base                       # baseline
  python evaluate.py --model lightonai/LightOnOCR-2-1B-base --adapter outputs/lora/final
  python evaluate.py --model outputs/full/final --sets probe_made_up,real_printed --limit 100

Sets = folder names under data/test (each with manifest.jsonl). Default: every set found.
Reading probes (python -m generator.generate --probe) show whether the model reads letters or guesses words:
  probe_correct     real Kazakh text
  probe_misspelled  Kazakh letters printed as Russian lookalikes: a good model copies them exactly
                    (per misspelled word: copied_misspelling - want high; restored_correct_spelling = silently
                    "fixed" back to proper Kazakh - want ~0)
  probe_made_up     made-up words: no word to guess, Kazakh-letter recall here = pure reading ability

Generation is greedy. To stop runaway repetition loops from dominating, generation is capped at
2x the reference length (+64 tokens); runs that hit the cap are reported as runaway_generations.
Results: printed as a table and saved to <out>/results.json and <out>/<set>.jsonl (per-sample outputs).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kzocr.common import DATA_DIR, fit_image, load_image, load_manifest
from kzocr.metrics import Scorer
from kzocr.model import load_model, ocr_batch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id or local path (base or merged model)")
    ap.add_argument("--adapter", help="LoRA adapter dir (outputs/<run>/final)")
    ap.add_argument("--sets", default="all")
    ap.add_argument("--limit", type=int, default=300, help="max samples per set (0 = all)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    test_dir = DATA_DIR / "test"
    names = sorted(p.parent.name for p in test_dir.glob("*/manifest.jsonl")) if args.sets == "all" \
        else args.sets.split(",")
    if not names:
        raise SystemExit(f"no test sets in {test_dir} - run the download scripts / generator --probe first")
    tag = Path(args.adapter or args.model).name if (args.adapter or Path(args.model).exists()) else args.model.split("/")[-1]
    out = args.out or Path("outputs") / "eval" / tag
    out.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model, args.adapter)
    tok = processor.tokenizer
    results = {}
    for name in names:
        rows = load_manifest(test_dir / name / "manifest.jsonl")
        rows = rows[: args.limit] if args.limit else rows
        scorer = Scorer(lowercase=bool(rows and rows[0].get("lowercase")))
        t0 = time.perf_counter()
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            # batch similar-sized samples together (sort by reference length)
            order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))
            for s in range(0, len(order), args.batch_size):
                batch = [rows[i] for i in order[s: s + args.batch_size]]
                images = [fit_image(load_image(r["image"]), r.get("kind", "page")) for r in batch]
                caps = [min(args.max_new_tokens, 2 * len(tok.encode(r["text"])) + 64) for r in batch]
                for r, (hyp, capped) in zip(batch, ocr_batch(model, processor, images, caps)):
                    m = scorer.add(r["text"], hyp, capped, r.get("orig_text"))
                    f.write(json.dumps({"image": r["image"], "ref": r["text"], "hyp": hyp, "runaway": capped, **m},
                                       ensure_ascii=False) + "\n")
        res = scorer.summary()
        res["seconds"] = round(time.perf_counter() - t0, 1)
        results[name] = res
        kz = f"{res['kazakh_letter_recall']:.1%}" if res["kazakh_letter_recall"] is not None else "  -  "
        extra = (f"  misspellings copied={res['copied_misspelling']:.0%} restored={res['restored_correct_spelling']:.0%}"
                 if "copied_misspelling" in res else "")
        print(f"{name:<22} n={res['samples']:<4} CER={res['cer']:6.2%}  WER={res['wer']:6.2%}  "
              f"kz-letters={kz}  exact={res['exact_match']:.0%}  runaways={res['runaway_generations']}{extra}",
              flush=True)

    meta = {"model": args.model, "adapter": args.adapter, "limit": args.limit}
    (out / "results.json").write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2))
    print(f"\nsaved {out / 'results.json'}")


if __name__ == "__main__":
    main()
