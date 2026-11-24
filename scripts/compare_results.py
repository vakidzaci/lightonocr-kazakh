"""Side-by-side comparison of evaluate.py results.

  python scripts/compare_results.py outputs/eval/baseline outputs/eval/lora-r32 [more runs ...]
"""

import json
import sys
from pathlib import Path


def main() -> None:
    runs = [Path(p) for p in sys.argv[1:]]
    if not runs:
        sys.exit(__doc__)
    data = {r.name: json.loads((r / "results.json").read_text())["results"] for r in runs}
    sets = sorted({s for res in data.values() for s in res})
    cols = list(data)
    metrics = [("cer", "CER", True), ("kazakh_letter_recall", "Kazakh letters", False),
               ("copied_misspelling", "misspelled words copied as printed", False),
               ("restored_correct_spelling", "misspelled words silently 'fixed'", True), ("runaway_generations", "runaways", True)]
    for key, label, lower_is_better in metrics:
        rows = [(s, [data[c].get(s, {}).get(key) for c in cols]) for s in sets]
        rows = [(s, v) for s, v in rows if any(x is not None for x in v)]
        if not rows:
            continue
        print(f"\n{label} ({'lower' if lower_is_better else 'higher'} is better)")
        print(f"  {'test set':<24}" + "".join(f"{c[:18]:>20}" for c in cols))
        for s, vals in rows:
            fmt = [("-" if v is None else f"{v}" if key == "runaway_generations" else f"{v:.2%}") for v in vals]
            print(f"  {s:<24}" + "".join(f"{x:>20}" for x in fmt))


if __name__ == "__main__":
    main()
