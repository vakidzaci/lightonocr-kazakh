"""Generate synthetic Kazakh OCR training data, or the fixed "reading probe" test sets.

Training data (lines + pages, with train/val split):
    python -m generator.generate --lines 40000 --pages 4000 --workers 8

Reading probes (fixed seed, identical for every model you evaluate):
    python -m generator.generate --probe

Text modes and their share (override with --mix): natural, misspelled, pseudo (made-up + mutated words),
entities (names, addresses, dates, IDs). See generator/textgen.py for why each one exists.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
from pathlib import Path

from kzocr.common import DATA_DIR, write_jsonl

from .fonts import FontPicker, find_fonts
from .render import LEVELS, render_line, render_page, render_probe_page
from .textgen import TextSource

DEFAULT_MIX = "natural=0.25,misspelled=0.2,pseudo=0.35,entities=0.2"
PROBE_LINES = 10
_W: dict = {}


def _init(fonts, mix, out_dir):
    _W["src"] = TextSource()
    _W["picker"] = FontPicker(fonts)
    _W["modes"], _W["weights"] = zip(*mix)
    _W["out"] = out_dir


def _make(job: tuple[int, str, int, str | None, str | None]) -> dict:
    idx, kind, seed, forced_mode, forced_level = job
    rng = random.Random(seed)
    src, picker = _W["src"], _W["picker"]
    level = forced_level or rng.choices(LEVELS, weights=[0.3, 0.45, 0.25])[0]
    rec = {"kind": kind, "source": "synthetic", "level": level}
    if kind == "line":
        mode = forced_mode or rng.choices(_W["modes"], weights=_W["weights"])[0]
        text, orig = src.text(rng, mode, rng.randint(12, 75))
        img = render_line(text, rng, picker, level)
        rec.update(mode=mode, text=text)
        if orig is not None:
            rec["orig_text"] = orig
    elif kind == "probe":
        pairs = [src.text(rng, forced_mode, rng.randint(20, 48)) for _ in range(PROBE_LINES)]
        img = render_probe_page([p for p, _ in pairs], rng, picker)
        rec.update(kind="page", mode=forced_mode, text="\n".join(p for p, _ in pairs))
        if forced_mode == "misspelled":
            rec["orig_text"] = "\n".join(o for _, o in pairs)
    else:
        img, text = render_page(src, rng, picker, level, list(_W["modes"]), list(_W["weights"]))
        rec.update(mode="page", text=text)
    name = f"images/{idx // 1000:04d}/{idx:07d}.jpg"
    path = _W["out"] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=93)
    rec["image"] = name
    return rec


def _cgroup_quota(d: Path, v2: bool) -> float | None:
    """CPU quota of one cgroup directory, in CPUs; None if it has none."""
    try:
        if v2:
            quota, period = (d / "cpu.max").read_text().split()  # "<quota> <period>" or "max <period>"
            return None if quota == "max" else int(quota) / int(period)
        quota = int((d / "cpu.cfs_quota_us").read_text())  # -1 = no quota
        return quota / int((d / "cpu.cfs_period_us").read_text()) if quota > 0 else None
    except (OSError, ValueError, ZeroDivisionError):
        return None


def usable_cpus() -> int:
    """CPUs this process can really use: its affinity mask, capped by any cgroup CPU quota up to the root.

    In a container (vast.ai, `docker --cpus`) os.cpu_count() reports every core of the host, not the
    container's share, so a pool sized from it starts far too many workers.
    """
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return n
    for line in lines:
        _, controllers, rel = line.split(":", 2)
        v2 = controllers == ""
        if not v2 and "cpu" not in controllers.split(","):
            continue
        root = Path("/sys/fs/cgroup" if v2 else "/sys/fs/cgroup/cpu")
        d = root / rel.strip().lstrip("/")
        while True:  # a limit may sit on a parent cgroup; inside a container the path may not exist -> root
            q = _cgroup_quota(d, v2)
            if q is not None:
                n = min(n, max(1, int(q)))
            if d == root or d == d.parent:
                break
            d = d.parent
    return n


def _run(jobs, fonts, mix, out_dir: Path, workers: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        _init(fonts, mix, out_dir)
        return [_make(j) for j in jobs]
    with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(fonts, mix, out_dir)) as pool:
        recs = []
        for i, r in enumerate(pool.imap(_make, jobs, chunksize=32), 1):
            recs.append(r)
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
        return recs


def parse_mix(s: str) -> list[tuple[str, float]]:
    return [(k, float(v)) for k, v in (p.split("=") for p in s.split(","))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lines", type=int, default=40000)
    ap.add_argument("--pages", type=int, default=4000)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--mix", default=DEFAULT_MIX)
    ap.add_argument("--workers", type=int, default=max(1, usable_cpus() - 1),
                    help="worker processes (default: usable CPUs - 1, respecting container CPU limits)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "train" / "synthetic")
    ap.add_argument("--probe", action="store_true", help="generate the fixed reading-probe test sets instead")
    ap.add_argument("--probe-size", type=int, default=60, help=f"pages per probe set ({PROBE_LINES} lines each)")
    args = ap.parse_args()

    fonts = find_fonts()
    mix = parse_mix(args.mix)
    print(f"{len(fonts)} fonts with Kazakh coverage; mix: {mix}")

    if args.probe:
        for name, mode in (("probe_correct", "natural"), ("probe_misspelled", "misspelled"), ("probe_made_up", "pseudo")):
            out = DATA_DIR / "test" / name
            jobs = [(i, "probe", 10_000_000 + i, mode, "clean") for i in range(args.probe_size)]
            recs = _run(jobs, fonts, mix, out, args.workers)
            write_jsonl(out / "manifest.jsonl", recs)
            print(f"{name}: {len(recs)} samples -> {out / 'manifest.jsonl'}")
        return

    rng = random.Random(args.seed)
    jobs = [(i, "line", rng.randint(0, 2**31), None, None) for i in range(args.lines)]
    jobs += [(args.lines + i, "page", rng.randint(0, 2**31), None, None) for i in range(args.pages)]
    print(f"rendering {args.lines} lines + {args.pages} pages into {args.out} with {args.workers} workers ...")
    recs = _run(jobs, fonts, mix, args.out, args.workers)
    random.Random(args.seed).shuffle(recs)
    n_val = int(len(recs) * args.val_fraction)
    write_jsonl(args.out / "manifest.jsonl", recs[n_val:])
    val_dir = DATA_DIR / "val" / "synthetic"
    val_dir.mkdir(parents=True, exist_ok=True)
    rel = Path("../../train/synthetic") if args.out == DATA_DIR / "train" / "synthetic" else args.out.resolve()
    write_jsonl(val_dir / "manifest.jsonl", [{**r, "image": str(rel / r["image"])} for r in recs[:n_val]])
    print(f"train: {len(recs) - n_val} -> {args.out / 'manifest.jsonl'}   val: {n_val} -> {val_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
