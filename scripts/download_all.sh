#!/usr/bin/env bash
# Download and prepare all datasets. Safe to re-run: finished steps are skipped (markers in data/.done/).
#   bash scripts/download_all.sh
# Env vars: TURKIC_CONFIG=tiny|medium|large (default medium), DARMM_MAX_CLEAN (default 40000)
# Approx. totals with defaults: ~45 GB downloaded (mostly streamed), ~25 GB left on disk, 30-90 min.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/.done

step() {  # step <name> <command...>
  local name=$1; shift
  if [ -f "data/.done/$name" ]; then echo "==> $name: already done"; return; fi
  echo "==> $name"
  "$@"
  touch "data/.done/$name"
}

step wikipedia  python data_scripts/download_wikipedia_kk.py
step darmm      python data_scripts/download_darmm.py --splits train,validation,test --max-clean "${DARMM_MAX_CLEAN:-40000}"
step turkicocr  python data_scripts/download_turkicocr.py --config "${TURKIC_CONFIG:-medium}" --splits train,validation,test
step docatlas   python data_scripts/download_docatlas.py --langs ru,en --max-per-lang 3000 --test-per-lang 100
step testsets   python data_scripts/download_testsets.py

echo "==> manifests:"
find data -name "*.jsonl" -not -path "*/.raw/*" | sort | while read -r f; do printf "   %7d  %s\n" "$(wc -l < "$f")" "$f"; done
du -sh data 2>/dev/null | sed 's/^/   total on disk: /'
