#!/usr/bin/env bash
# Whole pipeline: data -> synthetic data -> baseline eval -> train -> eval -> (merge). Re-runnable: finished steps
# are skipped and training resumes from its last checkpoint. Run it inside tmux so it survives disconnects:
#   tmux new -s ocr "bash scripts/run_all.sh configs/train_lora.yaml 2>&1 | tee -a run.log"
# Env vars: SYN_LINES (default 40000), SYN_PAGES (4000), EVAL_LIMIT (300 samples per test set)
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:-configs/train_lora.yaml}
cfg() { python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))$1)"; }
BASE=$(cfg "['model_name']"); OUT=$(cfg "['output_dir']"); MODE=$(cfg "['mode']"); RUN=$(basename "$OUT")
EVAL_LIMIT=${EVAL_LIMIT:-300}

bash scripts/download_all.sh

if [ ! -f data/train/synthetic/manifest.jsonl ]; then
  echo "==> synthetic training data"
  python -m generator.generate --lines "${SYN_LINES:-40000}" --pages "${SYN_PAGES:-4000}"
fi
if [ ! -f data/test/probe_correct/manifest.jsonl ]; then
  echo "==> reading probes"
  python -m generator.generate --probe
fi

if [ ! -f outputs/eval/baseline-n$EVAL_LIMIT/results.json ]; then
  echo "==> baseline evaluation of $BASE"
  python evaluate.py --model "$BASE" --limit "$EVAL_LIMIT" --out "outputs/eval/baseline-n$EVAL_LIMIT"
fi

echo "==> training ($CONFIG)"
RESUME=""; ls "$OUT"/checkpoint-* >/dev/null 2>&1 && RESUME="--resume"
[ -d "$OUT/final" ] || python train.py --config "$CONFIG" $RESUME

echo "==> evaluation of the fine-tuned model"
if [ "$MODE" = "lora" ]; then
  python evaluate.py --model "$BASE" --adapter "$OUT/final" --limit "$EVAL_LIMIT" --out "outputs/eval/$RUN-n$EVAL_LIMIT"
  [ -d "$OUT/merged" ] || python merge_lora.py --adapter "$OUT/final" --out "$OUT/merged"
else
  python evaluate.py --model "$OUT/final" --limit "$EVAL_LIMIT" --out "outputs/eval/$RUN-n$EVAL_LIMIT"
fi

python scripts/compare_results.py "outputs/eval/baseline-n$EVAL_LIMIT" "outputs/eval/$RUN-n$EVAL_LIMIT"
