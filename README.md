# finetune-kazakh-ocr-light-on

Fine-tune [LightOnOCR](https://huggingface.co/lightonai/LightOnOCR-2-1B) (a 1B end-to-end OCR vision-language
model) so it **reads Kazakh Cyrillic letters instead of guessing them**.

## Why

LightOnOCR is not trained on Kazakh. Our tests showed:

* On ordinary Cyrillic it reads well, but the 9 Kazakh letters **Ә Ғ Қ Ң Ө Ұ Ү Һ І** are often swapped for Russian
  lookalikes (ғ→г, ұ→у/ү, қ→к, ң→н, ө→о).
* On made-up words (nothing to guess from) only 49–70 % of Kazakh letters were read correctly vs ~90–96 % of ordinary letters.
* LightOnOCR-3 partly hides this by *recognising words*: it "fixes" printed text toward words it knows
  (even `Казакстан` → Russian `Казахстан`), and fails on names and rare words.

So the training data deliberately contains text where guessing cannot work - made-up words, words with confusable
letters, deliberate misspellings, names, addresses and IDs - alongside real Kazakh documents.

## Status

Every part of the pipeline has been tested end to end on small data on a 4 GB laptop GPU: all download scripts,
the generator, training on line crops and on full pages (LoRA), evaluation on every test set, merging and inference.
**A full-scale training run has not been done yet** - the numbers it produces are the open question.
Measured memory for planning: image encoder ≈0.4 GB for 1 page, 1.1 GB for 2 (grows faster than linearly), so
`batch_size: 4` fits 24 GB and `batch_size: 8` fits 80 GB.

## Quick start on vast.ai

**1. Rent an instance** (template: *PyTorch*, CUDA ≥ 12.8 image)

| Run | GPU | Disk |
|---|---|---|
| LoRA (`configs/train_lora.yaml`, `train_lora_pilot.yaml`) | 1× **RTX 4090 24 GB** (or RTX 5090 32 GB, A100) | 100 GB |
| Full fine-tune (`configs/train_full.yaml`) | 1× **A100 80 GB** or H100 80 GB | 150 GB |

Pick a host with fast internet (≥ 500 Mbps): ~45 GB of data is downloaded (mostly streamed). Use **one GPU** - the
model has only 1B parameters, and multi-GPU training is not supported (on a multi-GPU box: `export CUDA_VISIBLE_DEVICES=0`).

**2. Set up and run** (inside `tmux`, so it survives SSH disconnects)

```bash
git clone <your-repo-url> finetune-kazakh-ocr-light-on && cd finetune-kazakh-ocr-light-on
bash scripts/setup_vastai.sh                 # fonts + python packages + GPU check
export HF_TOKEN=hf_...                       # optional: only needed to push models to the Hub

tmux new -s ocr
EVAL_LIMIT=100 bash scripts/run_all.sh configs/train_lora_pilot.yaml 2>&1 | tee -a pilot.log   # is it working?
bash scripts/run_all.sh configs/train_lora.yaml 2>&1 | tee -a run.log                          # the real run
```

`run_all.sh` = download data → generate synthetic data → evaluate the base model → train → evaluate → merge LoRA →
print a before/after table. Every step is skipped if already done, and training resumes from its last checkpoint,
so you can simply re-run it after an interruption. The first run also downloads everything (30–90 min) and
evaluates the base model; the pilot's training itself is ~1–2 h on an RTX 4090. `EVAL_LIMIT` = samples per test
set (default 300; 14 test sets, so evaluation of a model takes roughly an hour at the default).

**3. Get the results** - `outputs/eval/*/results.json`, the merged model in `outputs/<run>/merged/`
(copy it off the instance or push it: `python merge_lora.py --adapter outputs/lora-r32/final --out outputs/lora-r32/merged --push your-name/lightonocr-2-1b-kazakh`).

Monitor training: `tensorboard --logdir outputs/ --port 6006` (open the port in the vast.ai UI), or watch the log.

## Step by step

```bash
bash scripts/download_all.sh                                   # all datasets (re-runnable)
python -m generator.generate --lines 40000 --pages 4000        # synthetic training data
python -m generator.generate --probe                           # fixed "reading probe" test sets
python evaluate.py --model lightonai/LightOnOCR-2-1B-base --out outputs/eval/baseline
python train.py --config configs/train_lora.yaml               # add --resume to continue
python evaluate.py --model lightonai/LightOnOCR-2-1B-base --adapter outputs/lora-r32/final --out outputs/eval/lora-r32
python scripts/compare_results.py outputs/eval/baseline outputs/eval/lora-r32
python merge_lora.py --adapter outputs/lora-r32/final --out outputs/lora-r32/merged
python infer.py my_scan.jpg --model outputs/lora-r32/merged
```

## Data

| Source | What | Used for | Size kept | License |
|---|---|---|---|---|
| Synthetic (`generator/`) | Lines + pages: natural text, misspellings, made-up / mutated words, names, addresses, dates, IDs, tables, stamps; 20+ font families, scan/photo degradations | train | 44k (configurable) | yours |
| [Darmm/darmm-ocr-kazakh-v2](https://huggingface.co/datasets/Darmm/darmm-ocr-kazakh-v2) | Kazakh Wikipedia text rendered as word/line crops (clean + degraded) and pages | train / val / test | ~117k train | Apache-2.0 |
| [alenisaw/turkicocr-cyrillic](https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic) | Synthetic official documents: certificates, letters, forms, tables, stamps (Kazakh pages only) | train / val / test | ~24k pages (`medium`) | CC-BY-4.0 |
| [ahmedheakl/docatlas_instruct](https://huggingface.co/datasets/ahmedheakl/docatlas_instruct) | Real Russian + English documents (exact labels) - keeps the model's old skills | train / test | 6k pages | CC-BY-4.0 |
| [Wikipedia kk](https://huggingface.co/datasets/wikimedia/wikipedia) | Text for the generator | generator | 400k sentences | CC-BY-SA-4.0 |
| **Your own pages** (`data_scripts/prepare_custom.py`) | e.g. scanned books with checked transcriptions | train / test | - | yours |

Evaluation only (never trained on):

| Test set | What |
|---|---|
| `probe_correct`, `probe_misspelled`, `probe_made_up` | Reading probes (see below) |
| `real_printed`, `real_handwritten` | [Omar-2002/kz_benchmark_data](https://huggingface.co/datasets/Omar-2002/kz_benchmark_data): 1,429 real printed + 150 handwritten Kazakh pages. Printed references were written by a VLM and not human-checked → approximate. |
| `ocrbench_printed`, `ocrbench_handwritten` | [issai/OCRBench-Kazakh](https://huggingface.co/datasets/issai/OCRBench-Kazakh) text recognition (scored lowercase) |
| `kazakhocr_cyrillic` | [henrygagnier/kazakh-ocr](https://huggingface.co/datasets/henrygagnier/kazakh-ocr) Cyrillic subset |
| `darmm_{clean,hard,page}`, `turkicocr_kk` | held-out splits of the training sources |
| `docatlas_ru`, `docatlas_en` | did Russian/English get worse? |
| `custom` | your held-out books |

**Reading the numbers:** `darmm_clean`, `darmm_hard` and `ocrbench_*` are tiny single-word/line images. The base
model often slips into LaTeX/Latin output on such crops (`$\mathcal{E}…$`, `KA3AXINHSTPAX`), so its scores there are
very poor and part of the fine-tuning gain is just learning that format. Judge *reading* on the page-level sets
(`real_printed`, `turkicocr_kk`, `darmm_page`, `kazakhocr_cyrillic`, `custom`) and the probes.

### Adding your own labeled pages (recommended)

Real, human-checked pages are the most valuable data. One folder per book, each image next to a `.txt` with the same
name; hold out whole books for testing:

```bash
python data_scripts/prepare_custom.py my_books/ --test-books abai_joldary
```

The configs already include `data/train/custom/manifest.jsonl` (repeated 3×) when it exists.

## The reading probes

`python -m generator.generate --probe` renders three fixed test sets (60 pages × 10 lines, fixed seed):

* **probe_correct** - real Kazakh text. Reading and guessing agree.
* **probe_misspelled** - Kazakh letters printed as Russian lookalikes (`Қазақстан` → `Казакстан`). A faithful OCR
  copies them: `copied_misspelling` should be high, `restored_correct_spelling` (silently "fixed") near 0.
* **probe_made_up** - made-up words: there is no word to guess, so `kazakh_letter_recall` here measures pure reading.

The goal of fine-tuning: Kazakh-letter recall on `probe_made_up` close to that of ordinary letters (~95 %).

## Training details

* **Model**: `lightonai/LightOnOCR-2-1B-base` (LightOn's recommended starting point for fine-tuning). The same code
  works for `LightOnOCR-2-1B` and `LightOnOCR-3-1B` (same architecture); LightOnOCR-3-0.8B/4B (Qwen3.5) are not supported.
* **What is trained**: the image encoder is frozen; LoRA (r=32) on all 28 text-decoder layers + the small
  image→text projector, or (full mode) the whole decoder + projector.
* **Loss**: only on the answer tokens, and logits are computed only there - not over the ~2-3k image tokens of a
  page (saves ~10 GB of GPU memory per batch of 4 pages).
* **Batching**: word/line crops and full pages go into separate batches (no padding crops to page size).
* **Targets** follow LightOnOCR's own output style: Markdown headings, HTML tables, blocks separated by blank lines.
  Stamps are not transcribed.
* **Sizes**: the default mix is ~175k samples ≈ 11k optimizer steps (effective batch 16). Speed depends heavily on the
  page/crop ratio - the trainer prints an ETA after the first steps; run the pilot first to measure.

Out of memory? Lower `training.batch_size` (and raise `grad_accum` to keep the effective batch), or lower
`data.max_length`.

## Using the result

```bash
python infer.py scan.jpg book.pdf --pages 1-3 --model outputs/lora-r32/merged --out results/
```

The merged model is a normal LightOnOCR checkpoint, so it also works with vLLM
(`vllm serve outputs/lora-r32/merged --limit-mm-per-prompt '{"image": 1}'`) and the original model card examples.

## Repository layout

```
configs/            training configs (pilot, LoRA, full, local smoke test)
data_scripts/       dataset downloaders + prepare_custom.py for your own pages
generator/          synthetic Kazakh text + rendering (fonts.py, textgen.py, render.py, generate.py)
kzocr/              shared code: data pipeline, metrics, model loading/generation
scripts/            setup_vastai.sh, download_all.sh, run_all.sh, compare_results.py
train.py  evaluate.py  infer.py  merge_lora.py
data/  outputs/     created at run time (git-ignored)
```

Local smoke test (any GPU with ≥ 4 GB): `python data_scripts/download_wikipedia_kk.py && python -m generator.generate --lines 300 --pages 0 && python train.py --config configs/smoke_test.yaml`.

## Licenses and attribution

Code in this repository: choose a license before publishing (Apache-2.0 matches LightOnOCR's).
The base model is Apache-2.0. Training sources are Apache-2.0 / CC-BY-4.0 / CC-BY-SA-4.0 - credit them
(Darmm, alenisaw/TurkicOCR, DocAtlas, Wikipedia) in your model card. `kz_benchmark_data` images are for
non-commercial research evaluation only, which is why it is used only as a test set.
