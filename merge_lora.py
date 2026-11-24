"""Merge a LoRA adapter into the base model so it loads like any LightOnOCR checkpoint (and works with vLLM),
then optionally push it to the Hugging Face Hub.

  python merge_lora.py --adapter outputs/lora/final --out outputs/lora/merged
  python merge_lora.py --adapter outputs/lora/final --out outputs/lora/merged --push your-name/lightonocr-2-1b-kazakh
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--base", help="base model (default: model_name from the adapter's train_config.yaml)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--push", help="Hub repo id to upload to")
    ap.add_argument("--public", action="store_true", help="create the Hub repo as public (default private)")
    args = ap.parse_args()

    base = args.base
    if base is None:
        cfg_file = args.adapter / "train_config.yaml"
        base = yaml.safe_load(cfg_file.read_text())["model_name"] if cfg_file.exists() else None
        if base is None:
            raise SystemExit("pass --base (no train_config.yaml next to the adapter)")
    model = LightOnOcrForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    processor = LightOnOcrProcessor.from_pretrained(str(args.adapter))
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"merged model saved to {args.out}")
    if args.push:
        model.push_to_hub(args.push, private=not args.public)
        processor.push_to_hub(args.push, private=not args.public)
        print(f"pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
