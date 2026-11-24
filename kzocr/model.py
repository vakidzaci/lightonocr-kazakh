"""Loading LightOnOCR (optionally with a LoRA adapter) and batched OCR generation."""

from __future__ import annotations

import torch
from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

from .common import user_message


def load_model(model_name: str, adapter: str | None = None, device: str | None = None,
               dtype: torch.dtype = torch.bfloat16):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        dtype = torch.float32
    model = LightOnOcrForConditionalGeneration.from_pretrained(model_name, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()  # plain model again: faster generation
    model = model.to(device).eval()
    processor = LightOnOcrProcessor.from_pretrained(adapter or model_name)
    return model, processor


@torch.inference_mode()
def ocr_batch(model, processor, images: list[Image.Image], max_new_tokens: int | list[int] = 4096,
              ) -> list[tuple[str, bool]]:
    """Greedy OCR for a batch of images. Returns [(text, hit_token_limit)]."""
    caps = max_new_tokens if isinstance(max_new_tokens, list) else [max_new_tokens] * len(images)
    processor.tokenizer.padding_side = "left"
    enc = processor.apply_chat_template(
        [[user_message(im)] for im in images], add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", padding=True,
    )
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    enc = {k: (v.to(dev, dtype=dtype) if v.is_floating_point() else v.to(dev)) for k, v in enc.items()}
    out = model.generate(**enc, max_new_tokens=max(caps), do_sample=False, use_cache=True)
    gen = out[:, enc["input_ids"].shape[1]:]
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    results = []
    for row, cap in zip(gen.tolist(), caps):
        n = next((i for i, t in enumerate(row) if t in eos), len(row))
        row = row[: min(n, cap)]
        results.append((processor.decode(row, skip_special_tokens=True), n >= cap))
    return results
