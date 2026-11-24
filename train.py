"""Fine-tune LightOnOCR on Kazakh OCR data.

  python train.py --config configs/train_lora.yaml              # LoRA, fits a 24 GB GPU
  python train.py --config configs/train_full.yaml              # full fine-tune of the text decoder, 80 GB GPU
  python train.py --config configs/train_lora.yaml --resume     # continue from the last checkpoint

The image encoder stays frozen (our tests suggest it already "sees" the Kazakh letters; the text decoder is what
falls back on Russian spellings). LoRA trains adapters on the 28 decoder layers (+ optionally the small
vision->text projector); full mode trains the whole decoder and projector.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.utils.checkpoint
import yaml
from transformers import (LightOnOcrForConditionalGeneration, LightOnOcrProcessor, Trainer, TrainingArguments,
                          set_seed)

from kzocr.common import REPO_ROOT
from kzocr.data import IGNORE_INDEX, KindGroupedSampler, OcrCollator, OcrDataset, build_mixture

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.WARNING)
log = logging.getLogger("train")
for name in ("train", "kzocr"):
    logging.getLogger(name).setLevel(logging.INFO)

# LoRA only on the language model's attention + MLP projections (not on the image encoder's same-named layers).
LM_TARGETS = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
LOSS_CHUNK = 1024  # answer tokens per logits chunk


def _ce_sum(h: torch.Tensor, y: torch.Tensor, lm_head: torch.nn.Module) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(lm_head(h).float(), y, reduction="sum")


class OcrTrainer(Trainer):
    """Adds kind-grouped batching and a memory-efficient loss.

    The stock loss computes vocabulary logits (152k entries) at *every* position, including the ~2-3k image tokens of
    a page: ~10 GB for a batch of 4 pages. Here logits are computed only where labels exist (the answer tokens).
    """

    def __init__(self, *args, loss_chunk: int = LOSS_CHUNK, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_chunk = loss_chunk
        self.loss_is_scaled_for_ga = True  # compute_loss below divides by the token count of the whole GA cycle

    def _get_train_sampler(self, train_dataset=None):
        ds = train_dataset if train_dataset is not None else self.train_dataset
        return KindGroupedSampler(ds.kinds, self.args.per_device_train_batch_size, seed=self.args.seed)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        core = getattr(model, "module", model)
        core = core.get_base_model() if hasattr(core, "get_base_model") else core
        hidden = core.model(**inputs, use_cache=False).last_hidden_state
        target = labels[:, 1:]
        mask = target != IGNORE_INDEX
        h, y = hidden[:, :-1][mask], target[mask]
        # Chunked + recomputed in backward: a dense page has ~1-3k answer tokens x 152k vocab, which would otherwise
        # keep several GB of fp32 logits alive per sample.
        loss_sum = sum(
            torch.utils.checkpoint.checkpoint(_ce_sum, h[i:i + self.loss_chunk], y[i:i + self.loss_chunk], core.lm_head,
                                              use_reentrant=False)
            for i in range(0, len(y), self.loss_chunk)
        ) if len(y) else hidden.sum() * 0.0
        n = num_items_in_batch if num_items_in_batch is not None else mask.sum()
        if num_items_in_batch is None and self.model.training:
            n = n * self.args.gradient_accumulation_steps
        loss = loss_sum / torch.as_tensor(n, device=loss_sum.device).clamp(min=1)
        return (loss, {"loss": loss}) if return_outputs else loss


def build_model(cfg: dict):
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.get("model_dtype", "bfloat16")]
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        cfg["model_name"], dtype=dtype, attn_implementation=cfg.get("attn_implementation", "sdpa"))
    model.config.use_cache = False
    for p in model.model.vision_encoder.parameters():
        p.requires_grad = False

    if cfg["mode"] == "lora":
        from peft import LoraConfig, get_peft_model

        lc = cfg.get("lora", {})
        model = get_peft_model(model, LoraConfig(
            r=lc.get("r", 32), lora_alpha=lc.get("alpha", 64), lora_dropout=lc.get("dropout", 0.05),
            target_modules=LM_TARGETS,
            modules_to_save=["vision_projection"] if lc.get("train_projector", True) else None,
        ))
    elif cfg["mode"] == "full":
        for p in model.model.vision_projection.parameters():
            p.requires_grad = cfg.get("full", {}).get("train_projector", True)
    else:
        raise ValueError("mode must be 'lora' or 'full'")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("trainable parameters: %.1fM / %.1fM (%.2f%%)", trainable / 1e6, total / 1e6, 100 * trainable / total)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="resume from the last checkpoint in output_dir")
    ap.add_argument("--max-steps", type=int, default=None, help="override (for smoke tests)")
    args = ap.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # compute_loss calls the model internals directly (to skip image-token logits), which bypasses the DDP
        # wrapper; a 1B model trains fine on one GPU. Use CUDA_VISIBLE_DEVICES=0 on multi-GPU machines.
        raise SystemExit("multi-GPU training is not supported - run with a single GPU")

    cfg = yaml.safe_load(Path(args.config).read_text())
    t = cfg["training"]
    set_seed(t.get("seed", 42))
    out_dir = Path(cfg["output_dir"])
    out_dir = out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir

    processor = LightOnOcrProcessor.from_pretrained(cfg["model_name"])
    train_ds = OcrDataset(build_mixture(cfg["data"]["train"], seed=t.get("seed", 42)))
    val_sources = cfg["data"].get("val") or []
    val_ds = OcrDataset(build_mixture(val_sources, seed=0)) if val_sources else None
    kinds = {k: train_ds.kinds.count(k) for k in set(train_ds.kinds)}
    log.info("train samples: %d %s | val samples: %d", len(train_ds), kinds, len(val_ds) if val_ds else 0)

    model = build_model(cfg)
    hub = cfg.get("hub") or {}
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=t.get("epochs", 1),
        max_steps=args.max_steps or t.get("max_steps", -1),
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t.get("eval_batch_size", t["batch_size"]),
        gradient_accumulation_steps=t.get("grad_accum", 1),
        learning_rate=float(t["lr"]),
        lr_scheduler_type=t.get("scheduler", "cosine"),
        warmup_steps=t.get("warmup", 0.03),
        weight_decay=t.get("weight_decay", 0.0),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        bf16=t.get("bf16", True),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=t.get("logging_steps", 10),
        logging_first_step=True,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=t.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=t.get("save_steps", 500),
        save_total_limit=t.get("save_total_limit", 3),
        dataloader_num_workers=t.get("num_workers", 4),
        dataloader_persistent_workers=t.get("num_workers", 4) > 0,
        remove_unused_columns=False,
        label_names=["labels"],
        prediction_loss_only=True,
        report_to=t.get("report_to", "tensorboard"),
        seed=t.get("seed", 42),
        push_to_hub=bool(hub.get("push")),
        hub_model_id=hub.get("repo_id"),
        hub_private_repo=hub.get("private", True),
        hub_strategy="checkpoint" if hub.get("push") else "every_save",
    )
    trainer = OcrTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                         loss_chunk=t.get("loss_chunk", LOSS_CHUNK),
                         data_collator=OcrCollator(processor, max_length=cfg["data"].get("max_length", 4096)))
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    final = out_dir / "final"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    (final / "train_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    log.info("saved %s model to %s", cfg["mode"], final)
    if hub.get("push"):
        trainer.push_to_hub(commit_message="final model")
    log.info("next: python evaluate.py --model %s %s", cfg["model_name"],
             f"--adapter {final}" if cfg["mode"] == "lora" else f"(or --model {final})")


if __name__ == "__main__":
    sys.exit(main())
