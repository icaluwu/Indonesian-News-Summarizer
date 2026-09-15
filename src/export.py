"""Merge LoRA adapter into base model; optionally export GGUF."""

import argparse
import os
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from unsloth import FastLanguageModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to train.yaml")
    parser.add_argument("--adapter", type=str, default="outputs/adapter", help="LoRA adapter directory")
    parser.add_argument("--out", type=str, default="outputs/merged", help="Output directory for merged model")
    parser.add_argument("--gguf", action="store_true", help="Also export GGUF quantized model")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    max_seq_length = config.get("max_seq_length", 4096)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = PeftModel.from_pretrained(model, args.adapter)

    Path(args.out).mkdir(parents=True, exist_ok=True)

    print("Merging LoRA into base (16-bit) ...")
    model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")
    print(f"Merged 16-bit model saved to {args.out}")

    if args.gguf:
        gguf_out = os.path.join(args.out, "gguf")
        try:
            model.save_pretrained_merged(
                gguf_out,
                tokenizer,
                save_method="merged_4bit_forced_gguf",
                quantization_method="q4_k_m",
            )
            print(f"GGUF (q4_k_m) saved to {gguf_out}")
        except Exception as e:
            print(f"WARNING: GGUF export failed: {e}")


if __name__ == "__main__":
    main()
