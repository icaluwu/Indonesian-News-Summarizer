"""Fine-tune Qwen3.8-27B with QLoRA on Indonesian news summarization data."""

import argparse
import os
from pathlib import Path

import torch
import yaml
from transformers import TrainingArguments
from trl import SFTTrainer

from .data import get_datasets
from .model import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to train.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 3407)
    torch.manual_seed(seed)

    model, tokenizer = load_model(config)

    train_ds, val_ds = get_datasets(config, tokenizer)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} | Total: {total_params:,} | Trainable/Total: {100*trainable/total_params:.2f}%")

    output_dir = config.get("output_dir", "outputs")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 2.0e-4),
        num_train_epochs=config.get("num_train_epochs", 1),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        weight_decay=config.get("weight_decay", 0.01),
        optim=config.get("optim", "adamw_8bit"),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 250),
        save_total_limit=config.get("save_total_limit", 2),
        seed=seed,
        report_to="none",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        evaluation_strategy="steps",
        eval_strategy="steps",
        eval_steps=config.get("save_steps", 250),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        args=training_args,
        max_seq_length=config.get("max_seq_length", 4096),
        packing=False,
    )

    trainer.train()

    adapter_path = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")

    hf_repo_id = config.get("hf_repo_id")
    hf_token = os.environ.get("HF_TOKEN", "")

    if config.get("push_adapter") and hf_repo_id:
        model.push_to_hub_merged(
            hf_repo_id,
            tokenizer,
            save_method="lora",
            token=hf_token,
        )
        print(f"LoRA adapter pushed to {hf_repo_id}")

    if config.get("push_merged") and hf_repo_id:
        try:
            model.push_to_hub_merged(
                hf_repo_id,
                tokenizer,
                save_method="merged_16bit",
                token=hf_token,
            )
            print(f"Merged 16-bit model pushed to {hf_repo_id}")
        except torch.cuda.OutOfMemoryError:
            print(
                "WARNING: Merged 16-bit push failed due to OOM on this GPU "
                "(2x T4 16GB). The LoRA adapter was already pushed. "
                "Use src/export.py --gguf on a machine with more VRAM "
                "to produce a GGUF quantization instead."
            )
        except Exception as e:
            print(f"WARNING: Merged push failed: {e}")

    print("Training complete.")


if __name__ == "__main__":
    main()
