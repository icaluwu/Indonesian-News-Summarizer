"""ROUGE evaluation for the Indonesian news summarizer LoRA adapter."""

import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from rouge_scorer import RougeScorer
from unsloth import FastLanguageModel

from .data import INSTRUCTION, _truncate_article


def _build_prompt(article: str, summary: str, tokenizer, max_seq_length: int) -> str:
    truncated = _truncate_article(
        article, summary, INSTRUCTION, tokenizer.eos_token, tokenizer, max_seq_length
    )
    return f"### Artikel:\n{truncated}\n\n{INSTRUCTION}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to train.yaml")
    parser.add_argument("--adapter", type=str, default="outputs/adapter", help="LoRA adapter directory")
    parser.add_argument("--n", type=int, default=50, help="Number of samples to evaluate")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 3407)
    model_name = config["model_name"]
    max_seq_length = config.get("max_seq_length", 4096)
    eval_samples = config.get("eval_samples", 200)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.for_inference(model)
    model = PeftModel.from_pretrained(model, args.adapter)

    dataset = load_dataset(config["dataset_name"], config["dataset_config"])
    ds = dataset["train"].shuffle(seed=seed)
    if len(ds) >= eval_samples:
        val_ds = ds["train"].train_test_split(test_size=eval_samples, seed=seed)["test"]
        val_ds = val_ds.select(range(eval_samples))
    else:
        val_ds = ds

    samples = val_ds.select(range(min(args.n, len(val_ds))))

    rouge_scorer = RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=False)

    predictions = []
    references = []

    for i, example in enumerate(samples):
        article = example["article"].strip()
        summary = example["summary"].strip()
        prompt = _build_prompt(article, summary, tokenizer, max_seq_length)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = outputs[0][input_len:]
        pred_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        if "### Ringkasan:" in pred_text:
            pred_text = pred_text.split("### Ringkasan:")[-1].strip()

        predictions.append(pred_text)
        references.append(summary)

        if i < 3:
            print(f"\n--- Sample {i + 1} ---")
            print(f"Reference: {summary}")
            print(f"Prediction:  {pred_text}")

    scores = [rouge_scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
    means = {key: sum(s[key].fmeasure for s in scores) / len(scores) for key in rouge_scorer._types}

    print("\n=== ROUGE Scores ===")
    print(f"{'Metric':<10} {'Mean F1':>10}")
    print("-" * 25)
    for key in ["rouge1", "rouge2", "rougeLsum"]:
        label = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeLsum": "ROUGE-L"}[key]
        print(f"{label:<10} {means[key]:>10.4f}")

    print(f"\nEvaluated {len(predictions)} samples.")


if __name__ == "__main__":
    main()
