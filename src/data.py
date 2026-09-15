"""Dataset preparation for Indonesian news summarization."""

import yaml
from datasets import load_dataset


INSTRUCTION = (
    "### Instruksi:\n"
    "Ringkas artikel berita di atas menjadi 1-3 kalimat dalam Bahasa Indonesia.\n\n"
    "### Ringkasan:\n"
)


def _build_training_text(article: str, summary: str, eos_token: str) -> str:
    return f"### Artikel:\n{article}\n\n{INSTRUCTION}{summary}{eos_token}"


def _truncate_article(
    article: str,
    summary: str,
    instruction_template: str,
    eos_token: str,
    tokenizer,
    max_seq_length: int,
) -> str:
    """Keep the first N article tokens so the full summary always fits."""
    overhead_tokens = tokenizer.encode(instruction_template + summary + eos_token, add_special_tokens=False)
    available = max_seq_length - len(overhead_tokens) - 2
    if available < 64:
        available = 64

    article_ids = tokenizer.encode(article, add_special_tokens=False)[:available]
    return tokenizer.decode(article_ids, skip_special_tokens=True).strip()


def get_datasets(config, tokenizer):
    """Load, split, and tokenize the XL-Sum Indonesian subset.

    Returns
    -------
    (train_ds, val_ds) : Hugging Face ``Dataset`` objects with a single
    ``text`` column ready for ``SFTTrainer(dataset_text_field="text")``.
    """
    dataset = load_dataset(config["dataset_name"], config["dataset_config"])

    max_seq_length = config["max_seq_length"]
    train_samples = config["train_samples"]
    eval_samples = config["eval_samples"]
    seed = config["seed"]

    rng_seed = seed
    ds = dataset.shuffle(seed=rng_seed)

    if len(ds) >= eval_samples:
        train_ds, val_ds = ds["train"].train_test_split(
            test_size=eval_samples, seed=rng_seed
        )
        val_ds = val_ds.select(range(eval_samples))
    else:
        train_ds = ds["train"]
        val_ds = ds["train"]

    if len(train_ds) > train_samples:
        train_ds = train_ds.select(range(train_samples))

    eos_token = tokenizer.eos_token

    def format_example(example):
        article = example["article"].strip()
        summary = example["summary"].strip()
        truncated_article = _truncate_article(
            article, summary, INSTRUCTION, eos_token, tokenizer, max_seq_length
        )
        return {"text": _build_training_text(truncated_article, summary, eos_token)}

    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names, desc="format train")
    val_ds = val_ds.map(format_example, remove_columns=val_ds.column_names, desc="format val")

    return train_ds, val_ds
