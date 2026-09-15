"""Model loading and LoRA initialization for Qwen3.8-27B via Unsloth."""

import unsloth
from unsloth import FastLanguageModel


def load_model(config):
    """Load the 4-bit base model and attach a QLoRA adapter.

    Parameters
    ----------
    config : dict
        Parsed YAML training config.

    Returns
    -------
    (model, tokenizer)
        The PEFT-wrapped model and its tokenizer.
    """
    model_name = config["model_name"]
    max_seq_length = config["max_seq_length"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    lora_r = config.get("lora_r", 16)
    lora_alpha = config.get("lora_alpha", 16)
    lora_dropout = config.get("lora_dropout", 0.0)
    target_modules = config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    seed = config.get("seed", 3407)

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    return model, tokenizer
