# IndoNewsSum — Indonesian News Summarizer

Fine-tunes **Qwen3.8-27B** with **QLoRA (Unsloth)** on the Indonesian subset of the
[XL-Sum](https://arxiv.org/abs/2107.07596) dataset to produce a zero-shot news
summarizer for Bahasa Indonesia.

- **Base model**: `unsloth/Qwen3.8-27B-bnb-4bit` (4-bit quantized, ~16 GB VRAM)
- **Training target**: 5 000 Indonesian XL-Sum train samples, 1 epoch, < 8 h on 2× T4
- **Deploy**: LoRA adapter + merged 16-bit model on Hugging Face Hub, GGUF for llama.cpp
- **Demo**: Gradio Space — `space/app.py`

---

## Architecture

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                        Training Pipeline                        │
 │                                                                 │
 │  XL-Sum (id) ──► data.py ──► SFTTrainer (TRL) ──► QLoRA       │
 │   (5K samples)    (format   (text field,     r=16, α=16,      │
 │                    +trunc)   packing=False)  dropout=0         │
 │                                                                  │
 │  Qwen3.8-27B-bnb-4bit                                           │
 │        │                                                         │
 │        ├──► q_proj, k_proj, v_proj, o_proj                     │
 │        ├──► gate_proj, up_proj, down_proj                       │
 │        │                                                         │
 │        └──► outputs/adapter  ──► HF Hub                         │
 │                          └──► merged_16bit  ──► HF Hub          │
 │                          └──► GGUF q4_k_m  (optional)          │
 └──────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────┐
 │                         Inference (Space)                       │
 │                                                                 │
 │  Article ──► Prompt template ──► 4-bit base + LoRA adapter     │
 │                              (greedy, max_new_tokens=128)       │
 │                              ──► ROUGE-1/2/L eval              │
 └──────────────────────────────────────────────────────────────────┘
```

---

## Repo Layout

```
indo-news-summarizer/
├── README.md
├── requirements.txt
├── configs/
│   └── train.yaml          # all hyper-parameters, dataset, LoRA settings
├── src/
│   ├── __init__.py
│   ├── data.py             # XL-Sum loading, formatting, truncation
│   ├── model.py            # Unsloth model + PEFT init
│   ├── train.py            # SFTTrainer orchestration
│   ├── evaluate.py         # ROUGE-1/2/L evaluation
│   └── export.py           # merged 16-bit + optional GGUF export
├── notebooks/
│   └── kaggle_run.ipynb    # Kaggle notebook orchestrator
└── space/
    ├── app.py              # Gradio demo
    └── requirements.txt
```

---

## Quickstart — Local

```bash
# Install
pip install -r requirements.txt

# Export HF token (required for pushing adapters)
export HF_TOKEN="hf_XXXXXXXXXXXX"

# Train
python -m src.train --config configs/train.yaml

# Evaluate
python -m src.evaluate --config configs/train.yaml --adapter outputs/adapter --n 50

# Export merged + GGUF
python -m src.export --config configs/train.yaml --adapter outputs/adapter --out outputs/merged --gguf
```

---

## Kaggle Step-by-Step

| # | Step | Where |
|---|------|-------|
| 1 | Create notebook, set **Accelerator = GPU T4 × 2**, **Internet = ON** | Kaggle UI |
| 2 | **Add-ons → Secrets** → add key `HF_TOKEN` = your HF access token | Kaggle UI |
| 3 | Open `notebooks/kaggle_run.ipynb` | Kaggle |
| 4 | Run cell **1** — clones this repo to `/kaggle/working/indo-news-summarizer` | Notebook |
| 5 | Run cell **2** — Kaggle-safe `--no-deps` install (no wheel re-download) | Notebook |
| 6 | Run cell **3** — injects `HF_TOKEN` from Kaggle Secrets into env | Notebook |
| 7 | Run cell **4** — `python src/train.py` (~ 6 h on 2× T4) | Notebook |
| 8 | Run cell **5** — `python src/evaluate.py` (ROUGE on 50 val samples) | Notebook |
| 9 | Run cell **6** — `python src/export.py --gguf` (merged 16-bit + GGUF) | Notebook |
| 10 | Deploy `space/` to a Gradio Space pointing at the HF repo | HF Spaces |

> **Session budget**: 12 h max per session, 30 h/week quota. Training finishes in
> ~6 h; evaluation and export add ~30 min. Total: well within the 12 h window.

---

## Config Reference

All values live in `configs/train.yaml`.

| Key | Default | Description |
|-----|---------|-------------|
| `model_name` | `unsloth/Qwen3.8-27B-bnb-4bit` | Pre-quantized 4-bit base model |
| `dataset_name` | `csebuetnlp/xlsum` | Hugging Face dataset ID |
| `dataset_config` | `indonesian` | Language config within the dataset |
| `max_seq_length` | `4096` | Context window (tokens) |
| `train_samples` | `5000` | Train subset size |
| `eval_samples` | `200` | Val subset size |
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `16` | LoRA scaling factor |
| `lora_dropout` | `0.0` | LoRA dropout |
| `target_modules` | `q,k,v,o,gate,up,down` | Attention + MLP projections |
| `learning_rate` | `2.0e-4` | Peak LR |
| `num_train_epochs` | `1` | Epochs |
| `per_device_train_batch_size` | `1` | Batch per GPU |
| `gradient_accumulation_steps` | `4` | Effective batch = 8 |
| `warmup_ratio` | `0.03` | Warmup fraction of total steps |
| `optim` | `adamw_8bit` | 8-bit AdamW (bitsandbytes) |
| `weight_decay` | `0.01` | L2 decay |
| `lr_scheduler_type` | `linear` | Linear decay |
| `logging_steps` | `10` | Console log interval |
| `save_steps` | `250` | Checkpoint interval |
| `save_total_limit` | `2` | Max checkpoints on disk |
| `seed` | `3407` | Random seed |
| `hf_repo_id` | *set before running* | HF Hub repo to push to |
| `push_adapter` | `true` | Push LoRA weights to Hub |
| `push_merged` | `true` | Push merged 16-bit to Hub |
| `export_gguf` | `false` | Also export GGUF during export.py |

---

## Evaluation Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| Base Qwen3.8-27B (no adapter) | — | — | — |
| + QLoRA (this repo, 5K id-XL-Sum) | *run `evaluate.py` to fill* | *run* | *run* |

Run `python -m src.evaluate --config configs/train.yaml --n 50` and paste the
mean F1 scores in the table above.

---

## Gradio Space

The Space at `space/app.py` loads:

```
unsloth/Qwen3.8-27B-bnb-4bit   (base, 4-bit)
+
icaluwu/Indonesia-News-Summarizer-Qwen3.8-27b   (LoRA adapter, pushed by train.py)
```

Set the Space secret `HF_TOKEN` (and optionally `HF_REPO_ID` to override the
default) before deploying.

---

## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## Citations

```bibtex
@article{xlsum,
  title     = {XL-Sum: Large-Scale Multilingual Abstractive Summarization for 40 Languages},
  author    = {Fabbri et al.},
  year      = {2021},
  journal   = {Proceedings of EMNLP 2021},
}

@misc{qwen3.8,
  title  = {Qwen3.8-27B},
  author = {Qwen Team, Alibaba},
  year   = {2025},
  url    = {https://huggingface.co/Qwen},
}

@misc{unsloth,
  title  = {Unsloth: 2–5× Faster, 60\% Smaller LLMs},
  author = {Pulkit Quant and the Unsloth team},
  year   = {2024},
  url    = {https://github.com/unslothai/unsloth},
}
```
