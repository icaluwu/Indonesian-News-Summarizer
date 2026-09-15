"""Gradio demo for IndoNewsSum — Indonesian News Summarizer (Qwen3.8-27B + QLoRA)."""

import os

import gradio as gr
import torch
from peft import PeftModel
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen3.8-27B-bnb-4bit"
HF_REPO = os.environ.get(
    "HF_REPO_ID",
    "icaluwu/Indonesia-News-Summarizer-Qwen3.8-27b",
)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

INSTRUCTION = (
    "### Instruksi:\n"
    "Ringkas artikel berita di atas menjadi 1-3 kalimat dalam Bahasa Indonesia.\n\n"
    "### Ringkasan:\n"
)

EXAMPLES = [
    {
        "Artikel": "Pemerintah Indonesia mengumumkan program subsidi energi terbarukan untuk rumah tangga berpenghasilan rendah. Program ini akan mulai berlaku pada Januari 2025 dan ditargetkan mencakup 2 juta rumah tangga di 34 provinsi. Anggaran sebesar Rp 12 triliun dialokasikan untuk mendukung transisi energi bersih.",
        "Ringkasan": "Pemerintah Indonesia meluncurkan program subsidi energi terbarukan untuk 2 juta rumah tangga berpenghasilan rendah mulai Januari 2025 dengan anggaran Rp 12 triliun.",
    },
    {
        "Artikel": "Timnas Sepak Bola Indonesia berhasil lolos ke babak kualifikasi Piala Asia 2027. Dalam pertandingan terakhir, skuad asuhan pelatih baru berhasil mencetak tiga gol tanpa balas melawan Thailand di Stadion Gelora Bung Karno. Kemenangan ini menjadi yang pertama di kandang dalam tiga tahun terakhir.",
        "Ringkasan": "Timnas Indonesia lolos ke kualifikasi Piala Asia 2027 setelah mengalahkan Thailand 3-0 di Gelora Bung Karno, kemenangan pertama di kandang dalam tiga tahun.",
    },
]


def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.for_inference(model)
    model = PeftModel.from_pretrained(model, HF_REPO, token=HF_TOKEN or None)
    return model, tokenizer


model, tokenizer = load_model()


def summarize(article: str) -> str:
    if not article.strip():
        return "Silakan masukkan artikel berita terlebih dahulu."
    prompt = f"### Artikel:\n{article.strip()}\n\n{INSTRUCTION}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


theme = gr.themes.Soft(primary_color="#1a4f8f")

demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(
        label="Artikel Berita",
        placeholder="Tempelkan artikel berita berbahasa Indonesia di sini ...",
        lines=8,
    ),
    outputs=gr.Textbox(label="Ringkasan", lines=3),
    title="IndoNewsSum — Peringkas Berita Indonesia (Qwen3.8-27B)",
    examples=[
        gr.Examples(
            examples=[["Pemerintah Indonesia mengumumkan program subsidi energi terbarukan untuk rumah tangga berpenghasilan rendah. Program ini akan mulai berlaku pada Januari 2025 dan ditargetkan mencakup 2 juta rumah tangga di 34 provinsi. Anggaran sebesar Rp 12 triliun dialokasikan untuk mendukung transisi energi bersih."],
                        ["Timnas Sepak Bola Indonesia berhasil lolos ke babak kualifikasi Piala Asia 2027. Dalam pertandingan terakhir, skuad asuhan pelatih baru berhasil mencetak tiga gol tanpa balas melawan Thailand di Stadion Gelora Bung Karno."],
                        ["Bank Indonesia memutuskan untuk menahan suku bunga acuan di level 6,0 persen pada pertemuan terakhir. Keputusan ini diambil setelah inflasi nasional tercatat 2,8 persen, masih dalam rentang target. BI menyatakan akan terus memantau perkembangan ekonomi makro."]],
            inputs=["Artikel Berita"],
            outputs=["Ringkasan"],
            labels=["Input", "Output"],
        )
    ],
    theme=theme,
)

demo.queue(max_size=8)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
