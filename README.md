# EauRouge-F1 — Qwen3-8B Fine-tune on F1-Dataset

Fine-tuned `unsloth/Qwen3-8B-unsloth-bnb-4bit` (299k dl) on `VineshF1/F1-Dataset` 10656 QA.

## Model
🤗 **HF:** → https://huggingface.co/Vinesh/EauRouge-F1-Qwen3-8B-v2

Dataset: https://github.com/VineshF1/F1-Dataset `10656` pairs `5745 race_result · 2039 qual · 1748 driver_bio · 364 champion · 239 career_total · 177 circuit`


## Training
* Colab T4, Unsloth, QLoRA r32 alpha32, `max_seq 1024` packed, `batch 4x2` eff 8, `1.5 epoch` 10656 `9590/1066`, `lr 2e-4` cosine `warmup 0.03`
* Prompt: `tokenizer.apply_chat_template` + `temp 0.1 / top_p 0.9 / repetition 1.1 / max_new 150`


## Repo
* Code only — weights on HF.
* `train.ipynb` — Colab notebook (Unsloth)

## Author

**Vinesh**

Built with ❤️
