# EauRouge-F1 — Qwen3-8B Fine-tune on F1-Dataset

Fine-tuned `unsloth/Qwen3-8B-unsloth-bnb-4bit` (299k dl) on `VineshF1/F1-Dataset` 10656 QA — fixes `7 vs 5 vs 4` hallucination.

## Model
🤗 **HF:** `Vinesh/EauRouge-F1-Qwen3-8B-v2` *(push after test — link will be live)* → https://huggingface.co/Vinesh/EauRouge-F1-Qwen3-8B-v2

Dataset: https://github.com/VineshF1/F1-Dataset `10656` pairs `5745 race_result · 2039 qual · 1748 driver_bio · 364 champion · 239 career_total · 177 circuit`

Base: `unsloth/Qwen3-8B-unsloth-bnb-4bit` 4.8% hallucination (Vectara) — T4 16GB

## Training
* Colab T4, Unsloth, QLoRA r32 alpha32, `max_seq 1024` packed, `batch 4x2` eff 8, `1.5 epoch` 10656 `9590/1066`, `lr 2e-4` cosine `warmup 0.03`
* Prompt: `tokenizer.apply_chat_template` + `temp 0.1 / top_p 0.9 / repetition 1.1 / max_new 150`

## Test
```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained("Vinesh/EauRouge-F1-Qwen3-8B-v2", load_in_4bit=True)
FastLanguageModel.for_inference(model)
def ask(q):
  p=tok.apply_chat_template([{"role":"user","content":q}], tokenize=False, add_generation_prompt=True)
  o=model.generate(**tok(p, return_tensors="pt").to("cuda"), max_new_tokens=150, temperature=0.1, pad_token_id=tok.eos_token_id)
  print(tok.decode(o[0], skip_special_tokens=True).split("assistant")[-1].strip())
ask("who is 7 time world champion drivers?") # Schumacher + Hamilton
ask("Who is the only 5-time World Champion?") # Fangio 1951,54,55,56,57
ask("Is verstappen 4 time champion") # 2021-24
```

## Repo
* Code only — weights on HF, not GitHub (see `.gitignore`)
* `train.ipynb` — Colab notebook (Unsloth)
* License: Code MIT, Data CC BY-SA (Jolpica/Wikipedia)

## Author
Vinesh — F1-Dataset 1950-2025
