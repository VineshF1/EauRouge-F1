#!/usr/bin/env python3
"""
eval_adapter.py - held-out evaluation for EauRouge-F1 (Qwen3-8B + LoRA).

Measures exact match and "contains the correct value" on N held-out questions
whose gold answer never appears in the training answers.

The split is computed from data/final/qa.jsonl alone (seed 42), so the same
100 questions are used for every adapter. A fingerprint of the test questions
is printed and written into the CSV so you can prove before/after used the
same rows.

Kaggle (before, published adapter):
  !wget -q https://raw.githubusercontent.com/VineshF1/F1-Dataset/main/data/final/qa.jsonl -O /kaggle/working/qa.jsonl
  !python eval_adapter.py --model VineshF1/EauRouge-F1-Qwen3-8B \
      --qa /kaggle/working/qa.jsonl --out /kaggle/working/eval_before.csv

Kaggle (after, local adapter folder):
  !python eval_adapter.py --model /kaggle/working/qwen3-8b-lora \
      --qa /kaggle/working/qa.jsonl --out /kaggle/working/eval_after.csv

Local data-only self check (no GPU, no model):
  python eval_adapter.py --dry-run --qa data/final/qa.jsonl
"""

import argparse
import collections
import csv
import hashlib
import json
import random
import re
import sys
import unicodedata

SPLIT_SEED = 42
TEST_FRACTION = 0.10
CAP_PER_FACT = 2


# ---------------------------------------------------------------- data / split
def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_key(row):
    """Unit that must stay on one side of the split.

    Every answer in qa.jsonl belongs to exactly one `source` (checked: 0 answers
    span two sources), so source == fact. The one synthetic source that holds
    every career_total row is split by answer instead, otherwise all 239 rows
    would land in a single side.
    """
    if row["entity_type"] == "career_total":
        return "fact::" + row["answer"]
    return "src::" + row["source"]


def grouped_split(rows, test_fraction=TEST_FRACTION, seed=SPLIT_SEED):
    """Return (train_idx, test_idx) with no shared source and no shared answer."""
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        groups[group_key(row)].append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_test = max(1, round(test_fraction * len(keys)))
    test_keys = set(keys[:n_test])
    train_idx, test_idx = [], []
    for key, idxs in groups.items():
        (test_idx if key in test_keys else train_idx).extend(idxs)
    return train_idx, test_idx


def cap_paraphrases(rows, max_per_fact=CAP_PER_FACT):
    """Keep at most `max_per_fact` rows per unique answer (deterministic)."""
    buckets = collections.defaultdict(list)
    for i, row in enumerate(rows):
        buckets[row["answer"]].append(i)
    keep = []
    for answer, idxs in buckets.items():
        if len(idxs) <= max_per_fact:
            keep.extend(idxs)
        else:
            rng = random.Random(int(hashlib.md5(answer.encode("utf-8")).hexdigest()[:8], 16))
            keep.extend(rng.sample(idxs, max_per_fact))
    return sorted(keep)


def split_fingerprint(questions):
    blob = "\n".join(sorted(questions)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ------------------------------------------------------------------------ text
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def _number_words_to_digits(text):
    for i, word in enumerate(_ONES):
        text = re.sub(rf"\b{word}\b", str(i), text)
    for i, word in enumerate(_TENS):
        text = re.sub(rf"\b{word}\b", str((i + 2) * 10), text)
    return text


def normalize(text):
    """Lowercase, fold accents, map number words to digits, keep [a-z0-9 ]."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = _number_words_to_digits(text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_think(text):
    """Remove <think>...</think>, including an unclosed trailing block."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<think>.*$", "", text, flags=re.S | re.I)
    return text.strip()


def key_values(row):
    """Strings that must all appear in a prediction to call it correct.

    Empty list = this type has no single checkable value (long prose answers);
    those rows are excluded from the contains-value score and counted separately.
    """
    answer = row["answer"]
    kind = row["entity_type"]
    question = row["question"]

    if kind in ("race_result", "qualifying"):
        return [answer.split(" (")[0]]
    if kind == "season_champion":
        return [answer.split(" won ")[0]]
    if kind == "career_total":
        m = re.match(r"(.+?) won (\d+) World Drivers' Championship", answer)
        if m:
            return [m.group(1), f"{m.group(2)} World Drivers Championship"]
        return []
    if kind == "h2h":
        m = re.match(r"In \d{4}, (.+?) finished ahead", answer)
        return [m.group(1)] if m else []
    if kind == "circuit":
        return [answer.split(" is ")[0]]
    if kind == "driver_bio":
        name = answer.split(" (")[0]
        m = re.search(r"Tell me about (.+?)'s F1 career", question)
        aliases = [name] + ([m.group(1)] if m else [])
        return aliases[:1]  # answer name is the canonical one
    if kind == "person":
        return [answer.split(" - ")[0].split(" is ")[0]]
    if kind == "incident":
        m = re.match(r"What happened to (.+?)\?", question)
        if m:
            return [m.group(1)]
        return [answer.split(" was ")[0]]
    return []


def score_row(prediction, row):
    gold = row["answer"]
    values = key_values(row)
    pred_n = normalize(strip_think(prediction))
    exact = int(pred_n == normalize(gold))
    if values:
        contains = int(all(normalize(v) in pred_n for v in values))
    else:
        contains = None
    return exact, contains, values


# ---------------------------------------------------------------------- prompts
def build_prompt(tokenizer, question):
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # matches the empty <think></think> the trainer sees
    )


# ------------------------------------------------------------------------ model
def load_model(model_name, base_model, max_seq_length):
    from unsloth import FastLanguageModel

    base = base_model or model_name
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base, max_seq_length=max_seq_length,
        dtype=None, load_in_4bit=True,
    )
    if base_model:  # --model is a raw adapter directory / repo
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_name)
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(model, tokenizer, questions, batch_size, max_new_tokens):
    import torch

    predictions = []
    with torch.inference_mode():
        for start in range(0, len(questions), batch_size):
            batch = questions[start:start + batch_size]
            prompts = [build_prompt(tokenizer, q) for q in batch]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=1024,
            ).to(model.device)
            out = model.generate(
                **enc,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.0,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            predictions.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
            print(f"  generated {min(start + batch_size, len(questions))}/{len(questions)}", flush=True)
    return predictions


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="HF repo or local dir (adapter or merged)")
    ap.add_argument("--base", default="", help="optional base model if --model is a raw PEFT adapter")
    ap.add_argument("--qa", default="data/final/qa.jsonl")
    ap.add_argument("--out", default="eval_results.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--dry-run", action="store_true",
                    help="score the gold answer against itself; no model, no GPU")
    args = ap.parse_args()

    rows = load_rows(args.qa)
    train_idx, test_idx = grouped_split(rows)
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]

    train_answers = {normalize(r["answer"]) for r in train_rows}
    pool = [r for r in test_rows if normalize(r["answer"]) not in train_answers]
    if not pool:
        sys.exit("no held-out rows with unseen answers")

    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))
    fp = split_fingerprint(r["question"] for r in pool)

    print(f"rows={len(rows)} train={len(train_rows)} test={len(test_rows)} "
          f"unseen_answer_pool={len(pool)}")
    print(f"sampled={len(sample)} split_fingerprint={fp}")
    print("sample mix:", dict(collections.Counter(r["entity_type"] for r in sample).most_common()))

    if args.dry_run:
        predictions = [r["answer"] for r in sample]
    else:
        if not args.model:
            sys.exit("--model is required unless --dry-run")
        model, tokenizer = load_model(args.model, args.base, args.max_seq_length)
        predictions = generate(
            model, tokenizer, [r["question"] for r in sample],
            args.batch_size, args.max_new_tokens,
        )

    records = []
    for row, pred in zip(sample, predictions):
        exact, contains, values = score_row(pred, row)
        records.append({
            "entity_type": row["entity_type"],
            "question": row["question"],
            "answer": row["answer"],
            "prediction": strip_think(pred),
            "exact_match": exact,
            "contains_value": "" if contains is None else contains,
            "key_value": " | ".join(values),
            "split_fingerprint": fp,
            "adapter": args.model or "dry-run",
        })

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    scored = [r for r in records if r["contains_value"] != ""]
    n_exact = sum(r["exact_match"] for r in records)
    n_contains = sum(r["contains_value"] for r in scored)
    print()
    print(f"exact_match      {n_exact}/{len(records)} = {n_exact / len(records):.1%}")
    print(f"contains_value   {n_contains}/{len(scored)} = {n_contains / len(scored):.1%} "
          f"({len(scored)} rows have a checkable value, {len(records) - len(scored)} prose rows excluded)")

    print("\nper type (exact / contains over rows with a value):")
    by_type = collections.defaultdict(list)
    for r in records:
        by_type[r["entity_type"]].append(r)
    for kind, group in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        val = [r for r in group if r["contains_value"] != ""]
        c = sum(r["contains_value"] for r in val)
        print(f"  {kind:16s} n={len(group):3d}  exact={sum(r['exact_match'] for r in group):3d}"
              f"  contains={c}/{len(val) if val else 0}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
