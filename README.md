# EauRouge-F1

LoRA fine-tune of `Qwen3-8B` on 10,656 Formula 1 question/answer pairs. It answers short
factual questions about race results, qualifying, circuits and championship totals.

**Adapter:** https://huggingface.co/VineshF1/EauRouge-F1-Qwen3-8B

## Results

100 held-out questions per set, greedy decoding.

| Test set | Model | Correct value | Exact |
|---|---|---|---|
| facts never seen in training | previous adapter | 36.8% | 6% |
| facts never seen in training | this adapter | 33.7% | 7% |
| facts seen, new wording | base model | 33% | 0% |
| facts seen, new wording | this adapter | **43%** | **11%** |

"Correct value" means the key fact appears somewhere in the answer; "exact" means the whole
answer matches. The adapter learns the answer format and recall on facts it trained on. It does
not learn facts that were never in the data - the 36.8 vs 33.7 difference is noise at n=100.

## Dataset

10,656 pairs from [F1-Dataset](https://github.com/VineshF1/F1-Dataset), split by source so no
fact lands on both sides: 9,608 train / 1,048 eval, zero shared answers (fingerprint
`dfe215c8d55cf477`). Training keeps at most 2 paraphrases per fact and adds 31 aggregate
questions generated from `champions.json`.

## Training

Unsloth QLoRA, r=32, 4-bit NF4, 1024 tokens, packed, batch 4x2, lr 2e-4 cosine, 3 epochs
(216 steps) on a free Kaggle T4. Loss on the answer tokens only. Eval loss flattened at 0.70
by step 100, so 2 epochs would give the same result.

## Files

- `Train.ipynb` - Kaggle notebook: trains, evaluates, pushes the adapter to the Hub
- `eval_adapter.py` - evaluation used for every number above

## Run the eval

```bash
curl -L -o qa.jsonl https://raw.githubusercontent.com/VineshF1/F1-Dataset/main/data/final/qa.jsonl
python eval_adapter.py --model VineshF1/EauRouge-F1-Qwen3-8B \
  --base unsloth/Qwen3-8B-unsloth-bnb-4bit --qa qa.jsonl --n 100
```

Needs a GPU. `--dry-run` checks the split and the scoring without loading the model.

## Limits

Trained on one-line answers, so open-ended questions get terse replies. Count questions are
unreliable. F1 only, English only, data frozen at the 2025 season.

## Author

**Vinesh**
