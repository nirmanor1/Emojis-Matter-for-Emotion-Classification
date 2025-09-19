# Do Emojis Matter? Multi-label Emotion Classification with and without Emojis

> University NLP Seminar Project · Reproducible code & results  

## TL;DR
Many NLP pipelines strip emojis during preprocessing. This repo reproduces and extends a controlled study comparing **with-emoji vs. no-emoji** text across **three model families**:
- **Classical:** TF-IDF + One-vs-Rest Logistic Regression  
- **Small/Finetuned:** `cardiffnlp/twitter-roberta-base` with sigmoid multi-label head  
- **Large/Instruction-following:** GPT-5 (zero-/few-shot prompts)

**Result:** Keeping emojis consistently improves multi-label emotion classification. The fine-tuned RoBERTa (with emojis) achieves the strongest overall performance and even outperforms the LLM configuration in this setup.

> Paper: “Do Emojis Matter? Exploring the Role of Emojis in Sentiment Classification with Small and Large Language Models.”
> Key details (datasets, metrics, prompts, and ablation) are implemented here.

---

## Why this matters
Emojis are not noise: they act as **discriminative features** that can **boost Jaccard, Micro-F1, and reduce Hamming Loss** across models. Treating emojis as first-class tokens allows compact models to **compete with** or **beat** much larger systems on emotion tasks.

---

## Datasets
We construct an **English-only, emoji-bearing** corpus by merging:
- EmoEvent (EN subset)
- GoEmotions (Reddit)
- SemEval-2018 Task 1 E-c (EN)
- TweetEval (emotion track; `optimism` dropped for label harmony)

Unified **7-label** schema: `{ anger, disgust, fear, joy, sadness, surprise, other }`.

> We **do not redistribute** datasets. Use `scripts/build_corpus.sh` to fetch/process from official sources and create:
- `data/processed/with_emoji/`  
- `data/processed/no_emoji/` (Unicode emojis removed; ASCII emoticons preserved)

---

## Key Results (test set)
(Replicating reported headline numbers from the paper)

| Model                       | Emojis | Jaccard | F1_micro | EMR  | Hamming ↓ |
|----------------------------|:------:|:-------:|:--------:|:----:|:---------:|
| TF-IDF + Logistic Regression |  Yes  | **0.3677** | **0.5994** | 0.3125 | **0.1611** |
| TF-IDF + Logistic Regression |   No  | 0.3297 | 0.5385 | 0.2969 | 0.1741 |
| RoBERTa (10 epochs)        |  Yes  | **0.6461** | **0.6629** | **0.5651** | **0.1105** |
| RoBERTa (10 epochs)        |   No  | 0.6183 | 0.6318 | 0.5443 | 0.1205 |
| GPT-5 (Zero-shot)          |  Yes  | **0.5940** | **0.6172** | 0.4792 | 0.1343 |
| GPT-5 (Zero-shot)          |   No  | 0.5686 | 0.5890 | 0.4688 | 0.1417 |

**Takeaways**
- Emojis improve all aggregate metrics for the classical baseline.  
- RoBERTa + emojis yields the strongest overall results.  
- GPT-5 benefits modestly from emojis; zero-shot ≈ few-shot.

## Code overview

### `few shot code.py`
LLM-based **zero-/few-shot** multi-label classifier that batches inputs and calls an Azure OpenAI chat model. It:
- Auto-detects whether texts contain emojis to pick the right prompt variant.
- Validates the model’s JSON output with Pydantic (`labels` restricted to the 7-class set).
- Batches (`BATCH_SIZE=20`) with retry + split-on-failure logic, then writes `predictions.csv` (`id,labels,num_labels`).
- **CLI:** `python "few shot code.py" data/test.csv predictions.csv`  (expects columns: `id,text`). :contentReference[oaicite:0]{index=0}

### `fine_tune_emoji_roberta.py`
Fine-tunes `cardiffnlp/twitter-roberta-base` for **multi-label** emotion classification on the **with-emojis** split.
- Config (defaults): `max_length=256`, `lr=1e-5`, `batch_size=16`, `epochs=10`, `warmup=0.1`, `weight_decay=0.01`, seed=42.
- Device-aware tweaks (CUDA/MPS/CPU), Kaggle/local path handling, HF `Trainer` with `problem_type="multi_label_classification"`.
- Metrics: **Jaccard**, **F1 (macro/micro/sample)**, **Hamming loss**, plus per-class precision/recall/F1; ensures at least one label (fallback to `other`).
- Outputs: saved model + tokenizer under `results/emoji_roberta_emotion_local/`, rich plots in `training_plots/`, 
  `multilabel_evaluation_results.(txt|json)`, and a detailed test CSV including `tweet_id`, `emojis`, and per-class binary columns.  
- **Run:** `python fine_tune_emoji_roberta.py` (expects data under `FinalData/split with emoji/`). :contentReference[oaicite:1]{index=1}

### `fine_tune_no_emoji_roberta.py`
Same as `fine_tune_emoji_roberta.py`.

