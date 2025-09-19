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
