# Burmese Pragmatics NLP — Context-Sensitive Politeness Classification

**Group: AttentionIsAllUNeed** | NLP Final Project | Asian Institute of Technology

> A 4-stage ablation study on pragmatic politeness classification in Burmese using fine-tuned XLM-RoBERTa, with a fully end-to-end pipeline that automatically predicts register, power relation, and tone.

---

## Overview

Pragmatic meaning in Burmese is highly sensitive to social context. The same underlying intent can be expressed in dozens of different ways depending on the speaker-listener relationship, the formality of the situation, and the speaker's emotional stance. This project builds a system that classifies the **politeness level** of Burmese utterances into 6 categories:

`neutral` · `polite` · `informal` · `professional` · `blunt` · `rude`

We design a **4-stage ablation study** that systematically adds richer contextual information to the model input, and measure how much each addition improves classification quality.

---

## Results

| Stage | Input | Macro F1 | Δ |
|---|---|---|---|
| Stage 1 | Utterance only | 0.701 | — |
| Stage 2 | + context + instruction | 0.758 | +0.057 |
| Stage 3 | + oracle register/power/tone | 0.799 | +0.041 |
| Stage 4 | + predicted register/power/tone | 0.702 | −0.097 |

**Key finding:** Contextual signals meaningfully improve classification. However, when metadata is predicted automatically rather than provided as ground truth, error propagation nearly eliminates the benefit — Stage 4 ≈ Stage 1. This quantifies the bottleneck: intermediate classifier accuracy.

---

## Intermediate Classifiers

Three auxiliary classifiers predict pragmatic metadata for the Stage 4 pipeline:

| Classifier | Task | Input | Classes | Macro F1 |
|---|---|---|---|---|
| A | Register | Utterance | 4 | 0.705 |
| B | Power relation | Utterance + context | 4 | 0.598 |
| C | Tone | Utterance + context | 5 | 0.613 |

Tone labels are grounded in **Appraisal Theory** (Martin & White, 2005), mapping 131 informal annotations to 5 principled categories: `neutral` · `positive` · `formal` · `negative` · `emotional`.

---

## Models on HuggingFace

| Model | Repo |
|---|---|
| Classifier A (Register) | [annasus10/xlmr-burmese-register](https://huggingface.co/annasus10/xlmr-burmese-register) |
| Classifier B (Power) | [annasus10/xlmr-burmese-power](https://huggingface.co/annasus10/xlmr-burmese-power) |
| Classifier C (Tone) | [annasus10/xlmr-burmese-tone](https://huggingface.co/annasus10/xlmr-burmese-tone) |
| Stage 1 | [annasus10/xlmr-burmese-pragmatics-stage1-v2](https://huggingface.co/annasus10/xlmr-burmese-pragmatics-stage1-v2) |
| Stage 2 | [annasus10/xlmr-burmese-pragmatics-stage2-v2](https://huggingface.co/annasus10/xlmr-burmese-pragmatics-stage2-v2) |
| Stage 3 | [annasus10/xlmr-burmese-pragmatics-stage3-v2](https://huggingface.co/annasus10/xlmr-burmese-pragmatics-stage3-v2) |

---

## Live Demo

| Demo | Description |
|---|---|
| [Demo v1](https://huggingface.co/spaces/annasus10/burmese-pragmatics-demo) | Stages 1/2/3 with manual metadata input |
| [Demo v2](https://huggingface.co/spaces/annasus10/burmese-pragmatics-demo-v2) | Full Stage 4 end-to-end pipeline — metadata predicted automatically |

---

## Dataset

[freococo/burmese-contextual-pragmatics](https://huggingface.co/datasets/freococo/burmese-contextual-pragmatics) — 2,200 Burmese utterances · 22 root meanings · CC0 license

---

## Methodology

### Base Model
All 7 models fine-tune `xlm-roberta-base` (Conneau et al., ACL 2020) — an encoder-only transformer trained on 100 languages including Burmese.

### Training
- 8 epochs · batch size 8 · lr 2e-5 · warmup 150 steps
- Weighted cross-entropy loss (handles severe class imbalance — neutral = 53% of data)
- Best checkpoint selected by macro F1 on validation set
- Fixed seed=42, identical 70/15/15 split across all 7 models

### Stage 4 Pipeline
```
Utterance + Context + Instruction
        │
        ├──► Classifier A ──► register prediction
        ├──► Classifier B ──► power relation prediction  
        └──► Classifier C ──► tone prediction
                │
                ▼
[register: X] [power: Y] [tone: Z] utterance </s> context </s> instruction
                │
                ▼
        Stage 3 Politeness Classifier
                │
                ▼
        Predicted Politeness Label
```

---

## Direction Change

This project originally proposed **text generation** with mT5. We pivoted to **classification** with XLM-RoBERTa for three reasons:

1. The dataset has only 22 root meanings with ~100 near-identical variations — insufficient diversity for a generation model to generalise
2. BLEU/chrF are unreliable when many surface forms are all valid for the same intent
3. Early training confirmed non-convergence (loss ~27, one run produced `training_loss=0.0, val_loss=NaN` from a GPU FP16 bug)

Classification provides clean metrics, a falsifiable research claim, and a well-defined evaluation setup.

---

## Research Questions

**RQ1:** How much does adding social context improve Burmese politeness classification over an utterance-only baseline?

**RQ2:** What is the oracle upper bound when explicit pragmatic metadata is provided?

**RQ3:** How much of the oracle benefit is preserved in an end-to-end pipeline, and what is the error propagation cost?

---

## Preprocessing

- **Politeness:** 13 original labels → 6 classes (rare labels merged by semantic similarity)
- **Tone:** 131 free-text values → 5 Appraisal Theory categories
- **Power relation:** 8 values → 4 classes (role-specific inferior variants merged)
- **Class imbalance:** weighted cross-entropy loss (neutral weight: 0.318, rude weight: 6.111)

---

## Next Steps

- [ ] Zero-shot baseline with GPT-4 and Gemini (Stage 0)
- [ ] Improve intermediate classifiers (target: >0.80 macro F1)
- [ ] Error analysis on misclassified Burmese examples
- [ ] Final paper

---

## Team

| Name | Email |
|---|---|
| Thet Su Sann | st126316@ait.asia |
| Thiri Shin Thant | st126018@ait.asia |
| Hein Min Htet | st126459@ait.asia |
| Tisa Bajracharya | st126686@ait.asia |

---

## References

- Conneau et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL 2020*.
- Martin, J.R. & White, P.R.R. (2005). *The Language of Evaluation: Appraisal in English*. Palgrave Macmillan.
- freococo (2024). Burmese Contextual Pragmatics Dataset. HuggingFace.