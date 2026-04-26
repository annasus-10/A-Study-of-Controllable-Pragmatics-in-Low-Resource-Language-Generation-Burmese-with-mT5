# Burmese Pragmatics NLP — Context-Sensitive Politeness Classification

**Group: AttentionIsAllYouNeed** | NLP Final Project | Asian Institute of Technology

> A 4-stage ablation study on pragmatic politeness classification in Burmese using fine-tuned XLM-RoBERTa, with a fully end-to-end pipeline that automatically predicts register, power relation, and tone. Extended with improved intermediate classifier architectures and zero-shot LLM evaluation.

---

## Overview

Pragmatic meaning in Burmese is highly sensitive to social context. The same underlying intent can be expressed in dozens of different ways depending on the speaker-listener relationship, the formality of the situation, and the speaker's emotional stance. This project builds a system that classifies the **politeness level** of Burmese utterances into 6 categories:

`neutral` · `polite` · `informal` · `professional` · `blunt` · `rude`

We design a **4-stage ablation study** that systematically adds richer contextual information to the model input, and measure how much each addition improves classification quality. We further investigate two improved intermediate classifier architectures and compare against zero-shot LLMs.

---

## Results

### Politeness Classification — Full Ablation

| Model | Input | Accuracy | Macro F1 | Δ |
|---|---|---|---|---|
| Stage 1 | Utterance only | 0.739 | 0.701 | — |
| Stage 2 | + context + instruction | 0.794 | 0.758 | +0.057 |
| Stage 3 | + oracle register/power/tone | 0.846 | **0.799** | +0.041 |
| Stage 4 | + predicted metadata (original) | 0.727 | 0.702 | −0.097 |
| Stage 4 v2 | + predicted metadata (codependent chain) | 0.724 | 0.696 | — |
| Stage 4 MT | + predicted metadata (multi-task) | 0.773 | 0.681 | — |

### Zero-Shot LLM Baselines (oracle metadata provided)

| Model | Accuracy | Macro F1 |
|---|---|---|
| ChatGPT o4-mini | 0.564 | 0.635 |
| Gemini 3 Fast | 0.500 | 0.493 |

**Key findings:**
- Contextual signals meaningfully improve classification (+0.057 Stage 1→2)
- Oracle metadata provides a clear upper bound (+0.041 Stage 2→3)
- Error propagation costs −0.097, with Stage 4 ≈ Stage 1 despite having metadata
- **Improving intermediate classifiers does NOT improve end-to-end performance** — multi-task achieves the best intermediate F1 but the worst Stage 4 result (0.681). The bottleneck is the oracle/predicted distribution mismatch, not classifier accuracy
- Fine-tuned Stage 1 (utterance only, no metadata) outperforms ChatGPT o4-mini with full oracle metadata by +0.066 macro F1

---

## Intermediate Classifiers

Three auxiliary classifiers predict pragmatic metadata for the Stage 4 pipeline. We trained three variants:

### Original (Independent)

| Classifier | Task | Input | Classes | Macro F1 |
|---|---|---|---|---|
| A | Register | Utterance | 4 | 0.705 |
| B | Power relation | Utterance + context | 4 | 0.598 |
| C | Tone | Utterance + context | 5 | 0.613 |

### Codependent Chain (v2)

B_v2 receives a `[register: X]` prefix from A's prediction; C_v2 receives `[register: X] [power: Y]` prefixes. Teacher forcing at train time, chained predictions at inference.

| Classifier | Task | Macro F1 | Δ |
|---|---|---|---|
| B_v2 | Power relation | 0.637 | +0.039 |
| C_v2 | Tone | 0.621 | +0.008 |

### Multi-Task (Shared Encoder)

Single XLM-R backbone with three independent classification heads trained jointly.

| Head | Task | Macro F1 | Δ |
|---|---|---|---|
| MT-A | Register | 0.719 | +0.014 |
| MT-B | Power relation | 0.660 | +0.062 |
| MT-C | Tone | 0.632 | +0.019 |

Tone labels are grounded in **Appraisal Theory** (Martin & White, 2005), mapping 131 informal annotations to 5 principled categories: `neutral` · `positive` · `formal` · `negative` · `emotional`.

---

## Error Analysis

### Stage 3 vs Stage 4 Disagreements (49 cases)

| Metadata dimension | Raw error rate | Present in failures |
|---|---|---|
| Register | 26.1% | 63.3% |
| Tone | 33.0% | 61.2% |
| Power | 26.1% | 46.9% |

**Dominant failure pattern:** `neutral → polite` (21/49 cases), caused by register misclassification `colloquial → standard`. When the classifier upgrades the register, the politeness model infers a more formal context and predicts polite instead of neutral.

**Power relation errors** (`equal → inferior_to_superior`) also systematically inflate polite predictions for contextually neutral utterances.

### Zero-Shot LLM Failure Patterns

Both LLMs massively under-predict `neutral` (ChatGPT: 50 predictions vs 174 true; Gemini: 103 vs 174) and over-predict `informal` (+72 ChatGPT, +61 Gemini). The dominant confusion is `neutral → informal` in both models. This reflects the inability of general LLMs to distinguish pragmatic neutrality from casual register in Burmese without domain-specific fine-tuning.

---

## Models on HuggingFace

| Model | Repo |
|---|---|
| Classifier A (Register) | [annasus10/xlmr-burmese-register](https://huggingface.co/annasus10/xlmr-burmese-register) |
| Classifier B (Power) | [annasus10/xlmr-burmese-power](https://huggingface.co/annasus10/xlmr-burmese-power) |
| Classifier C (Tone) | [annasus10/xlmr-burmese-tone](https://huggingface.co/annasus10/xlmr-burmese-tone) |
| Classifier B_v2 (Power, codependent) | [annasus10/xlmr-burmese-power-v2](https://huggingface.co/annasus10/xlmr-burmese-power-v2) |
| Classifier C_v2 (Tone, codependent) | [annasus10/xlmr-burmese-tone-v2](https://huggingface.co/annasus10/xlmr-burmese-tone-v2) |
| Stage 1 | [annasus10/xlmr-burmese-pragmatics-stage1-v2](https://huggingface.co/annasus10/xlmr-burmese-pragmatics-stage1-v2) |
| Stage 2 | [annasus10/xlmr-burmese-pragmatics-stage2-v2](https://huggingface.co/annasus10/xlmr-burmese-pragmatics-stage2-v2) |
| Stage 3 (Best) | [annasus10/xlmr-burmese-pragmatics-stage3-v2](https://huggingface.co/annasus10/xlmr-burmese-pragmatics-stage3-v2) |

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
All models fine-tune `xlm-roberta-base` (Conneau et al., ACL 2020) — an encoder-only transformer trained on 100 languages including Burmese.

### Training
- 8 epochs · batch size 8 · lr 2e-5 · warmup 150 steps
- Weighted cross-entropy loss (handles severe class imbalance — neutral = 53% of data)
- Best checkpoint selected by macro F1 on validation set
- Fixed seed=42, identical 70/15/15 split across all models

### Stage 4 Pipeline (Original)
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

### Stage 4 v2 — Codependent Chain
```
Utterance ──► A ──► pred_register
                        │
Utterance + Context + [register: pred_register] ──► B_v2 ──► pred_power
                        │
Utterance + Context + [register: pred_register] [power: pred_power] ──► C_v2 ──► pred_tone
                        │
                        ▼
        Stage 3 Politeness Classifier
```

### Stage 4 MT — Multi-Task
```
Utterance + Context
        │
        ▼
  XLM-R Encoder (shared)
        │
   ┌────┴────┬─────────┐
   ▼         ▼         ▼
Head A     Head B    Head C
register   power     tone
        │
        ▼
Stage 3 Politeness Classifier
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

**RQ1:** How much does adding social context improve Burmese politeness classification over an utterance-only baseline? → **+0.057 macro F1**

**RQ2:** What is the oracle upper bound when explicit pragmatic metadata is provided? → **0.799 macro F1 (Stage 3)**

**RQ3:** How much of the oracle benefit is preserved in an end-to-end pipeline? → **−0.097 drop; Stage 4 ≈ Stage 1**

**RQ4:** Can improved intermediate classifiers close the oracle-pipeline gap? → **No. Better intermediate classifiers produce different error patterns incompatible with the Stage 3 politeness model's training distribution.**

---

## Preprocessing

- **Politeness:** 13 original labels → 6 classes (rare labels merged by semantic similarity)
- **Tone:** 131 free-text values → 5 Appraisal Theory categories
- **Power relation:** 8 values → 4 classes (role-specific inferior variants merged)
- **Class imbalance:** weighted cross-entropy loss (neutral weight: 0.318, rude weight: 6.111)

---

## Team

| Name | ID | Email |
|---|---|---|
| Thet Su Sann | ST126316 | st126316@ait.asia |
| Thiri Shin Thant | ST126018 | st126018@ait.asia |
| Hein Min Htet | ST126459 | st126459@ait.asia |
| Tisa Bajracharya | ST126686 | st126686@ait.asia |

---

## References

- Conneau et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL 2020*.
- Martin, J.R. & White, P.R.R. (2005). *The Language of Evaluation: Appraisal in English*. Palgrave Macmillan.
- freococo (2024). Burmese Contextual Pragmatics Dataset. HuggingFace.
- Finkel et al. (2006). Solving the Problem of Cascading Errors. *EMNLP 2006*.
- Plank (2022). The "Problem" of Human Label Variation. *EMNLP 2022*.