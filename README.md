# Controllable Pragmatics in Low-Resource Language Generation (Burmese with mT5)

**NLP Final Project — Group AttentionIsAllYouNeed**

---

## Project Overview

This project investigates controllable pragmatic text generation for a low-resource language (Burmese) using a multilingual transformer model (mT5).

While most text generation systems focus on semantic correctness, this work emphasizes pragmatic appropriateness—how language varies based on social and contextual factors such as politeness, tone, and power relations. As highlighted in the proposal, semantic correctness alone does not guarantee socially appropriate language use .

---

## Motivation

In Burmese, linguistic expression is highly sensitive to sociolinguistic context. The same semantic intent (e.g., requesting help) can be expressed differently depending on:

* Formal versus casual settings
* Speaker–listener relationships
* Levels of politeness

Existing NLP systems:

* Primarily focus on semantic meaning
* Often ignore sociolinguistic context
* Are largely developed for high-resource languages

This project addresses these limitations by modeling fine-grained pragmatics in a low-resource setting.

---

## Problem Statement

Current NLP approaches face the following limitations:

1. Overemphasis on semantic meaning while ignoring social context
2. Limited research on low-resource languages such as Burmese
3. Lack of controllability over sociolinguistic attributes



---

## Research Objectives

The objectives of this study are:

* To investigate controllable text generation in Burmese
* To analyze the impact of contextual and sociolinguistic attributes
* To develop a system capable of generating socially appropriate utterances

Key attributes considered:

* Context
* Politeness
* Tone
* Power relations

---

## Research Questions

* **RQ1:** How does fine-tuning a multilingual text-to-text model compare to a zero-shot baseline?
* **RQ2:** To what extent does incorporating contextual information improve generation quality?
* **RQ3:** Which sociolinguistic attributes contribute most to generation quality?



---

## Dataset

This project uses the Burmese Contextual Pragmatics Dataset (HuggingFace), which contains:

* Approximately 2,200 samples
* 22 conversational intents
* Four key attributes:

| Attribute      | Description                  |
| -------------- | ---------------------------- |
| Context        | Formal or casual setting     |
| Politeness     | High, medium, or low         |
| Tone           | Direct, neutral, or indirect |
| Power Relation | Speaker-listener hierarchy   |



---

## Methodology

### Input Representation

Structured input is defined as:

```
Meaning: ask for help  
Context: formal  
Politeness: high  
Tone: neutral  
Power: lower → higher  
```

This is converted into a textual prompt:

```
Generate Burmese sentence:
Meaning: ask for help
Context: formal
Politeness: high
Tone: neutral
Power: lower → higher
```

---

### Model

* Model: mT5 (multilingual text-to-text transformer)
* Approach: Fine-tuning using structured prompts and target Burmese sentences



---

### System Pipeline

1. Structured attribute input
2. Prompt construction
3. Model fine-tuning
4. Sentence generation
5. Evaluation

The full framework is illustrated in the proposal (Figure 1, page 3) .

---

## Experimental Design

We evaluate multiple model configurations:

| Model         | Input Features                     |
| ------------- | ---------------------------------- |
| Zero-shot mT5 | No fine-tuning                     |
| Meaning-only  | Semantic meaning only              |
| Context-aware | Meaning + context                  |
| Full model    | Meaning + all pragmatic attributes |

Expected progression:

Zero-shot → Meaning-only → Context-aware → Full model

---

## Evaluation

A dual evaluation strategy is used:

### 1. Text Similarity

* BLEU
* chrF

### 2. Pragmatic Consistency

* Attribute classification (politeness, tone, power)
* Comparison between predicted and target attributes



---

## Analysis Plan

* Attribute ablation to measure individual contribution
* Attribute-level evaluation
* Error analysis of incorrect generations
* Performance analysis on rare attribute combinations

---

## Expected Results

* The full model is expected to achieve the best performance
* Incorporating contextual and pragmatic attributes improves output quality
* Not all attributes contribute equally
* Text similarity metrics do not fully capture pragmatic correctness



---

## Limitations

* Small dataset (~2,200 samples)
* Potential bias in classifier-based evaluation
* Limited generalization beyond dataset scope
* Focus restricted to Burmese



---

## Current Progress

Completed:

* Dataset exploration and analysis
* Methodology design
* Prompt construction framework
* Experimental setup planning
* Proposal and diagrams

Next steps:

* Implement training pipeline
* Fine-tune mT5 model
* Develop evaluation classifier
* Conduct experiments



---

## Team

**Group: AttentionIsAllYouNeed**

* Thet Su Sann
* Thiri Shin Thant
* Hein Min Htet
* Tisa Bajracharya

---

## References

* Burmese Contextual Pragmatics Dataset (freococo, 2024)
* mT5: Multilingual Text-to-Text Transformer (Xue et al., 2021)
* BLEU and chrF evaluation metrics

---

## Key Contribution

This project contributes:

* A framework for controllable pragmatic generation
* A focus on low-resource language modeling
* A dual evaluation approach combining semantic and pragmatic metrics

---

## Future Work

* Expand dataset size
* Extend to additional languages
* Improve pragmatic evaluation methods
* Explore larger model variants (mT5-base, mT5-large)

---

