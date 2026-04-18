import gradio as gr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_REPOS = {
    "Stage 1 — Utterance only":              "annasus10/xlmr-burmese-pragmatics-stage1",
    "Stage 2 — Utterance + context":         "annasus10/xlmr-burmese-pragmatics-stage2",
    "Stage 3 — Full pragmatic input (best)": "annasus10/xlmr-burmese-pragmatics-stage3",
}

LABELS = ["blunt", "informal", "neutral", "polite", "professional", "rude"]
MAX_LENGTH = 128
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# Cache models so we don't reload every prediction
model_cache = {}

def load_model(stage):
    if stage not in model_cache:
        repo = MODEL_REPOS[stage]
        model_cache[stage] = AutoModelForSequenceClassification.from_pretrained(repo)
        model_cache[stage].eval()
    return model_cache[stage]

# ── Inference ─────────────────────────────────────────────────────────────────

def predict(stage, utterance, context, instruction, register, power, tone):
    if not utterance.strip():
        return {}, "Please enter a Burmese utterance."

    # Build input text based on stage
    if stage == "Stage 1 — Utterance only":
        text = utterance.strip()

    elif stage == "Stage 2 — Utterance + context":
        text = (
            utterance.strip()
            + " </s> "
            + context.strip()
            + " </s> "
            + instruction.strip()
        )

    else:  # Stage 3
        text = (
            f"[register: {register}] "
            f"[power: {power}] "
            f"[tone: {tone}] "
            + utterance.strip()
            + " </s> "
            + context.strip()
            + " </s> "
            + instruction.strip()
        )

    # Tokenize
    inputs = tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    # Predict
    model = load_model(stage)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze().numpy()

    predicted_class = LABELS[np.argmax(probs)]
    confidence = float(np.max(probs))
    scores = {label: float(prob) for label, prob in zip(LABELS, probs)}

    summary = f"Predicted: **{predicted_class}** ({confidence*100:.1f}% confidence)"
    return scores, summary

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Burmese Pragmatics Classifier") as demo:

    gr.Markdown("""
    # Burmese Politeness Classifier
    **Group: AttentionIsAllUNeed** — NLP Final Project

    This demo classifies the **politeness level** of a Burmese utterance using a fine-tuned
    XLM-RoBERTa model. Choose a stage to see how adding contextual signals improves prediction.

    | Stage | Input | Test Macro F1 |
    |---|---|---|
    | Stage 1 | Utterance only | 0.545 |
    | Stage 2 | + context + instruction | 0.706 |
    | Stage 3 | + register + power + tone | 0.825 |
    """)

    with gr.Row():
        stage = gr.Dropdown(
            choices=list(MODEL_REPOS.keys()),
            value="Stage 3 — Full pragmatic input (best)",
            label="Model stage"
        )

    gr.Markdown("### Input")

    utterance = gr.Textbox(
        label="Burmese utterance (required)",
        placeholder="မင်း တကယ် လှတယ်။",
        lines=2
    )

    with gr.Group(visible=True) as context_group:
        context = gr.Textbox(
            label="Context",
            placeholder="e.g. General social interaction.",
            lines=1
        )
        instruction = gr.Textbox(
            label="Instruction",
            placeholder="e.g. Simple standard compliment.",
            lines=1
        )

    with gr.Group(visible=True) as meta_group:
        with gr.Row():
            register = gr.Dropdown(
                choices=["colloquial", "standard", "formal", "slang"],
                value="colloquial",
                label="Register"
            )
            power = gr.Dropdown(
                choices=["equal", "inferior_to_superior", "superior_to_inferior",
                         "any", "customer_to_staff", "customer_to_seller"],
                value="equal",
                label="Power relation"
            )
            tone = gr.Textbox(
                label="Tone",
                placeholder="e.g. Neutral, Warm, Casual",
                value="Neutral"
            )

    # Show/hide fields based on stage
    def update_visibility(selected_stage):
        show_context = selected_stage != "Stage 1 — Utterance only"
        show_meta    = selected_stage == "Stage 3 — Full pragmatic input (best)"
        return gr.update(visible=show_context), gr.update(visible=show_meta)

    stage.change(
        fn=update_visibility,
        inputs=stage,
        outputs=[context_group, meta_group]
    )

    predict_btn = gr.Button("Classify", variant="primary")

    gr.Markdown("### Result")
    summary_out = gr.Markdown()
    scores_out  = gr.Label(label="Confidence scores", num_top_classes=6)

    predict_btn.click(
        fn=predict,
        inputs=[stage, utterance, context, instruction, register, power, tone],
        outputs=[scores_out, summary_out]
    )

    gr.Markdown("""
    ---
    **Classes:** neutral · polite · informal · professional · blunt · rude

    **Dataset:** [freococo/burmese-contextual-pragmatics](https://huggingface.co/datasets/freococo/burmese-contextual-pragmatics)

    **Models:** [Stage 1](https://huggingface.co/annasus10/xlmr-burmese-pragmatics-stage1) ·
    [Stage 2](https://huggingface.co/annasus10/xlmr-burmese-pragmatics-stage2) ·
    [Stage 3](https://huggingface.co/annasus10/xlmr-burmese-pragmatics-stage3)
    """)

demo.launch()