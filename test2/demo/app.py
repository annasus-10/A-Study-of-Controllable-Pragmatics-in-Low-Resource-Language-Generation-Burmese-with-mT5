import gradio as gr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ────────────────────────────────────────────────────────────────────

POLITENESS_MODELS = {
    "Stage 1 — Utterance only":              "annasus10/xlmr-burmese-pragmatics-stage1-v2",
    "Stage 2 — Utterance + context":         "annasus10/xlmr-burmese-pragmatics-stage2-v2",
    "Stage 3 — Oracle (ground truth meta)":  "annasus10/xlmr-burmese-pragmatics-stage3-v2",
}

CLASSIFIER_REPOS = {
    "register": "annasus10/xlmr-burmese-register",
    "power":    "annasus10/xlmr-burmese-power",
    "tone":     "annasus10/xlmr-burmese-tone",
}

POLITENESS_LABELS = ["blunt", "informal", "neutral", "polite", "professional", "rude"]
REGISTER_LABELS   = ["colloquial", "formal", "slang", "standard"]
POWER_LABELS      = ["any", "equal", "inferior_to_superior", "superior_to_inferior"]
TONE_LABELS       = ["emotional", "formal", "negative", "neutral", "positive"]

MAX_LENGTH = 128
tokenizer  = AutoTokenizer.from_pretrained("xlm-roberta-base")
model_cache = {}

def load_model(repo, num_labels):
    if repo not in model_cache:
        model_cache[repo] = AutoModelForSequenceClassification.from_pretrained(
            repo, num_labels=num_labels
        ).eval()
    return model_cache[repo]

def run_classifier(model, text, label_list):
    inputs = tokenizer(text, max_length=MAX_LENGTH, truncation=True,
                       padding="max_length", return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze().numpy()
    pred  = label_list[int(np.argmax(probs))]
    return pred, {l: float(p) for l, p in zip(label_list, probs)}

# ── Stage 1/2/3 prediction ────────────────────────────────────────────────────

def predict_staged(stage, utterance, context, instruction, register, power, tone):
    if not utterance.strip():
        return {}, "Please enter a Burmese utterance."

    if stage == "Stage 1 — Utterance only":
        text = utterance.strip()

    elif stage == "Stage 2 — Utterance + context":
        text = (utterance.strip()
                + " </s> " + context.strip()
                + " </s> " + instruction.strip())

    else:  # Stage 3
        text = (f"[register: {register}] [power: {power}] [tone: {tone}] "
                + utterance.strip()
                + " </s> " + context.strip()
                + " </s> " + instruction.strip())

    model = load_model(POLITENESS_MODELS[stage], len(POLITENESS_LABELS))
    pred, scores = run_classifier(model, text, POLITENESS_LABELS)
    summary = f"Predicted: **{pred}** ({scores[pred]*100:.1f}% confidence)"
    return scores, summary

# ── Stage 4 end-to-end prediction ────────────────────────────────────────────

def predict_pipeline(utterance, context, instruction):
    if not utterance.strip():
        return {}, {}, {}, {}, "Please enter a Burmese utterance."

    # Classifier A — register
    model_A = load_model(CLASSIFIER_REPOS["register"], len(REGISTER_LABELS))
    pred_reg, scores_reg = run_classifier(model_A, utterance.strip(), REGISTER_LABELS)

    # Classifier B — power relation
    text_B = utterance.strip() + " </s> " + context.strip()
    model_B = load_model(CLASSIFIER_REPOS["power"], len(POWER_LABELS))
    pred_pow, scores_pow = run_classifier(model_B, text_B, POWER_LABELS)

    # Classifier C — tone
    text_C = utterance.strip() + " </s> " + context.strip()
    model_C = load_model(CLASSIFIER_REPOS["tone"], len(TONE_LABELS))
    pred_tone, scores_tone = run_classifier(model_C, text_C, TONE_LABELS)

    # Final politeness classifier with predicted metadata
    text_S4 = (f"[register: {pred_reg}] [power: {pred_pow}] [tone: {pred_tone}] "
               + utterance.strip()
               + " </s> " + context.strip()
               + " </s> " + instruction.strip())

    model_S3 = load_model(POLITENESS_MODELS["Stage 3 — Oracle (ground truth meta)"],
                          len(POLITENESS_LABELS))
    pred_pol, scores_pol = run_classifier(model_S3, text_S4, POLITENESS_LABELS)

    summary = (f"**Register:** {pred_reg} | "
               f"**Power:** {pred_pow} | "
               f"**Tone:** {pred_tone}\n\n"
               f"**Politeness:** {pred_pol} ({scores_pol[pred_pol]*100:.1f}% confidence)")

    return scores_reg, scores_pow, scores_tone, scores_pol, summary

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Burmese Pragmatics Classifier v2") as demo:

    gr.Markdown("""
    # Burmese Politeness Classifier — v2
    **Group: AttentionIsAllUNeed** — NLP Final Project

    Classifies the **politeness level** of Burmese utterances using fine-tuned XLM-RoBERTa.
    Tone labels are grounded in **Appraisal Theory** (Martin & White, 2005).

    | Stage | Input | Test Macro F1 |
    |---|---|---|
    | Stage 1 | Utterance only | 0.701 |
    | Stage 2 | + context + instruction | 0.758 |
    | Stage 3 | + oracle register/power/tone | 0.799 |
    | Stage 4 | + predicted register/power/tone | 0.702 |
    """)

    with gr.Tabs():

        # ── Tab 1: Stages 1/2/3 ──────────────────────────────────────────────
        with gr.Tab("Stages 1 / 2 / 3"):
            stage = gr.Dropdown(
                choices=list(POLITENESS_MODELS.keys()),
                value="Stage 3 — Oracle (ground truth meta)",
                label="Model stage"
            )

            utterance_1 = gr.Textbox(
                label="Burmese utterance (required)",
                placeholder="မင်း တကယ် လှတယ်။",
                lines=2
            )

            with gr.Group(visible=True) as ctx_group:
                context_1    = gr.Textbox(label="Context",     placeholder="e.g. General social interaction.", lines=1)
                instruction_1= gr.Textbox(label="Instruction", placeholder="e.g. Simple standard compliment.", lines=1)

            with gr.Group(visible=True) as meta_group:
                with gr.Row():
                    register_1 = gr.Dropdown(
                        choices=REGISTER_LABELS, value="colloquial", label="Register"
                    )
                    power_1 = gr.Dropdown(
                        choices=POWER_LABELS, value="equal", label="Power relation"
                    )
                    tone_1 = gr.Dropdown(
                        choices=TONE_LABELS, value="neutral",
                        label="Tone (Appraisal Theory)"
                    )

            def update_visibility(s):
                show_ctx  = s != "Stage 1 — Utterance only"
                show_meta = s == "Stage 3 — Oracle (ground truth meta)"
                return gr.update(visible=show_ctx), gr.update(visible=show_meta)

            stage.change(fn=update_visibility, inputs=stage,
                         outputs=[ctx_group, meta_group])

            btn_1      = gr.Button("Classify", variant="primary")
            summary_1  = gr.Markdown()
            scores_1   = gr.Label(label="Confidence scores", num_top_classes=6)

            btn_1.click(
                fn=predict_staged,
                inputs=[stage, utterance_1, context_1, instruction_1,
                        register_1, power_1, tone_1],
                outputs=[scores_1, summary_1]
            )

        # ── Tab 2: Stage 4 end-to-end ─────────────────────────────────────────
        with gr.Tab("Stage 4 — End-to-End Pipeline"):
            gr.Markdown("""
            ### Fully automatic pipeline
            Enter only the utterance, context, and instruction.
            Register, power relation, and tone are **predicted automatically**
            by three intermediate classifiers, then fed into the politeness model.
            """)

            utterance_4   = gr.Textbox(label="Burmese utterance (required)",
                                        placeholder="မင်း တကယ် လှတယ်။", lines=2)
            context_4     = gr.Textbox(label="Context",
                                        placeholder="e.g. General social interaction.", lines=1)
            instruction_4 = gr.Textbox(label="Instruction",
                                        placeholder="e.g. Simple standard compliment.", lines=1)

            btn_4 = gr.Button("Run Pipeline", variant="primary")

            summary_4  = gr.Markdown()

            with gr.Row():
                scores_reg  = gr.Label(label="Predicted register",  num_top_classes=4)
                scores_pow  = gr.Label(label="Predicted power",      num_top_classes=4)
                scores_tone = gr.Label(label="Predicted tone",       num_top_classes=5)

            scores_pol = gr.Label(label="Final politeness", num_top_classes=6)

            btn_4.click(
                fn=predict_pipeline,
                inputs=[utterance_4, context_4, instruction_4],
                outputs=[scores_reg, scores_pow, scores_tone, scores_pol, summary_4]
            )

    gr.Markdown("""
    ---
    **Dataset:** [freococo/burmese-contextual-pragmatics](https://huggingface.co/datasets/freococo/burmese-contextual-pragmatics)
    · **Models:** [Stage 1](https://huggingface.co/annasus10/xlmr-burmese-pragmatics-stage1-v2)
    · [Stage 2](https://huggingface.co/annasus10/xlmr-burmese-pragmatics-stage2-v2)
    · [Stage 3](https://huggingface.co/annasus10/xlmr-burmese-pragmatics-stage3-v2)
    · [Register](https://huggingface.co/annasus10/xlmr-burmese-register)
    · [Power](https://huggingface.co/annasus10/xlmr-burmese-power)
    · [Tone](https://huggingface.co/annasus10/xlmr-burmese-tone)
    """)

demo.launch()