import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import os
from functools import lru_cache

# --- Configuration & Model Loading ---
MODEL_PATH = "./fine_tuned_bert"
DEFAULT_MODEL = "bert-base-uncased"


def load_resources():
    path = MODEL_PATH if os.path.exists(MODEL_PATH) else DEFAULT_MODEL
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    mapping_path = os.path.join(MODEL_PATH, "label_mapping.json")
    id2label = None
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            id2label = json.load(f).get("id2label")
    return tokenizer, model, id2label, device


TOKENIZER, MODEL, ID2LABEL, DEVICE = load_resources()


@lru_cache(maxsize=128)
def get_prediction(text):
    if not text.strip():
        return {}
    inputs = TOKENIZER(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    with torch.inference_mode():
        logits = MODEL(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

    confidences = probabilities[0].tolist()
    if ID2LABEL:
        return {ID2LABEL[str(i)]: float(conf) for i, conf in enumerate(confidences)}
    return {f"Class {i}": float(conf) for i, conf in enumerate(confidences)}


# --- World-Class Premium CSS ---
AUTHOR_NAME = "sumitsingh159"
_initial = AUTHOR_NAME.strip()[0].upper() if AUTHOR_NAME.strip() else "A"

# ================= CSS =================
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

body {
    background: #020617;
    font-family: 'Inter', sans-serif !important;
}

/* Container */
.gradio-container {
    background: radial-gradient(circle at 20% 0%, #0f172a, #020617 60%);
}

/* Header */
#app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 40px;
}

.header-left {
    display: flex;
    gap: 14px;
    align-items: center;
    max-width: 70%;
}

.header-icon {
    font-size: 28px;
}

.header-title {
    font-size: 26px;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header-subtitle {
    font-size: 13px;
    color: #94a3b8;
    line-height: 1.5;
    margin-top: 4px;
}

/* Author badge */
.author-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.05);
    padding: 6px 12px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.1);
}

.avatar {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 700;
}

.by {
    font-size: 12px;
    color: #94a3b8;
}

.name {
    font-size: 13px;
    font-weight: 600;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.03);
    border-radius: 18px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
}

/* Input */
textarea {
    background: rgba(2,6,23,0.8) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: white !important;
    font-size: 15px !important;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 600 !important;
}

.primary-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(59,130,246,0.25);
}

/* Footer */
.footer {
    margin-top: 50px;
    text-align: center;
    color: #64748b;
    font-size: 12px;
}
"""

# ================= HEADER =================
header_html = f"""
<div id="app-header">
  <div class="header-left">
    <div class="header-icon">🧠</div>
    <div>
      <div class="header-title">Semantic Text Classification</div>
      <div class="header-subtitle">
         This project implements a sophisticated text classification framework powered by a fine-tuned BERT transformer model. It is
  architected to autonomously categorize unstructured text into four primary domains: Technology, Sports, Politics, and
  Entertainment. By analyzing deep semantic relationships and contextual nuances, the system provides high-confidence predictions
  through an optimized neural pipeline. The integrated interface enables seamless, real-time linguistic analysis, offering a robust
  foundation for automated content organization and intelligent data processing.
      </div>
    </div>
  </div>
  <div class="author-badge">
    <div class="avatar">{_initial}</div>
    
    <span class="name">{AUTHOR_NAME}</span>
  </div>
</div>
"""

# ================= FOOTER =================
footer_html = f"""
<div class="footer">
  Powered by BERT-BASE-UNCASED • Built with ❤️ by {AUTHOR_NAME}
</div>
"""

# ================= UI =================
with gr.Blocks(css=css) as demo:
    gr.HTML(header_html)

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group(elem_classes="card"):
                input_text = gr.Textbox(
                    placeholder="Paste your text for neural classification...",
                    lines=8,
                    show_label=False
                )

                with gr.Row():
                    submit_btn = gr.Button("Analyze", elem_classes="primary-btn")
                    clear_btn = gr.Button("Clear")

        with gr.Column(scale=2):
            with gr.Group(elem_classes="card"):
                output_label = gr.Label(
                    num_top_classes=4,
                    label="Prediction"
                )

    gr.Examples(
        examples=[
            ["Breakthroughs in fusion energy research could provide limitless clean power."],
            ["The underdog team clinched the victory in a spectacular 3-pointer."],
            ["Global markets reacted sharply to the central bank decision."],
            ["A new cinematic masterpiece blends psychological horror with surrealist art."]
        ],
        inputs=input_text,
        label="Try Examples"
    )

    gr.HTML(footer_html)

    # Events
    submit_btn.click(fn=get_prediction, inputs=input_text, outputs=output_label)
    input_text.submit(fn=get_prediction, inputs=input_text, outputs=output_label)
    clear_btn.click(lambda: "", None, input_text, queue=False)

# ================= RUN =================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
