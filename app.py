"""Customer Support Ticket Router - Gradio app.

    python app.py          -> open http://127.0.0.1:7860

The main decision is made by the quantized Transformer (BERT-mini distilled, INT8).
All other models of the ladder are shown side by side for comparison.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import gradio as gr  # noqa: E402

from pipeline import (APP_MODEL, MODEL_FILES, MODELS_DIR, Preprocessor, load_model,  # noqa: E402
                      load_thresholds, route)

if not os.path.exists(os.path.join(MODELS_DIR, MODEL_FILES[APP_MODEL])):
    sys.exit("No trained models found. Run `python train.py` first (see README).")

print("Loading models ...")
PRE = Preprocessor.load(os.path.join(MODELS_DIR, "vocab.json"))
THRESHOLDS = load_thresholds()
MODELS = {}
for name in MODEL_FILES:
    if os.path.exists(os.path.join(MODELS_DIR, MODEL_FILES[name])):
        MODELS[name] = load_model(name)
print("Loaded:", ", ".join(MODELS))

EXAMPLES = [
    "Hello!",
    "my dishwaser is not draining, model DW-6534",
    "I lost my credit card yesterday, please block it",
    "the fridge is making a weird buzzing noise",
    "can you book me a flight to Istanbul for Friday",
    "who won the football match last night",
    "is my washing machine still under warranty?",
]


def handle(message):
    if not message or not message.strip():
        return "Please type a message.", None
    rows = []
    main = None
    for name, model in MODELS.items():
        start = time.perf_counter()
        r = route(message, PRE, model, THRESHOLDS.get(name, 0.5))
        ms = (time.perf_counter() - start) * 1000
        rows.append([name, r["intent"], round(r["confidence"], 3), r["team"], round(ms, 1)])
        if name == APP_MODEL:
            main, main_ms = r, ms

    r = main
    codes = ", ".join(r["model_codes"]) or "none"
    fixes = ", ".join(f"{a} -> {b}" for a, b in r["corrections"]) or "none"
    top3 = " · ".join(f"`{i}` {p:.2f}" for i, p in r["top3"])
    md = f"""### Routed to: **{r['team']}**
**Intent:** `{r['intent']}`  |  **Confidence:** {r['confidence']:.2f}  |  **Decision:** {r['status']}

{"**Bot reply:** " + r['reply'] if r['reply'] else ""}

| Step | Output |
|---|---|
| Normalized text (regex) | `{r['normalized']}` |
| Spelling fixes (edit distance) | {fixes} |
| Model text | `{r['text']}` |
| Extracted model codes (regex) | {codes} |
| Top-3 intents | {top3} |
| Model / latency | {APP_MODEL}, {main_ms:.1f} ms on CPU |
"""
    return md, rows


with gr.Blocks(title="Support Ticket Router") as demo:
    gr.Markdown("# Customer Support Ticket Router\n"
                "Type a customer message. It is normalized, spell-checked and classified by the "
                f"**quantized Transformer ({APP_MODEL})**, then routed to a team. "
                "Low-confidence or out-of-scope messages go to a **human agent**; greetings get a rule-based reply.")
    with gr.Row():
        inp = gr.Textbox(label="Customer message", placeholder="e.g. my dishwasher won't drain, model DW-6534",
                         lines=2, scale=4)
        btn = gr.Button("Route message", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inputs=inp)
    out = gr.Markdown()
    table = gr.Dataframe(headers=["Model", "Intent", "Confidence", "Team", "Latency (ms)"],
                         label="All models side by side", interactive=False)
    btn.click(handle, inp, [out, table])
    inp.submit(handle, inp, [out, table])

if __name__ == "__main__":
    demo.launch()
