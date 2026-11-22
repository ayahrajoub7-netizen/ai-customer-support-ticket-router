"""Builds the presentation (docs/presentation.pptx) from YOUR results in results/results.json.

    python docs/make_slides.py --name "Your Name" --student-id 123456 \
        --repo https://github.com/you/ticket-router --video https://youtu.be/xxxx

Then open the .pptx (PowerPoint / Google Slides / LibreOffice) and export it as PDF,
or run:  soffice --headless --convert-to pdf docs/presentation.pptx --outdir docs
"""
import argparse
import json
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DOCS = os.path.join(ROOT, "docs")
BLUE = RGBColor(0x1F, 0x4E, 0x79)
GREY = RGBColor(0x55, 0x55, 0x55)
APP_MODEL = "BERT-mini distilled (INT8)"


def add_slide(prs, title, section=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.9)).text_frame
    tb.text = title
    tb.paragraphs[0].font.size = Pt(30)
    tb.paragraphs[0].font.bold = True
    tb.paragraphs[0].font.color.rgb = BLUE
    if section:
        st = s.shapes.add_textbox(Inches(10.3), Inches(7.0), Inches(2.9), Inches(0.4)).text_frame
        st.text = section
        st.paragraphs[0].font.size = Pt(12)
        st.paragraphs[0].font.color.rgb = GREY
    return s


def bullets(slide, items, x=0.6, y=1.3, w=12.1, h=5.8, size=18):
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        level = 1 if item.startswith("  ") else 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if level == 0 else "– ") + item.strip()
        p.level = level
        p.font.size = Pt(size - 2 * level)
        p.space_after = Pt(6)
    return tf


def table(slide, rows, x=0.5, y=1.4, w=12.3, col_w=None, size=13):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                                   Inches(0.4 * len(rows)))
    t = shape.table
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(size)
                p.font.bold = r == 0
    if col_w:
        for c, cw in enumerate(col_w):
            t.columns[c].width = Inches(cw)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="[Your Name]")
    ap.add_argument("--student-id", default="[Student ID]")
    ap.add_argument("--repo", default="[GitHub repository link]")
    ap.add_argument("--video", default="[Demo video link]")
    args = ap.parse_args()

    R = json.load(open(os.path.join(RES, "results.json")))
    rows = {x["model"]: x for x in R["rows"]}
    failures = json.load(open(os.path.join(RES, "failures.json")))
    base = rows["tf-idf + LogReg (baseline)"]
    ffn = rows["word2vec + FFN"]
    teacher = rows["BERT-base (FP32)"]
    student = rows["BERT-mini distilled (FP32)"]
    q = rows[APP_MODEL]
    pct = lambda v: f"{v * 100:.1f}%"  # noqa: E731

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # 1 ---------------------------------------------------------------- title
    s = add_slide(prs, "Customer Support Ticket Router")
    bullets(s, [
        "SWE015 – Introduction to Large Language Models · Homework: Building an Intelligent Application",
        "Use Case 2 – Customer Support Ticket Router",
        f"Name: {args.name}",
        f"Student ID: {args.student_id}",
        f"Repository: {args.repo}",
        f"Demo video: {args.video}",
    ], y=1.6, size=22)

    # 2 ---------------------------------------------------------------- introduction
    s = add_slide(prs, "Introduction – the problem", "Introduction")
    bullets(s, [
        "A home-appliance brand gets thousands of support messages a day, e.g. "
        "“my dishwasher won't drain, model DW-6534”. Each must reach the right team instantly.",
        "Who it is for: the support centre – customers get a faster answer, agents get fewer misrouted tickets.",
        "Our application: free-text message → normalize → predict the intent with a confidence → show the team.",
        "  Low-confidence or out-of-scope messages → “Human agent”; greetings / small talk → rule-based reply.",
        "  The quantized Transformer makes the decision; all models are shown side by side.",
        "What success looks like:",
        "  high intent accuracy on a held-out test set, high out-of-scope recall (unknown requests are not misrouted),",
        "  a small model that answers in a few milliseconds on a CPU.",
    ])

    # 3 ---------------------------------------------------------------- data
    s = add_slide(prs, "Data – sources, splits, preprocessing", "Development")
    bullets(s, [
        "CLINC150 (Larson et al., 2019, github.com/clinc/oos-eval, “data_full”): 150 intents in 10 domains "
        "+ an out-of-scope (oos) class.",
        "+ 80 hand-written home-appliance messages (4 intents: dishwasher, washer, fridge, warranty/repair), "
        "split 12 / 3 / 5 per intent; train copies oversampled ×4.",
        f"Splits used by every model: train {R['train_size']:,} · validation {R['val_size']:,} · "
        f"test {R['test_size']:,} ({R['test_oos']:,} out-of-scope).",
        "Intent → domain → support team (11 teams + “Human agent”).",
        "Preprocessing (same for all models): regex lower-casing, URL/e-mail removal, repeated-letter "
        "collapsing, model-code extraction (DW-6534), edit-distance spelling correction of product words.",
        "The validation set is used only to choose each model's confidence threshold for out-of-scope.",
    ])

    # 4 ---------------------------------------------------------------- architecture
    s = add_slide(prs, "Architecture", "Development")
    s.shapes.add_picture(os.path.join(DOCS, "architecture.png"), Inches(1.2), Inches(1.1), height=Inches(5.8))

    # 5 ---------------------------------------------------------------- technique map
    s = add_slide(prs, "Technique map", "Development")
    table(s, [
        ["Area", "Technique", "Component in the application (file)"],
        ["A · Neural networks", "Feedforward network trained with SGD + backpropagation, dropout",
         "word2vec + FFN intent classifier (src/models.py: W2VFeedForward)"],
        ["B · Classical NLP", "Regular expressions", "Normalization + model-code extraction (src/preprocess.py)"],
        ["B · Classical NLP", "Minimum edit distance", "Spelling correction of product names (SpellCorrector)"],
        ["B · Classical NLP", "ELIZA-style rules", "Small-talk replies (small_talk_reply)"],
        ["B · Classical NLP", "tf-idf + logistic regression; word2vec", "Baseline (TfidfLogReg); own skip-gram embeddings"],
        ["C · Transformers", "Fine-tuning BERT with a task head; WordPiece tokenizer",
         "BERT-base intent classifier = teacher (BertClassifier)"],
        ["D · Efficient models", "Knowledge distillation", "BERT-base → BERT-mini student (train.py)"],
        ["D · Efficient models", "Dynamic INT8 quantization; benchmarking size/latency/accuracy",
         "BERT-mini INT8 – the model the app runs (quantize, src/evaluate.py)"],
    ], col_w=[2.4, 4.6, 5.3], size=12)

    # 6 ---------------------------------------------------------------- models
    s = add_slide(prs, "The model ladder – how each model was trained", "Development")
    bullets(s, [
        "1. Classical baseline: tf-idf (unigrams + bigrams, sublinear tf) + logistic regression (C = 20).",
        "2. Neural network: our own word2vec (skip-gram, 200-d, trained on the training messages) → averaged "
        "message vector → feedforward net (200 → 512 ReLU → dropout 0.3 → 155), SGD with momentum 0.9, "
        "lr 0.5 cosine schedule, 80 epochs.",
        "3. Transformer: bert-base-uncased (110M params) + classification head, fine-tuned 3 epochs, AdamW "
        "lr 5e-5, batch 32, max 40 WordPiece tokens.",
        "4. Distilled student: BERT-mini (google/bert_uncased_L-4_H-256_A-4, 4 layers, 11M params) trained "
        "10 epochs on 0.5 · cross-entropy + 0.5 · T² · KL(teacher ‖ student), temperature T = 2.",
        "5. Quantized Transformer: the student after dynamic INT8 quantization (used in the app).",
        "Out-of-scope: “oos” is a training class, plus a confidence threshold per model tuned on validation.",
    ], size=17)

    # 7 ---------------------------------------------------------------- quantization
    s = add_slide(prs, "Distillation + quantization method", "Development")
    bullets(s, [
        "Why: BERT-base is accurate but large and slow on a CPU-only support server.",
        "Step 1 – knowledge distillation: the student learns from the teacher's soft probabilities "
        "(which wrong intents are “close”), not only from the gold label.",
        "Step 2 – post-training dynamic quantization (PyTorch torch.ao.quantization.quantize_dynamic):",
        "  every nn.Linear: weights stored as INT8, activations quantized on the fly at inference;",
        "  every nn.Embedding: weight-only 8-bit quantization (embeddings are most of a small BERT's size).",
        "No retraining and no calibration data are needed; runs on any CPU.",
        f"Effect on the student: {student['size_mb']:.1f} MB → {q['size_mb']:.1f} MB, "
        f"accuracy {pct(student['accuracy'])} → {pct(q['accuracy'])}, "
        f"latency {student['latency_ms']:.1f} ms → {q['latency_ms']:.1f} ms.",
        f"Same method on BERT-base for reference: {teacher['size_mb']:.0f} MB → "
        f"{rows['BERT-base (INT8)']['size_mb']:.0f} MB.",
    ])

    # 8 ---------------------------------------------------------------- results table
    s = add_slide(prs, "Results – same held-out test set", "Development")
    trows = [["Model", "Accuracy", "OOS recall", "In-scope acc.", "Team routing acc.", "Size (MB)", "Latency (ms)"]]
    for x in R["rows"]:
        trows.append([x["model"] + ("  ← app" if x["model"] == APP_MODEL else ""), pct(x["accuracy"]),
                      pct(x["oos_recall"]), pct(x["in_scope_accuracy"]), pct(x["team_accuracy"]),
                      f"{x['size_mb']:.1f}", f"{x['latency_ms']:.2f}"])
    table(s, trows, y=1.3, col_w=[3.6, 1.3, 1.4, 1.5, 1.8, 1.2, 1.5], size=13)
    note = s.shapes.add_textbox(Inches(0.5), Inches(4.6), Inches(12.3), Inches(1.6)).text_frame
    note.word_wrap = True
    note.text = (f"Test set: {R['test_size']:,} messages ({R['test_oos']:,} out-of-scope). Accuracy counts oos as a "
                 f"class. Latency = mean time to classify one message, averaged over {R['latency_runs']} runs. "
                 f"Hardware: {R['hardware']}. Size = size of the saved model files.")
    note.paragraphs[0].font.size = Pt(13)
    note.paragraphs[0].font.color.rgb = GREY
    s.shapes.add_picture(os.path.join(RES, "results.png"), Inches(7.6), Inches(5.3), height=Inches(2.1))

    # 9 ---------------------------------------------------------------- conclusion
    best = max(R["rows"], key=lambda x: x["accuracy"])
    speedup = teacher["latency_ms"] / q["latency_ms"]
    shrink = teacher["size_mb"] / q["size_mb"]
    s = add_slide(prs, "Conclusion – what the comparison showed", "Conclusion")
    bullets(s, [
        f"Most accurate: {best['model']} ({pct(best['accuracy'])} accuracy, "
        f"{pct(best['oos_recall'])} out-of-scope recall).",
        f"BERT-base vs. classical baseline: {pct(teacher['accuracy'])} vs. {pct(base['accuracy'])} accuracy; "
        f"out-of-scope recall {pct(teacher['oos_recall'])} vs. {pct(base['oos_recall'])}. "
        "Pre-training helps with paraphrases the baseline never saw.",
        f"The classical baseline is a strong, cheap model ({base['latency_ms']:.2f} ms, {base['size_mb']:.0f} MB), "
        "but it misroutes most out-of-scope messages: tf-idf only sees surface words.",
        (f"Our own word2vec + FFN reaches {pct(ffn['accuracy'])} (baseline {pct(base['accuracy'])}) with only "
         f"{ffn['size_mb']:.1f} MB and {ffn['latency_ms']:.2f} ms; but embeddings trained on ~15k short messages "
         "and averaging (no word order) keep it far below BERT."),
        f"Distilled + INT8 student (app model): {pct(q['accuracy'])} accuracy, {shrink:.0f}× smaller and "
        f"{speedup:.0f}× faster than BERT-base ({q['latency_ms']:.1f} ms per message on one CPU thread), "
        f"for {abs(teacher['accuracy'] - q['accuracy']) * 100:.1f} accuracy points.",
        (f"Quantization itself changed the student's accuracy by only "
         f"{(q['accuracy'] - student['accuracy']) * 100:+.1f} points; the gap to BERT-base comes from the "
         "smaller student, not from INT8."),
    ], size=17)

    # 10 --------------------------------------------------------------- limitations
    wrong = sorted(failures, key=lambda f: -f["confidence"])
    oos_miss = next((f for f in wrong if f["gold"] == "oos"), None)
    other = next((f for f in wrong if f["gold"] != "oos"), None)
    ex = []
    for f in [oos_miss, other]:
        if f:
            ex.append(f"  “{f['text']}” – true: {f['gold']}, predicted: {f['pred']} "
                      f"(confidence {f['confidence']:.2f})")
    s = add_slide(prs, "Limitations, failure examples, next steps", "Conclusion")
    bullets(s, [
        "Failure examples (app model, from results/failures.json):", *ex,
        "Limitations:",
        "  English only – the real scenario also has Turkish messages; BERT-uncased does not cover them.",
        "  CLINC150 is a general assistant dataset; only 80 appliance messages are domain specific.",
        "  Confident mistakes on out-of-scope messages that look like a known intent are still possible.",
        "Next steps:",
        "  Add Turkish with MASSIVE (tr-TR) and a multilingual encoder (XLM-R or BERTurk).",
        "  Collect real support tickets; export to ONNX Runtime; calibrate confidence (temperature scaling).",
    ], size=16)

    # 11 --------------------------------------------------------------- AI usage
    s = add_slide(prs, "AI usage statement", "Conclusion")
    bullets(s, [
        "Tool used: Claude (Anthropic), an AI coding assistant.",
        "Used for: choosing the use case, drafting the code (data loading, preprocessing, models, training, "
        "evaluation, Gradio app), the architecture-diagram script, the README and the slide text.",
        "Used for: writing the 80 hand-written home-appliance example messages.",
        "I ran all training and evaluation myself; every number in this presentation comes from my own run "
        "(results/results.json), not from an AI answer, paper or model card.",
        "I reviewed, tested and can explain every component of the code and the architecture.",
    ])

    out = os.path.join(DOCS, "presentation.pptx")
    prs.save(out)
    print("saved", out, f"({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
