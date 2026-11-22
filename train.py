"""Train and evaluate the whole model ladder, then write the results table.

    python train.py                      # full run (BERT-base teacher takes the longest)
    python train.py --fast               # quick smoke test on a small subset

Each stage is skipped if its model already exists in models/ (use --force to retrain).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data import OOS, label_list, load_splits  # noqa: E402
from evaluate import hardware, measure_latency, metrics, predict_with_threshold, tune_threshold  # noqa: E402
from models import BertClassifier, TfidfLogReg, W2VFeedForward, dir_size_mb  # noqa: E402
from pipeline import MODEL_FILES, MODELS_DIR, RESULTS_DIR, APP_MODEL, Preprocessor, load_model  # noqa: E402

TEACHER = "BERT-base (FP32)"
STUDENT = "BERT-mini distilled (FP32)"
STUDENT_Q = APP_MODEL
TEACHER_Q = "BERT-base (INT8)"  # extra row: shows quantization on the big model too


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="bert-base-uncased", help="HF hub id or local folder")
    ap.add_argument("--student", default="google/bert_uncased_L-4_H-256_A-4", help="HF hub id or local folder")
    ap.add_argument("--teacher-epochs", type=int, default=3)
    ap.add_argument("--student-epochs", type=int, default=10)
    ap.add_argument("--fast", action="store_true", help="tiny subset, for testing the code")
    ap.add_argument("--force", action="store_true", help="retrain even if models exist")
    args = ap.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------ data + preprocessing
    labels = label_list()
    splits = load_splits()
    if args.fast:
        rng = np.random.RandomState(0)
        for k, n in [("train", 3000), ("val", 600), ("test", 800)]:
            idx = rng.choice(len(splits[k]), n, replace=False)
            splits[k] = [splits[k][i] for i in idx]
        args.teacher_epochs, args.student_epochs = 1, 2
    for k, v in splits.items():
        print(f"{k}: {len(v)} messages")

    train_vocab = {w for t, _ in splits["train"] for w in t.lower().split()}
    pre = Preprocessor(train_vocab)
    pre.save(os.path.join(MODELS_DIR, "vocab.json"))
    X = {k: [pre(t)["text"] for t, _ in v] for k, v in splits.items()}
    Y = {k: [y for _, y in v] for k, v in splits.items()}

    def path(name):
        return os.path.join(MODELS_DIR, MODEL_FILES.get(name, "bert_teacher_int8"))

    def need(name):
        return args.force or not os.path.exists(path(name))

    timings = {}

    # ------------------------------------------------------------ 1. classical baseline
    name = "tf-idf + LogReg (baseline)"
    if need(name):
        print(f"\n=== Training {name}")
        t0 = time.time()
        TfidfLogReg(labels).fit(X["train"], Y["train"]).save(path(name))
        timings[name] = time.time() - t0

    # ------------------------------------------------------------ 2. word2vec + FFN
    name = "word2vec + FFN"
    if need(name):
        print(f"\n=== Training {name}")
        t0 = time.time()
        W2VFeedForward(labels).fit(X["train"], Y["train"], epochs=20 if args.fast else 80).save(path(name))
        timings[name] = time.time() - t0

    # ------------------------------------------------------------ 3. BERT-base teacher
    if need(TEACHER):
        print(f"\n=== Fine-tuning {TEACHER} ({args.teacher})")
        t0 = time.time()
        teacher = BertClassifier.from_pretrained(TEACHER, args.teacher, labels)
        teacher.fit(X["train"], Y["train"], epochs=args.teacher_epochs, lr=5e-5, batch_size=32)
        teacher.save(path(TEACHER))
        timings[TEACHER] = time.time() - t0

    # ------------------------------------------------------------ 4. distilled student
    if need(STUDENT):
        print(f"\n=== Distilling {TEACHER} -> {STUDENT} ({args.student})")
        t0 = time.time()
        teacher = load_model(TEACHER)
        print("  computing teacher logits on the training set ...")
        t_logits = teacher.logits(X["train"])
        student = BertClassifier.from_pretrained(STUDENT, args.student, labels)
        student.fit(X["train"], Y["train"], epochs=args.student_epochs, lr=2e-4, batch_size=64,
                    teacher_logits=t_logits, temperature=2.0, alpha=0.5)
        student.save(path(STUDENT))
        timings[STUDENT] = time.time() - t0

    # ------------------------------------------------------------ 5. quantization
    if need(STUDENT_Q):
        print(f"\n=== Quantizing {STUDENT} -> INT8")
        load_model(STUDENT).quantized(STUDENT_Q).save_quantized(path(STUDENT_Q))
    if need(TEACHER_Q):
        print(f"\n=== Quantizing {TEACHER} -> INT8")
        load_model(TEACHER).quantized(TEACHER_Q).save_quantized(path(TEACHER_Q))

    # ------------------------------------------------------------ evaluation
    print("\n=== Evaluating every model on the same validation + test sets")
    rows, thresholds, failures = [], {}, []
    names = [n for n in MODEL_FILES] + [TEACHER_Q]
    for name in names:
        if name == TEACHER_Q:
            model = BertClassifier.load_quantized(TEACHER_Q, path(TEACHER_Q), labels)
        else:
            model = load_model(name)
        thr = tune_threshold(model.predict_proba(X["val"]), labels, Y["val"])
        thresholds[name] = thr
        probs = model.predict_proba(X["test"])
        pred, conf = predict_with_threshold(probs, labels, thr)
        m = metrics(pred, Y["test"])
        lat, p95 = measure_latency(model, X["test"], runs=50 if args.fast else 200)
        size = dir_size_mb(path(name))
        if name.startswith("word2vec"):
            size = dir_size_mb(path(name))
        rows.append({"model": name, **m, "threshold": thr, "size_mb": size,
                     "latency_ms": lat, "latency_p95_ms": p95})
        print(f"  {name:30s} acc={m['accuracy']:.4f} oos_recall={m['oos_recall']:.4f} "
              f"size={size:.1f}MB latency={lat:.2f}ms")
        if name == APP_MODEL:
            for (raw, gold), p, c in zip(splits["test"], pred, conf):
                if p != gold:
                    failures.append({"text": raw, "gold": gold, "pred": p, "confidence": float(c)})

    with open(os.path.join(MODELS_DIR, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    hw = hardware()
    result = {"hardware": hw, "latency_runs": 50 if args.fast else 200, "fast_mode": args.fast,
              "test_size": len(Y["test"]), "test_oos": sum(y == OOS for y in Y["test"]),
              "train_size": len(Y["train"]), "val_size": len(Y["val"]),
              "training_seconds": timings, "rows": rows}
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "failures.json"), "w") as f:
        json.dump(failures, f, indent=2)
    write_markdown(result)
    plot(result)
    print(f"\nDone. Results in {RESULTS_DIR}/results.md")


def write_markdown(r):
    lines = [f"Test set: {r['test_size']} messages ({r['test_oos']} out-of-scope). "
             f"Latency: mean over {r['latency_runs']} single-message predictions on {r['hardware']}.", "",
             "| Model | Accuracy | In-scope acc. | OOS recall | Routing (team) acc. | Size (MB) | Latency (ms) |",
             "|---|---|---|---|---|---|---|"]
    for x in r["rows"]:
        lines.append(f"| {x['model']} | {x['accuracy']*100:.1f}% | {x['in_scope_accuracy']*100:.1f}% | "
                     f"{x['oos_recall']*100:.1f}% | {x['team_accuracy']*100:.1f}% | {x['size_mb']:.1f} | "
                     f"{x['latency_ms']:.2f} |")
    with open(os.path.join(RESULTS_DIR, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = r["rows"]
    names = [x["model"].replace(" (", "\n(") for x in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, key, title, fmt in [
        (axes[0], "accuracy", "Test accuracy (%)", lambda v: f"{v*100:.1f}"),
        (axes[1], "size_mb", "Model size (MB)", lambda v: f"{v:.1f}"),
        (axes[2], "latency_ms", "Latency per message (ms, 1 thread)", lambda v: f"{v:.1f}"),
    ]:
        vals = [x[key] * (100 if key == "accuracy" else 1) for x in rows]
        bars = ax.bar(range(len(rows)), vals, color="#2a6f97")
        for b, x in zip(bars, rows):
            if x["model"] == APP_MODEL:
                b.set_color("#e07a1f")
        for b, x in zip(bars, rows):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt(x[key]), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model ladder on the same test set (orange = model used in the app)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "results.png"), dpi=150)


if __name__ == "__main__":
    main()
