"""Evaluation helpers: confidence threshold for out-of-scope, metrics, size and latency."""
import platform
import time

import numpy as np
import torch

from data import OOS, team_for


def predict_with_threshold(probs, labels, threshold):
    """Argmax intent; if confidence < threshold -> out-of-scope ('oos')."""
    idx = probs.argmax(1)
    conf = probs.max(1)
    return [labels[i] if c >= threshold else OOS for i, c in zip(idx, conf)], conf


def metrics(pred, gold):
    pred, gold = np.array(pred), np.array(gold)
    in_scope = gold != OOS
    teams_pred = np.array([team_for(p) for p in pred])
    teams_gold = np.array([team_for(g) for g in gold])
    return {
        "accuracy": float((pred == gold).mean()),                         # all test messages
        "in_scope_accuracy": float((pred[in_scope] == gold[in_scope]).mean()),
        "oos_recall": float((pred[~in_scope] == OOS).mean()),             # oos sent to human
        "team_accuracy": float((teams_pred == teams_gold).mean()),        # correct routing
    }


def tune_threshold(probs, labels, gold):
    """Pick the confidence threshold that maximizes accuracy on the VALIDATION set."""
    best_t, best_acc = 0.0, -1
    for t in np.arange(0.0, 0.96, 0.01):
        pred, _ = predict_with_threshold(probs, labels, t)
        acc = np.mean(np.array(pred) == np.array(gold))
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    return round(best_t, 2)


def measure_latency(model, texts, runs=200, warmup=10):
    """Average time (ms) to classify ONE message, over `runs` different messages, 1 CPU thread."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for t in texts[:warmup]:
            model.predict_proba([t])
        times = []
        for i in range(runs):
            t = texts[i % len(texts)]
            start = time.perf_counter()
            model.predict_proba([t])
            times.append((time.perf_counter() - start) * 1000)
    finally:
        torch.set_num_threads(old)
    return float(np.mean(times)), float(np.percentile(times, 95))


def hardware():
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if platform.system() == "Darwin":
        import subprocess
        try:
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        except Exception:
            pass
    return f"{cpu}, {platform.system()}, 1 CPU thread, PyTorch {torch.__version__}"
