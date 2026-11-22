"""Draws docs/architecture.png (the architecture diagram used in the slides)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.png")
C = {"ui": "#dbe9f6", "b": "#fde2c8", "model": "#d7f0dd", "q": "#f9c79a", "post": "#ece0f5", "train": "#eeeeee"}

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.axis("off")


def box(x, y, w, h, text, color, bold=False, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=color, ec="#444", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(x1, y1, x2, y2, label=None, style="-|>", dx=0, dy=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16, lw=1.4, color="#333"))
    if label:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, label, ha="center", fontsize=8.5, color="#333")


# runtime frame
ax.add_patch(FancyBboxPatch((2, 28), 156, 60, boxstyle="round,pad=0.3", fc="none", ec="#2a6f97", lw=1.5, ls="--"))
ax.text(4, 85.5, "RUNTIME - app.py: one Python process on the user's CPU (no GPU, no cloud API)", fontsize=10,
        color="#2a6f97", fontweight="bold")

box(5, 62, 22, 16, "User\n(browser)\n\nGradio web UI\ntext box", C["ui"], True)
box(33, 62, 30, 16, "Preprocessing  [Area B]\n\n1. regex normalization\n(lower-case, URLs, repeats)\n"
    "2. regex model-code extraction\n(e.g. DW-6534)\n3. edit-distance spelling fix", C["b"], fs=9)
box(69, 70, 26, 10, "ELIZA-style regex rules\n[Area B]\ngreeting / thanks / bye?", C["b"], fs=9)
box(69, 46, 26, 18, "Intent model (main)\nBERT-mini distilled\nINT8 dynamic quantized\n(12 MB)  [Areas C + D]\n"
    "WordPiece tokenizer", C["q"], True, fs=9)
box(101, 46, 26, 18, "Post-processing\n\nsoftmax confidence\n< threshold or 'oos'\n-> human agent\nelse intent -> team map",
    C["post"], fs=9)
box(133, 58, 22, 20, "Output in UI\n\nrouted team\nintent + confidence\nbot reply (small talk)\nmodel codes, latency",
    C["ui"], True, fs=9)
box(33, 31, 122, 10, "Comparison models shown side by side (same preprocessing):   tf-idf + LogReg (FP64, sklearn)   |   "
    "own word2vec + FFN (FP32)   |   BERT-base (FP32)   |   BERT-mini distilled (FP32)", C["model"], fs=9)

arrow(27, 70, 33, 70, "message")
arrow(63, 72, 69, 75)
arrow(63, 66, 69, 58, "clean text", dx=5, dy=2)
arrow(95, 75, 133, 72, "rule matched: reply")
arrow(95, 55, 101, 55, "probabilities")
arrow(127, 55, 133, 63, "team")
arrow(48, 62, 48, 41, style="-|>")

# training (offline)
ax.add_patch(FancyBboxPatch((2, 1), 156, 24, boxstyle="round,pad=0.3", fc="none", ec="#777", lw=1.5, ls="--"))
ax.text(4, 22.5, "OFFLINE TRAINING - train.py (CPU or Colab GPU)", fontsize=10, color="#555", fontweight="bold")
box(5, 4, 26, 15, "Data\nCLINC150 (150 intents + oos)\n+ 80 hand-written\nappliance messages", C["train"], fs=9)
box(37, 4, 22, 15, "tf-idf + LogReg\nbaseline [B]\n\nword2vec (skip-gram)\n+ FFN, SGD [A]", C["train"], fs=9)
box(65, 4, 22, 15, "BERT-base\nfine-tuned with\nclassification head\n(teacher) [C]", C["train"], fs=9)
box(93, 4, 24, 15, "Knowledge distillation\nteacher -> BERT-mini\n(4 layers, 256 dim) [D]", C["train"], fs=9)
box(123, 4, 32, 15, "Dynamic INT8 quantization\n(Linear + Embedding) [D]\n\nevaluation on same test set\n-> results table", C["train"], fs=9)
arrow(31, 11.5, 37, 11.5)
arrow(59, 11.5, 65, 11.5)
arrow(87, 11.5, 93, 11.5, "logits")
arrow(117, 11.5, 123, 11.5)
arrow(139, 19, 139, 28, "trained models/ folder", style="-|>", dx=10, dy=0)

fig.tight_layout()
fig.savefig(OUT, dpi=160)
print("saved", OUT)
