"""The model ladder. Every model has the same interface:

    model.predict_proba(list_of_preprocessed_texts) -> numpy array (n_texts, n_labels)

1. TfidfLogReg      - classical baseline (Area B: tf-idf + logistic regression)
2. W2VFeedForward   - own word2vec + feedforward net trained with SGD/backprop (Area A + B)
3. BertClassifier   - BERT-base fine-tuned with a classification head (Area C)
4. BertClassifier   - BERT-mini student distilled from BERT-base (Area D, distillation)
5. BertClassifier   - the student after dynamic INT8 quantization (Area D, quantization)
"""
import os
import warnings

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def dir_size_mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / 1e6
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total / 1e6


# =================================================================== 1. tf-idf + LR
class TfidfLogReg:
    name = "tf-idf + LogReg"

    def __init__(self, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.labels = labels
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.clf = LogisticRegression(C=20, max_iter=2000)

    def fit(self, texts, y):
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, [self.labels.index(l) for l in y])
        return self

    def predict_proba(self, texts):
        p = self.clf.predict_proba(self.vec.transform(texts))
        full = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        full[:, self.clf.classes_] = p
        return full

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


# =================================================================== 2. word2vec + FFN
class FFN(nn.Module):
    """Feedforward network: averaged embedding -> hidden (ReLU, dropout) -> softmax."""

    def __init__(self, dim, hidden, n_labels, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_labels),
        )

    def forward(self, x):
        return self.net(x)


class W2VFeedForward:
    name = "word2vec + FFN"

    def __init__(self, labels, dim=200, hidden=512):
        self.labels, self.dim, self.hidden = labels, dim, hidden
        self.w2v = None
        self.net = None

    def _embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            vecs = [self.w2v.wv[w] for w in t.split() if w in self.w2v.wv]
            if vecs:
                v = np.mean(vecs, axis=0)
                out[i] = v / (np.linalg.norm(v) + 1e-8)
        return out

    def fit(self, texts, y, epochs=80, lr=0.5, batch_size=64, seed=42):
        from gensim.models import Word2Vec
        torch.manual_seed(seed)
        # (a) train our own word2vec (skip-gram) on the training messages
        sents = [t.split() for t in texts]
        self.w2v = Word2Vec(sents, vector_size=self.dim, window=5, min_count=1, sg=1,
                            negative=10, epochs=40, workers=1, seed=seed)
        # (b) feedforward network on averaged word vectors, trained with SGD + backprop
        X = torch.tensor(self._embed(texts))
        Y = torch.tensor([self.labels.index(l) for l in y])
        self.net = FFN(self.dim, self.hidden, len(self.labels))
        opt = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(len(X))
            total = 0.0
            for i in range(0, len(X), batch_size):
                idx = perm[i:i + batch_size]
                loss = F.cross_entropy(self.net(X[idx]), Y[idx])
                opt.zero_grad()
                loss.backward()          # backpropagation
                opt.step()               # SGD update
                total += loss.item() * len(idx)
            sched.step()
            if (ep + 1) % 10 == 0:
                print(f"  [FFN] epoch {ep + 1}/{epochs} loss={total / len(X):.4f}")
        self.net.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, texts):
        logits = self.net(torch.tensor(self._embed(texts)))
        return F.softmax(logits, dim=-1).numpy()

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.w2v.save(os.path.join(path, "word2vec.model"))
        torch.save({"state": self.net.state_dict(), "dim": self.dim, "hidden": self.hidden,
                    "labels": self.labels}, os.path.join(path, "ffn.pt"))

    @staticmethod
    def load(path):
        from gensim.models import Word2Vec
        ck = torch.load(os.path.join(path, "ffn.pt"), weights_only=False)
        m = W2VFeedForward(ck["labels"], ck["dim"], ck["hidden"])
        m.w2v = Word2Vec.load(os.path.join(path, "word2vec.model"))
        m.net = FFN(m.dim, m.hidden, len(m.labels))
        m.net.load_state_dict(ck["state"])
        m.net.eval()
        return m


# =================================================================== 3-5. BERT models
def quantize(model):
    from torch.ao.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig
    return torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear: default_dynamic_qconfig, nn.Embedding: float_qparams_weight_only_qconfig})


class BertClassifier:
    """Wraps a Hugging Face BERT with a sequence-classification head."""

    def __init__(self, name, model, tokenizer, labels, max_len=40):
        self.name, self.model, self.tok, self.labels, self.max_len = name, model, tokenizer, labels, max_len

    @staticmethod
    def from_pretrained(name, path_or_hub_id, labels, max_len=40):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path_or_hub_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            path_or_hub_id, num_labels=len(labels),
            id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
        return BertClassifier(name, model, tok, labels, max_len)

    def encode(self, texts):
        return self.tok(list(texts), padding=True, truncation=True, max_length=self.max_len,
                        return_tensors="pt")

    @torch.no_grad()
    def logits(self, texts, batch_size=128):
        self.model.eval()
        out = []
        for i in range(0, len(texts), batch_size):
            out.append(self.model(**self.encode(texts[i:i + batch_size])).logits.float())
        return torch.cat(out)

    def predict_proba(self, texts):
        return F.softmax(self.logits(texts), dim=-1).numpy()

    def fit(self, texts, y, epochs=3, lr=5e-5, batch_size=32, teacher_logits=None,
            temperature=2.0, alpha=0.5, seed=42):
        """Fine-tune. If teacher_logits is given, train with knowledge distillation:
        loss = alpha * CE(student, gold) + (1 - alpha) * T^2 * KL(student_T || teacher_T)."""
        from transformers import get_linear_schedule_with_warmup
        torch.manual_seed(seed)
        Y = torch.tensor([self.labels.index(l) for l in y])
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * ((len(texts) + batch_size - 1) // batch_size)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        for ep in range(epochs):
            self.model.train()
            perm = torch.randperm(len(texts)).tolist()
            total, n = 0.0, 0
            for i in range(0, len(texts), batch_size):
                idx = perm[i:i + batch_size]
                logits = self.model(**self.encode([texts[j] for j in idx])).logits
                loss = F.cross_entropy(logits, Y[idx])
                if teacher_logits is not None:
                    T = temperature
                    kd = F.kl_div(F.log_softmax(logits / T, -1),
                                  F.softmax(teacher_logits[idx] / T, -1),
                                  reduction="batchmean") * T * T
                    loss = alpha * loss + (1 - alpha) * kd
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                total += loss.item() * len(idx)
                n += len(idx)
                if (i // batch_size) % 100 == 0:
                    print(f"  [{self.name}] epoch {ep + 1}/{epochs} step {i // batch_size} "
                          f"loss={total / n:.4f}", flush=True)
            print(f"  [{self.name}] epoch {ep + 1} done, loss={total / n:.4f}", flush=True)
        self.model.eval()
        return self

    def quantized(self, name):
        """Post-training dynamic INT8 quantization (runs on CPU):
        * every Linear layer: INT8 weights, activations quantized on the fly
        * every Embedding table: weight-only 8-bit quantization (the word embeddings are
          most of a small BERT's size)"""
        qmodel = quantize(self.model.cpu().eval())
        return BertClassifier(name, qmodel, self.tok, self.labels, self.max_len)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)

    def save_quantized(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model_int8.pt"))
        self.model.config.save_pretrained(path)
        self.tok.save_pretrained(path)

    @staticmethod
    def load(name, path, labels, max_len=40):
        return BertClassifier.from_pretrained(name, path, labels, max_len)

    @staticmethod
    def load_quantized(name, int8_path, labels, max_len=40):
        """Rebuild the architecture from the saved config, quantize it, then load INT8 weights."""
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        config = AutoConfig.from_pretrained(int8_path)
        model = AutoModelForSequenceClassification.from_config(config).eval()
        qmodel = quantize(model)
        qmodel.load_state_dict(torch.load(os.path.join(int8_path, "model_int8.pt"), weights_only=False))
        tok = AutoTokenizer.from_pretrained(int8_path)
        return BertClassifier(name, qmodel.eval(), tok, labels, max_len)
