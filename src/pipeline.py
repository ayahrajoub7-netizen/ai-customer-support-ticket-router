"""The routing pipeline used by both the evaluation and the app:

  raw message -> regex normalization + model-code extraction -> edit-distance spelling fix
              -> small-talk rules (ELIZA)  ... or ... -> intent model -> confidence check -> team
"""
import json
import os

from data import HUMAN_AGENT, OOS, label_list, team_for, INTENT_TO_DOMAIN
from preprocess import SpellCorrector, extract_model_codes, normalize, small_talk_reply

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")

MODEL_FILES = {  # display name -> folder inside models/
    "tf-idf + LogReg (baseline)": "tfidf_lr.joblib",
    "word2vec + FFN": "w2v_ffn",
    "BERT-base (FP32)": "bert_teacher",
    "BERT-mini distilled (FP32)": "bert_student",
    "BERT-mini distilled (INT8)": "bert_student_int8",
}
APP_MODEL = "BERT-mini distilled (INT8)"   # the quantized Transformer the app runs


class Preprocessor:
    def __init__(self, known_words):
        self.speller = SpellCorrector(known_words)

    def __call__(self, text):
        codes = extract_model_codes(text)
        norm = normalize(text)
        corrected, changes = self.speller.correct(norm)
        return {"normalized": norm, "text": corrected, "model_codes": codes, "corrections": changes}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(sorted(self.speller.known), f)

    @staticmethod
    def load(path):
        with open(path) as f:
            return Preprocessor(json.load(f))


def load_model(display_name, models_dir=MODELS_DIR):
    from models import BertClassifier, TfidfLogReg, W2VFeedForward
    labels = label_list()
    path = os.path.join(models_dir, MODEL_FILES[display_name])
    if display_name.startswith("tf-idf"):
        m = TfidfLogReg.load(path)
    elif display_name.startswith("word2vec"):
        m = W2VFeedForward.load(path)
    elif "INT8" in display_name:
        m = BertClassifier.load_quantized(display_name, path, labels)
    else:
        m = BertClassifier.load(display_name, path, labels)
    m.name = display_name
    return m


def load_thresholds(models_dir=MODELS_DIR):
    with open(os.path.join(models_dir, "thresholds.json")) as f:
        return json.load(f)


def route(message, pre, model, threshold):
    """Full pipeline for one message. Returns a dict that the UI displays."""
    p = pre(message)
    reply = small_talk_reply(p["normalized"])
    probs = model.predict_proba([p["text"]])[0]
    top = probs.argsort()[::-1][:3]
    labels = model.labels
    intent, conf = labels[top[0]], float(probs[top[0]])
    if reply is not None:
        team, status = TEAM_SMALL_TALK, "rule-based small-talk reply (ELIZA)"
    elif intent == OOS:
        team, status = HUMAN_AGENT, "out-of-scope -> human agent"
    elif conf < threshold:
        team, status = HUMAN_AGENT, f"low confidence ({conf:.2f} < {threshold:.2f}) -> human agent"
    else:
        team, status = team_for(intent), "routed automatically"
        if INTENT_TO_DOMAIN.get(intent) == "small_talk":
            reply = small_talk_reply_for_intent(intent)
    return {
        **p, "intent": intent, "confidence": conf, "team": team, "status": status, "reply": reply,
        "top3": [(labels[i], float(probs[i])) for i in top],
    }


TEAM_SMALL_TALK = "Chatbot (small talk)"


def small_talk_reply_for_intent(intent):
    canned = {
        "greeting": "Hello! How can I help you today?",
        "goodbye": "Goodbye! Have a nice day.",
        "thank_you": "You're welcome!",
        "are_you_a_bot": "I'm an automatic routing assistant.",
        "what_is_your_name": "I'm the support routing assistant.",
    }
    return canned.get(intent, "I'm just a routing assistant, but happy to chat! What do you need help with?")
