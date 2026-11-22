"""Area B - classical NLP:
  * regular-expression normalization and product model-code extraction
  * minimum edit distance (Levenshtein) spelling correction of product names
  * ELIZA-style rule-based replies for greetings / small talk
"""
import re

# ---------------------------------------------------------------- regex normalization
# letters + optional hyphen + 3-5 digits, e.g. DW-6534, WM1200, rf550x
MODEL_CODE_RE = re.compile(r"\b([A-Za-z]{1,4})-?(\d{3,5}[A-Za-z]?)\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
REPEAT_RE = re.compile(r"(\w)\1{2,}")          # "pleeeease" -> "pleease"
NON_TEXT_RE = re.compile(r"[^a-z0-9' ]+")
SPACES_RE = re.compile(r"\s+")
CONTRACTIONS = {"won't": "will not", "can't": "can not", "n't": " not", "'re": " are", "'m": " am"}

# words that look like a model code but are not one
NOT_MODEL_CODES = {"e", "f", "error", "code", "pm", "am"}


def extract_model_codes(text):
    """Find product model codes such as 'DW-6534', 'WM 1200', 'rf550'."""
    codes = []
    for prefix, number in MODEL_CODE_RE.findall(text):
        if prefix.lower() in NOT_MODEL_CODES:
            continue
        codes.append(f"{prefix.upper()}-{number.upper()}")
    return codes


def normalize(text):
    """Lower-case, remove URLs/e-mails/model codes, collapse repeated letters and punctuation."""
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = EMAIL_RE.sub(" ", t)
    t = MODEL_CODE_RE.sub(lambda m: " " if m.group(1) not in NOT_MODEL_CODES else m.group(0), t)
    t = t.replace("’", "'")
    for k, v in CONTRACTIONS.items():
        t = t.replace(k, v)
    t = REPEAT_RE.sub(r"\1\1", t)
    t = NON_TEXT_RE.sub(" ", t)
    return SPACES_RE.sub(" ", t).strip()


# ---------------------------------------------------------------- minimum edit distance
PRODUCT_LEXICON = [
    "dishwasher", "washer", "washing", "machine", "refrigerator", "fridge", "freezer",
    "oven", "microwave", "dryer", "cooktop", "dispenser", "detergent", "compressor",
    "warranty", "technician", "appliance", "laundry", "spin", "drain", "leaking",
]


def edit_distance(a, b, sub_cost=1):
    """Levenshtein minimum edit distance with dynamic programming (insert=delete=1)."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else sub_cost
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m]


class SpellCorrector:
    """Replace unknown words with the closest product-lexicon word (distance <= 2)."""

    def __init__(self, known_words=(), lexicon=PRODUCT_LEXICON, max_dist=2):
        self.known = set(known_words) | set(lexicon)
        self.lexicon = lexicon
        self.max_dist = max_dist

    def correct_word(self, w):
        if w in self.known or len(w) < 4 or w.isdigit():
            return w
        best, best_d = w, self.max_dist + 1
        for cand in self.lexicon:
            if abs(len(cand) - len(w)) > self.max_dist:
                continue
            dist = edit_distance(w, cand)
            if dist < best_d:
                best, best_d = cand, dist
        return best

    def correct(self, text):
        words = text.split()
        fixed = [self.correct_word(w) for w in words]
        changes = [(a, b) for a, b in zip(words, fixed) if a != b]
        return " ".join(fixed), changes


# ---------------------------------------------------------------- ELIZA-style small talk
ELIZA_RULES = [
    (re.compile(r"^(hi|hello|hey|merhaba|selam|good (morning|afternoon|evening))\b"),
     "Hello! Please describe your problem and I will route you to the right team."),
    (re.compile(r"\bhow are you\b"), "I'm fine, thank you. How can I help you with your request?"),
    (re.compile(r"\b(thanks|thank you|tesekkurler|cheers)\b"), "You're welcome! Is there anything else I can help with?"),
    (re.compile(r"^(bye|goodbye|see you|hosca kal)\b"), "Goodbye! Have a nice day."),
    (re.compile(r"\bare you (a )?(bot|robot|human)\b"), "I'm an automatic routing assistant. A human agent can take over at any time."),
    (re.compile(r"\bi (feel|am) (\w+)"), "Why do you feel {1}? Tell me about the problem and I'll find the right team."),
]
SMALL_TALK_DEFAULT = "Nice to chat! Please tell me about your request so I can route it."


def small_talk_reply(norm_text):
    """Return a rule-based reply if the message is greeting/small talk, else None."""
    if len(norm_text.split()) > 6:  # long messages are real requests
        return None
    for pattern, template in ELIZA_RULES:
        m = pattern.search(norm_text)
        if m:
            return template.format(*m.groups())
    return None


if __name__ == "__main__":
    msg = "Hiii, my DISHWASER is not drainnnning!!! model DW-6534, see www.example.com"
    print("model codes:", extract_model_codes(msg))
    n = normalize(msg)
    print("normalized :", n)
    print("corrected  :", SpellCorrector().correct(n))
    print("edit_distance('dishwaser','dishwasher') =", edit_distance("dishwaser", "dishwasher"))
    print("small talk :", small_talk_reply(normalize("Hello there")))
