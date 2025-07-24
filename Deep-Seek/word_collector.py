import os
import re
import math
from pathlib import Path
import pickle
from collections import Counter, defaultdict

from openai import OpenAI
from sklearn.datasets import fetch_20newsgroups

# ──────────────────────────────────────────────────────────────────────────────
# DeepSeek client setup
# ──────────────────────────────────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = Path(__file__).resolve().parent
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# ──────────────────────────────────────────────────────────────────────────────
# Config & cache paths
# ──────────────────────────────────────────────────────────────────────────────
DOC_KWS_K        = 10
CLASS_CACHE      = BASE_DIR / "class_imp.pkl"
TEST_SCORES_CACHE = BASE_DIR / "test_scores.pkl"

# ──────────────────────────────────────────────────────────────────────────────
# Text utilities
# ──────────────────────────────────────────────────────────────────────────────
def char_entropy(word: str) -> float:
    counts = Counter(word)
    total = len(word)
    return -sum((c/total) * math.log((c/total) + 1e-12) for c in counts.values())

def clean_data(text: str) -> list[str]:
    text = re.sub(r"\W+", " ", text.lower())
    words = [w for w in text.split() if w.isalpha() and len(w) >= 2]
    return [w for w in words if 1.0 <= char_entropy(w) <= 4.0]

# ──────────────────────────────────────────────────────────────────────────────
# 1) Get & cache class_importances
# ──────────────────────────────────────────────────────────────────────────────
def build_and_cache_class_importances(train_docs, train_labels):
    if os.path.exists(CLASS_CACHE):
        return pickle.load(open(CLASS_CACHE, "rb"))

    num_classes = len(set(train_labels))
    class_imp = defaultdict(lambda: [0.0] * num_classes)

    for doc, cls in zip(train_docs, train_labels):
        # call DeepSeek
        messages = [
            {"role": "system", "content": "Output only lines of `term:score`."},
            {"role": "user",   "content": doc[:8000] + f"\n\nIdentify the {DOC_KWS_K} most generalizable keywords, one `term:score` per line."}
        ]
        resp = client.chat.completions.create(
            model="deepseek-chat", messages=messages, stream=False
        )
        for line in resp.choices[0].message.content.splitlines():
            if ":" not in line: continue
            term, scr = line.split(":",1)
            try:
                class_imp[term.strip()][cls] += float(scr)
            except ValueError:
                continue

    # normalize
    for term, vec in class_imp.items():
        s = sum(vec)
        if s>0: class_imp[term] = [v/s for v in vec]

    pickle.dump(class_imp, open(CLASS_CACHE, "wb"))
    return class_imp

# ──────────────────────────────────────────────────────────────────────────────
# 2) Get & cache test-doc term scores
# ──────────────────────────────────────────────────────────────────────────────
def score_and_cache_test_docs(test_docs, top_terms):
    if os.path.exists(TEST_SCORES_CACHE):
        return pickle.load(open(TEST_SCORES_CACHE, "rb"))

    test_scores = {}
    for idx, doc in enumerate(test_docs):
        tokens = clean_data(doc)
        candidates = [w for w in set(tokens) if w in top_terms]
        if not candidates:
            test_scores[idx] = []
            continue

        term_list = "\n".join(candidates)
        messages = [
            {"role":"system","content":"Output only lines of `term:score`."},
            {"role":"user","content": doc[:8000] + "\n\nScore each term 0.0–1.0, one `term:score` per line:\n\n" + term_list}
        ]
        resp = client.chat.completions.create(
            model="deepseek-chat", messages=messages, stream=False
        )
        scored = []
        for line in resp.choices[0].message.content.splitlines():
            if ":" not in line: continue
            term, scr = line.split(":",1)
            try:
                scored.append((term.strip(), float(scr)))
            except ValueError:
                continue
        test_scores[idx] = scored

    pickle.dump(test_scores, open(TEST_SCORES_CACHE, "wb"))
    return test_scores

# ──────────────────────────────────────────────────────────────────────────────
# Main: just build & save caches
# ──────────────────────────────────────────────────────────────────────────────
def main():
    data = fetch_20newsgroups(subset="all", remove=("headers","footers","quotes"))
    docs, labels = data.data, data.target
    split = int(0.8 * len(docs))
    train_docs, test_docs = docs[:split], docs[split:]
    train_lbls, test_lbls = labels[:split], labels[split:]

    class_imp = build_and_cache_class_importances(train_docs, train_lbls)
    top_terms = list(class_imp.keys())

    test_scores = score_and_cache_test_docs(test_docs, top_terms)

    print(f"Saved {len(class_imp)} class-importance terms to {CLASS_CACHE}")
    print(f"Saved scores for {len(test_scores)} test docs to {TEST_SCORES_CACHE}")

if __name__ == "__main__":
    main()

